"""
Remote Training Service — SSH-based LLaMA-Factory training runner.

Executes LoRA finetuning on remote GPU server via paramiko SSH,
produces LoRA adapter, and starts vLLM inference server for evaluation.
"""

import asyncio
import json
import logging
import os
import time
import tempfile
from typing import Optional

import paramiko

logger = logging.getLogger(__name__)

# Remote server config (from .env)
SSH_HOST = os.getenv("TRAINING_SSH_HOST", os.getenv("SSH_HOST", "10.242.33.21"))
SSH_PORT = int(os.getenv("TRAINING_SSH_PORT", "22"))
SSH_USER = os.getenv("TRAINING_SSH_USER", "user")
SSH_PASSWORD = os.getenv("TRAINING_SSH_PASSWORD", "Server123.")
CONDA_ENV = os.getenv("TRAINING_CONDA_ENV", "wangshuo_agent")

# Remote paths
REMOTE_LLAMAFACTORY_DIR = "/home/user/workspace/wangshuo/LLaMa-Factory"
REMOTE_WORKSPACE = "/home/user/workspace/wangshuo/imts_training"
REMOTE_VLLM_PORT = 8001

# Default training params — use remote server local model path
DEFAULT_BASE_MODEL = "/home/user/workspace/wangshuo/Models/Qwen3-8B"
DEFAULT_LORA_RANK = 8
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 3
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRADIENT_ACCUMULATION = 4
DEFAULT_CUTOFF_LEN = 2048

# ModelScope model ID for downloading Qwen3-8B if not present locally
MODELSCOPE_MODEL_IDS = {
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
}


def _get_ssh_client() -> paramiko.SSHClient:
    """Create and return a paramiko SSH client."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=10)
    return client


def _run_remote_command(cmd: str, timeout: int = 600) -> tuple:
    """Run a command on the remote server via SSH.

    Returns (stdout, stderr, exit_code).
    """
    client = _get_ssh_client()
    try:
        # Prepend conda activation
        full_cmd = (
            f"source /home/user/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate {CONDA_ENV} && "
            f"export DISABLE_VERSION_CHECK=1 && "
            f"export TORCH_CUDA_ARCH_LIST=12.0 && "
            f"{cmd}"
        )
        stdin, stdout, stderr = client.exec_command(full_cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        return out, err, exit_code
    finally:
        client.close()


def _generate_yaml_config(
    job_id: str,
    iteration: int,
    base_model: str,
    dataset_remote_path: str,
    lora_rank: int = DEFAULT_LORA_RANK,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gradient_accumulation: int = DEFAULT_GRADIENT_ACCUMULATION,
    cutoff_len: int = DEFAULT_CUTOFF_LEN,
) -> str:
    """Generate LLaMA-Factory YAML config for LoRA SFT training."""
    output_dir = f"{REMOTE_WORKSPACE}/{job_id}/iter_{iteration}/lora_output"

    config = {
        "model_name_or_path": base_model,
        "trust_remote_code": True,
        # method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_rank,
        "lora_target": "all",
        # dataset
        "dataset": "imts_custom",
        "dataset_dir": f"{REMOTE_WORKSPACE}/{job_id}/iter_{iteration}",
        "template": "qwen3",
        "cutoff_len": cutoff_len,
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        # output
        "output_dir": output_dir,
        "logging_steps": 1,
        "save_steps": 200,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "report_to": "none",
        # train
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
        "resume_from_checkpoint": None,
    }

    # Multi-GPU: use DeepSpeed ZeRO-2 for 2 GPUs
    config["deepspeed"] = f"{REMOTE_WORKSPACE}/ds_config_zero2.json"

    lines = []
    for k, v in config.items():
        if isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, float):
            lines.append(f"{k}: {v}")
        elif v is None:
            lines.append(f"{k}: null")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _generate_deepspeed_config() -> str:
    """Generate DeepSpeed ZeRO-2 config for 2-GPU training (LLaMA-Factory compatible)."""
    return json.dumps({
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            "round_robin_gradients": True,
        },
    }, indent=2)


MIN_FREE_GPU_MEM_MB = 10000  # Minimum free GPU memory (MB) per GPU for training


def _detect_available_gpus() -> list[int]:
    """Detect GPU indices with >= MIN_FREE_GPU_MEM_MB free memory.

    Returns list of GPU indices that have sufficient free memory.
    Falls back to [0, 1] if query fails.
    """
    cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null"
    out, err, code = _run_remote_command(cmd, timeout=10)
    if code != 0 or not out.strip():
        logger.warning(f"Failed to query GPU status, falling back to [0,1]: {err}")
        return [0, 1]

    available = []
    for line in out.strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) == 2:
            try:
                idx = int(parts[0].strip())
                free_mb = int(parts[1].strip())
                if free_mb >= MIN_FREE_GPU_MEM_MB:
                    available.append(idx)
                    logger.info(f"GPU {idx}: {free_mb} MiB free — available for training")
                else:
                    logger.info(f"GPU {idx}: {free_mb} MiB free (< {MIN_FREE_GPU_MEM_MB}) — skipping")
            except ValueError:
                continue

    return available


def _preflight_check(available_gpus: list[int]) -> str | None:
    """Pre-flight validation before training. Returns error message or None if OK."""
    if len(available_gpus) < 1:
        return f"No GPU available with >= {MIN_FREE_GPU_MEM_MB} MiB free memory. Available GPUs: {available_gpus}"

    # Check CUDA driver
    cmd = "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1"
    out, _, code = _run_remote_command(cmd, timeout=10)
    if code != 0 or not out.strip():
        return "nvidia-smi not responding — CUDA driver may not be installed"

    logger.info(f"Pre-flight check passed: driver={out.strip()}, available GPUs={available_gpus}")
    return None


def _generate_dataset_info(dataset_remote_path: str) -> str:
    """Generate LLaMA-Factory dataset_info.json for custom dataset."""
    return json.dumps({
        "imts_custom": {
            "file_name": dataset_remote_path,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }, indent=2, ensure_ascii=False)


def _transfer_file(local_content: str, remote_path: str) -> bool:
    """Write string content to remote server via SFTP."""
    client = _get_ssh_client()
    try:
        sftp = client.open_sftp()
        try:
            # Ensure parent directory exists
            parent = os.path.dirname(remote_path).replace("\\", "/")
            _run_remote_command(f"mkdir -p {parent}")
            with sftp.file(remote_path, "w") as f:
                f.write(local_content)
            logger.info(f"Transferred file to remote: {remote_path}")
            return True
        finally:
            sftp.close()
    except Exception as e:
        logger.error(f"Failed to transfer file to {remote_path}: {e}")
        return False
    finally:
        client.close()


def _prepare_remote_workspace(job_id: str, iteration: int) -> str:
    """Create remote workspace directory structure."""
    base = f"{REMOTE_WORKSPACE}/{job_id}/iter_{iteration}"
    _run_remote_command(f"mkdir -p {base}")
    return base


def _upload_dataset_to_remote(local_path: str, job_id: str, iteration: int) -> str:
    """Upload dataset file to remote server for training.

    Returns the remote path of the uploaded file.
    """
    remote_base = _prepare_remote_workspace(job_id, iteration)
    filename = os.path.basename(local_path)
    remote_path = f"{remote_base}/{filename}"

    client = _get_ssh_client()
    try:
        sftp = client.open_sftp()
        try:
            sftp.put(local_path, remote_path)
            logger.info(f"Uploaded dataset: {local_path} -> {remote_path}")
            return remote_path
        finally:
            sftp.close()
    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise
    finally:
        client.close()


def _convert_dataset_to_alpaca_format(dataset_path: str) -> str:
    """Convert CSV/JSONL dataset to Alpaca-format JSON for LLaMA-Factory.

    Returns local path of converted file.
    """
    import pandas as pd

    if dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
        df = pd.read_json(dataset_path, lines=dataset_path.endswith(".jsonl"))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")

    alpaca_data = []
    for _, row in df.iterrows():
        instruction = str(row.get("instruction", row.get("question", row.get("input", ""))))
        input_text = str(row.get("input", row.get("context", "")))
        output = str(row.get("output", row.get("answer", row.get("response", row.get("content", "")))))

        if not instruction.strip() and not output.strip():
            continue

        alpaca_data.append({
            "instruction": instruction,
            "input": input_text if input_text and input_text != "nan" else "",
            "output": output,
        })

    # Write to temp file
    temp_dir = tempfile.mkdtemp(prefix="imts_alpaca_")
    converted_path = os.path.join(temp_dir, "alpaca_train.json")
    with open(converted_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Converted {len(alpaca_data)} samples to Alpaca format: {converted_path}")
    return converted_path


def _stop_vllm_server():
    """Kill vLLM inference server on remote GPU to free GPU memory."""
    try:
        _run_remote_command(
            f"pkill -f 'vllm serve' 2>/dev/null; pkill -f 'VLLM::EngineCore' 2>/dev/null; sleep 2",
            timeout=15
        )
        logger.info("vLLM server stopped, GPU memory freed")
    except Exception as e:
        logger.warning(f"Failed to stop vLLM server: {e}")


def _is_vllm_running() -> bool:
    """Check if vLLM inference server is already running on remote."""
    out, err, code = _run_remote_command(
        f"curl -sf http://localhost:{REMOTE_VLLM_PORT}/v1/models", timeout=10
    )
    return '"data"' in out and code == 0


def _start_vllm_server(model_path: str, lora_path: str = None) -> bool:
    """Start vLLM inference server on remote GPU.

    Args:
        model_path: Base model path or name (e.g., "Qwen/Qwen3-8B")
        lora_path: Optional LoRA adapter path for served model

    Returns True if server started successfully.
    """
    # Kill any existing vLLM server
    _run_remote_command(
        f"pkill -f 'vllm serve' 2>/dev/null; sleep 1", timeout=15
    )

    # Write a startup script to remote and execute it (avoids SSH channel timeout from conda activate + nohup)
    vllm_cmd = f"vllm serve {model_path}"
    if lora_path:
        vllm_cmd += f" --enable-lora --lora-modules imts-lora={lora_path}"
    vllm_cmd += f" --port {REMOTE_VLLM_PORT} --dtype bfloat16 --max-lora-rank 16"
    vllm_cmd += f" --gpu-memory-utilization 0.85 --attention-backend FLEX_ATTENTION"

    script_content = (
        f"#!/bin/bash\n"
        f"source /home/user/miniconda3/etc/profile.d/conda.sh\n"
        f"conda activate {CONDA_ENV}\n"
        f"export DISABLE_VERSION_CHECK=1\n"
        f"export TORCH_CUDA_ARCH_LIST=12.0\n"
        f"nohup {vllm_cmd} > {REMOTE_WORKSPACE}/vllm.log 2>&1 &\n"
        f"echo $!\n"
    )
    script_path = f"{REMOTE_WORKSPACE}/start_vllm.sh"
    launch_client = paramiko.SSHClient()
    launch_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    launch_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=10)
    try:
        sftp = launch_client.open_sftp()
        with sftp.open(script_path, "w") as f:
            f.write(script_content)
        sftp.close()
        stdin, stdout, stderr = launch_client.exec_command(f"chmod +x {script_path} && bash {script_path}", timeout=15)
        out = stdout.read().decode("utf-8", errors="replace").strip()
        launch_client.close()
        logger.info(f"vLLM launched with PID: {out}")
    except Exception as launch_err:
        logger.warning(f"vLLM launch script upload/execute had issues: {launch_err}. vLLM may still be starting in background.")
        try:
            launch_client.close()
        except:
            pass

    # Wait for vLLM to become ready (up to 360s — FLEX_ATTENTION + LoRA needs torch.compile which takes ~150s)
    # Use a single persistent SSH connection for the polling loop to avoid connection exhaustion
    poll_client = paramiko.SSHClient()
    poll_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    poll_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=10)

    for attempt in range(72):
        time.sleep(5)
        try:
            stdin, stdout, stderr = poll_client.exec_command(
                f"curl -sf http://localhost:{REMOTE_VLLM_PORT}/v1/models",
                timeout=10
            )
            out = stdout.read().decode("utf-8", errors="replace")
            exit_code = stdout.channel.recv_exit_status()
            if '"data"' in out and exit_code == 0:
                logger.info(f"vLLM server started successfully on port {REMOTE_VLLM_PORT} (attempt {attempt+1})")
                poll_client.close()
                return True
        except Exception as poll_err:
            logger.warning(f"vLLM polling SSH error at attempt {attempt+1}: {poll_err}")
            # Reconnect if SSH session broke
            try:
                poll_client.close()
            except:
                pass
            poll_client = paramiko.SSHClient()
            poll_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            poll_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=10)

        if attempt % 6 == 0:
            logger.info(f"Waiting for vLLM server... attempt {attempt + 1}/{72}")

    poll_client.close()
    logger.error("vLLM server failed to start within 360s")
    return False


class TrainingResult:
    """Container for training results."""

    def __init__(
        self,
        success: bool,
        output_dir: str = "",
        final_loss: float = 0.0,
        loss_history: list = None,
        epochs: int = 0,
        learning_rate: float = 0.0,
        batch_size: int = 0,
        lora_path: str = "",
        error: str = "",
        training_time_seconds: float = 0.0,
        vllm_ready: bool = False,
    ):
        self.success = success
        self.output_dir = output_dir
        self.final_loss = final_loss
        self.loss_history = loss_history or []
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lora_path = lora_path
        self.error = error
        self.training_time_seconds = training_time_seconds
        self.vllm_ready = vllm_ready

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "model_name": DEFAULT_BASE_MODEL,
            "output_dir": self.output_dir,
            "final_loss": self.final_loss,
            "loss_history": self.loss_history,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "lora_path": self.lora_path,
            "error": self.error,
            "training_time_seconds": self.training_time_seconds,
            "vllm_ready": self.vllm_ready,
        }


def run_remote_training(
    job_id: str,
    iteration: int,
    dataset_path: str,
    base_model: str = DEFAULT_BASE_MODEL,
    lora_rank: int = DEFAULT_LORA_RANK,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gradient_accumulation: int = DEFAULT_GRADIENT_ACCUMULATION,
    sync_redis_client=None,
) -> TrainingResult:
    """Execute real LoRA training on remote GPU server via SSH.

    Full pipeline:
    1. Convert dataset to Alpaca format
    2. Upload dataset + config to remote server
    3. Run LLaMA-Factory `run_exp()` via SSH
    4. Parse training logs for loss history
    5. Start vLLM server with LoRA adapter for evaluation
    6. Return TrainingResult with all metadata

    This runs in a thread pool to avoid blocking the async event loop.
    """
    start_time = time.time()
    output_dir = f"{REMOTE_WORKSPACE}/{job_id}/iter_{iteration}/lora_output"

    try:
        # 1. Convert dataset to Alpaca format
        logger.info(f"[Training] Converting dataset to Alpaca format: {dataset_path}")
        alpaca_path = _convert_dataset_to_alpaca_format(dataset_path)

        # 2. Prepare remote workspace
        remote_base = _prepare_remote_workspace(job_id, iteration)

        # 3. Upload dataset
        remote_dataset_path = _upload_dataset_to_remote(alpaca_path, job_id, iteration)
        # Use just the filename for LLaMA-Factory dataset_dir lookup
        dataset_filename = os.path.basename(remote_dataset_path)

        # 4. Upload dataset_info.json
        dataset_info = _generate_dataset_info(dataset_filename)
        dataset_info_path = f"{remote_base}/dataset_info.json"
        _transfer_file(dataset_info, dataset_info_path)

        # 5. Detect available GPUs + pre-flight check
        available_gpus = _detect_available_gpus()
        preflight_error = _preflight_check(available_gpus)
        if preflight_error:
            logger.error(f"[Training] Pre-flight check failed: {preflight_error}")
            return TrainingResult(
                success=False,
                error=preflight_error,
                training_time_seconds=time.time() - start_time,
            )

        # 6. Configure training — use single-GPU mode (DeepSpeed fails on this server)
        # Pick GPU with most free memory
        best_gpu = available_gpus[0]
        gpu_free_mems = {}
        for g in available_gpus:
            # Re-query to get current free memory for each GPU
            mem_cmd = f"nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i {g}"
            mem_out, _, _ = _run_remote_command(mem_cmd, timeout=10)
            try:
                free_mb = int(mem_out.strip().split(",")[1].strip())
                gpu_free_mems[g] = free_mb
            except (ValueError, IndexError):
                gpu_free_mems[g] = 0

        if gpu_free_mems:
            best_gpu = max(gpu_free_mems, key=gpu_free_mems.get)
        logger.info(f"[Training] GPU free memory: {gpu_free_mems}, selecting GPU {best_gpu}")

        single_batch = min(batch_size, 1)
        single_accum = gradient_accumulation * batch_size  # compensate for smaller batch
        config_yaml = _generate_yaml_config(
            job_id=job_id, iteration=iteration, base_model=base_model,
            dataset_remote_path=dataset_filename, lora_rank=lora_rank,
            learning_rate=learning_rate, num_epochs=num_epochs,
            batch_size=single_batch, gradient_accumulation=single_accum,
        )
        # Remove deepspeed key for single-GPU mode
        ds_line = f"deepspeed: {REMOTE_WORKSPACE}/ds_config_zero2.json"
        config_yaml = config_yaml.replace(ds_line, "")
        yaml_remote_path = f"{remote_base}/train_config.yaml"
        _transfer_file(config_yaml, yaml_remote_path)

        gpu_str = str(best_gpu)

        # Downgrade transformers/tokenizers for llamafactory compatibility (4.55.0)
        logger.info("[Training] Downgrading transformers to 4.55.0 for llamafactory compatibility...")
        downgrade_cmd = "pip install transformers==4.55.0 tokenizers==0.21.1 --no-deps -q"
        out_dw, err_dw, code_dw = _run_remote_command(downgrade_cmd, timeout=120)
        if code_dw != 0:
            logger.warning(f"[Training] Transformers downgrade failed: {err_dw[:200]}")

        train_cmd = (
            f"cd {REMOTE_LLAMAFACTORY_DIR} && "
            f"CUDA_VISIBLE_DEVICES={gpu_str} llamafactory-cli train {yaml_remote_path}"
        )
        logger.info(f"[Training] Single-GPU mode: GPU={gpu_str}, batch=1, accum={single_accum}")

        # 7. Run training — streaming mode if sync_redis_client provided
        logger.info(f"[Training] Starting remote training: {base_model}, epochs={num_epochs}")
        logger.info(f"[Training] Command: llamafactory-cli train {yaml_remote_path}")

        if sync_redis_client:
            # Streaming mode: nohup background launch + poll trainer_log.jsonl
            exit_code, loss_history, was_stopped = _run_remote_training_streaming(
                job_id=job_id, iteration=iteration, gpu_str=gpu_str,
                yaml_remote_path=yaml_remote_path, output_dir=output_dir,
                num_epochs=num_epochs, sync_redis_client=sync_redis_client,
            )
            if exit_code is None:
                # Streaming launch failed, fallback to blocking mode
                logger.warning("[Training] Streaming launch failed, falling back to blocking mode")
                out, err, exit_code = _run_remote_command(train_cmd, timeout=1800)
                loss_history = None
                was_stopped = False
            else:
                out = ""  # Not available in streaming mode
                err = ""
        else:
            # Fallback: blocking mode (original behavior)
            out, err, exit_code = _run_remote_command(train_cmd, timeout=1800)
            loss_history = None
            was_stopped = False

        training_time = time.time() - start_time

        # Upgrade transformers back to 4.56.2 for vLLM compatibility
        logger.info("[Training] Upgrading transformers back to 4.56.2 for vLLM...")
        upgrade_cmd = "pip install transformers==4.56.2 tokenizers==0.22.0 --no-deps -q"
        out_up, err_up, code_up = _run_remote_command(upgrade_cmd, timeout=120)

        if was_stopped:
            logger.info(f"[Training] Training was stopped by user after {training_time:.1f}s")
            return TrainingResult(
                success=False,
                error="Training stopped by user",
                training_time_seconds=training_time,
            )

        if exit_code != 0:
            logger.error(f"[Training] Remote training failed: exit_code={exit_code}")
            logger.error(f"[Training] stderr: {err[:500]}")
            return TrainingResult(
                success=False,
                error=f"Training failed (exit_code={exit_code}): {err[:200]}",
                training_time_seconds=training_time,
            )

        # 8. Parse loss history
        if not loss_history:
            # Either blocking mode or streaming mode found no losses
            loss_history = _parse_loss_from_trainer_state(output_dir)
            if not loss_history:
                loss_history = _parse_loss_from_logs(out + err)
        final_loss = loss_history[-1] if loss_history else 0.0

        # 9. Verify LoRA adapter was produced
        lora_check_out, _, _ = _run_remote_command(f"ls {output_dir}/adapter_model.safetensors 2>/dev/null && echo EXISTS || echo MISSING", timeout=10)
        lora_exists = "EXISTS" in lora_check_out

        lora_path = output_dir if lora_exists else ""

        if not lora_exists:
            # Check checkpoint dirs
            ckpt_out, _, _ = _run_remote_command(f"ls -d {output_dir}/checkpoint-* 2>/dev/null | tail -1", timeout=10)
            if ckpt_out.strip():
                lora_path = ckpt_out.strip()
                logger.info(f"[Training] Using checkpoint LoRA: {lora_path}")
            else:
                logger.error(f"[Training] No LoRA adapter found at {output_dir}")

        logger.info(
            f"[Training] Complete: loss={final_loss:.4f}, "
            f"time={training_time:.1f}s, lora={lora_path}"
        )

        # 10. Start vLLM server with LoRA adapter for evaluation
        vllm_ready = False
        if lora_path:
            try:
                logger.info(f"[Training] Starting vLLM with LoRA adapter: {lora_path}")
                vllm_started = _start_vllm_server(base_model, lora_path)
                if vllm_started:
                    vllm_ready = True
                    logger.info("[Training] vLLM server started successfully with LoRA adapter")
                else:
                    logger.error("[Training] vLLM server failed to start — evaluation cannot proceed without trained model")
            except Exception as vllm_err:
                logger.error(f"[Training] vLLM startup error: {vllm_err}")

        return TrainingResult(
            success=True,
            output_dir=output_dir,
            final_loss=final_loss,
            loss_history=loss_history,
            epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            lora_path=lora_path,
            training_time_seconds=training_time,
            vllm_ready=vllm_ready,
        )

    except Exception as e:
        logger.error(f"[Training] Exception: {e}", exc_info=True)
        return TrainingResult(
            success=False,
            error=str(e),
            training_time_seconds=time.time() - start_time,
        )


def _parse_loss_from_trainer_state(output_dir: str) -> list:
    """Parse loss values from LLaMA-Factory's trainer_log.jsonl on the remote server.

    LLaMA-Factory writes trainer_log.jsonl (JSONL format with one JSON per step),
    NOT trainer_state.json (which is only produced by vanilla HuggingFace Trainer).
    """
    log_path = f"{output_dir}/trainer_log.jsonl"
    out, err, code = _run_remote_command(f"cat {log_path}", timeout=10)

    if code != 0 or not out.strip():
        logger.warning(f"[Training] Could not read trainer_log.jsonl: {err[:200]}")
        return []

    try:
        import json
        losses = []
        for line in out.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                loss_val = entry.get("loss")
                if loss_val is not None:
                    losses.append(float(loss_val))
            except json.JSONDecodeError:
                continue
        if losses:
            logger.info(f"[Training] Parsed {len(losses)} loss values from trainer_log.jsonl")
        return losses
    except ValueError as e:
        logger.warning(f"[Training] Failed to parse trainer_log.jsonl: {e}")
        return []


def _parse_loss_from_logs(log_text: str) -> list:
    """Parse training loss values from LLaMA-Factory CLI log output (fallback)."""
    import re

    losses = []
    # Pattern: "{'loss': 0.1234, ...}" or "loss=0.1234"
    for match in re.finditer(r"'loss':\s*([\d.]+)", log_text):
        losses.append(float(match.group(1)))
    for match in re.finditer(r"loss[=:]\s*([\d.]+)", log_text):
        val = float(match.group(1))
        if val not in losses:
            losses.append(val)

    return losses


def emit_training_loss_sync(
    job_id: str, epoch: int, step: int, loss: float,
    loss_history: list, sync_redis_client
):
    """Emit a TRAINING_LOSS message from a synchronous thread.

    Uses sync Redis publish for real-time SSE delivery and rpush for
    history persistence. No async context needed.
    """
    from message_types import IMTSMessage

    msg = IMTSMessage(
        msg_type="TRAINING_LOSS",
        job_id=job_id,
        stage="TRAINING",
        timestamp=int(time.time() * 1000),
        progress=min(90, step * 10),
        data={"epoch": epoch, "step": step, "loss": loss, "loss_history": loss_history}
    )
    msg_json = msg.to_json()
    channel = f"job_events:{job_id}"
    sync_redis_client.publish(channel, msg_json)
    sync_redis_client.rpush(f"imts_messages:{job_id}", msg_json)
    sync_redis_client.expire(f"imts_messages:{job_id}", 3600)
    logger.debug(f"[Training] Emitted TRAINING_LOSS step={step} loss={loss:.4f}")


def _run_remote_training_streaming(
    job_id: str,
    iteration: int,
    gpu_str: str,
    yaml_remote_path: str,
    output_dir: str,
    num_epochs: int,
    sync_redis_client=None,
) -> tuple:
    """Run training as a background process and stream loss via polling.

    Uses the nohup + polling pattern (like _start_vllm_server):
    1. Upload launch script → get PID
    2. Poll trainer_log.jsonl every 5s for new loss entries
    3. Emit TRAINING_LOSS messages via sync Redis
    4. Check Redis stop flag — kill remote process if set
    5. Return (exit_code, loss_history, was_stopped) on completion

    Returns (exit_code, loss_history_list, was_stopped_bool).
    """
    # A. Upload background launch script to remote server (SFTP)
    log_path = f"{output_dir}/train_console.log"
    script_content = (
        f"#!/bin/bash\n"
        f"source /home/user/miniconda3/etc/profile.d/conda.sh\n"
        f"conda activate {CONDA_ENV}\n"
        f"export DISABLE_VERSION_CHECK=1\n"
        f"export TORCH_CUDA_ARCH_LIST=12.0\n"
        f"mkdir -p {output_dir}\n"
        f"cd {REMOTE_LLAMAFACTORY_DIR}\n"
        f"CUDA_VISIBLE_DEVICES={gpu_str} nohup llamafactory-cli train {yaml_remote_path} > {log_path} 2>&1 &\n"
        f"echo $!\n"
    )
    script_path = f"{REMOTE_WORKSPACE}/start_training.sh"

    launch_client = paramiko.SSHClient()
    launch_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    launch_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=10)
    try:
        sftp = launch_client.open_sftp()
        with sftp.open(script_path, "w") as f:
            f.write(script_content)
        sftp.close()

        # Execute launch script (returns PID immediately)
        stdin, stdout, stderr = launch_client.exec_command(
            f"chmod +x {script_path} && bash {script_path}", timeout=15
        )
        pid_output = stdout.read().decode("utf-8", errors="replace").strip()
        launch_err = stderr.read().decode("utf-8", errors="replace").strip()
        if launch_err:
            logger.warning(f"[Training] Launch script stderr: {launch_err[:200]}")
        try:
            pid = int(pid_output.strip().split("\n")[-1])
            logger.info(f"[Training] Training process launched with PID: {pid}")
        except (ValueError, IndexError):
            logger.warning(f"[Training] Could not parse PID from: {pid_output}, falling back to blocking mode")
            launch_client.close()
            return None, [], False
    except Exception as launch_err:
        logger.warning(f"[Training] Launch script failed: {launch_err}. Falling back to blocking mode.")
        try:
            launch_client.close()
        except Exception:
            pass
        return None, [], False
    finally:
        try:
            launch_client.close()
        except Exception:
            pass

    # B. Create persistent SSH connection for polling
    poll_client = paramiko.SSHClient()
    poll_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    poll_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=10)

    # C. Streaming polling loop
    last_parsed_count = 0
    accumulated_losses = []
    was_stopped = False
    max_poll_attempts = 360  # 5s * 360 = 1800s = 30 min
    trainer_log_path = f"{output_dir}/trainer_log.jsonl"

    for attempt in range(max_poll_attempts):
        time.sleep(5)

        # Check Redis stop flag
        if sync_redis_client:
            stop_key = f"imts_stop:{job_id}"
            if sync_redis_client.exists(stop_key):
                logger.info(f"[Training] Stop signal detected for job {job_id}, killing remote process {pid}")
                try:
                    poll_client.exec_command(f"kill {pid}", timeout=10)
                    # Also kill any child processes
                    poll_client.exec_command(f"pkill -P {pid} 2>/dev/null", timeout=5)
                except Exception:
                    pass
                was_stopped = True
                poll_client.close()
                return -1, accumulated_losses, True

        # Check if training process is still alive
        process_alive = False
        try:
            stdin, stdout, stderr = poll_client.exec_command(
                f"ps -p {pid} -o pid= 2>/dev/null", timeout=10
            )
            ps_out = stdout.read().decode("utf-8", errors="replace").strip()
            process_alive = str(pid) in ps_out
        except Exception:
            # Reconnect on SSH failure
            try:
                poll_client.close()
            except Exception:
                pass
            poll_client = paramiko.SSHClient()
            poll_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            poll_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=10)
            # Retry the check after reconnect
            try:
                stdin, stdout, stderr = poll_client.exec_command(
                    f"ps -p {pid} -o pid= 2>/dev/null", timeout=10
                )
                ps_out = stdout.read().decode("utf-8", errors="replace").strip()
                process_alive = str(pid) in ps_out
            except Exception:
                process_alive = True  # last resort: assume alive

        # Parse trainer_log.jsonl for new loss entries (JSONL format)
        try:
            stdin, stdout, stderr = poll_client.exec_command(
                f"cat {trainer_log_path} 2>/dev/null", timeout=10
            )
            log_content = stdout.read().decode("utf-8", errors="replace").strip()
            if log_content:
                try:
                    import json
                    lines = log_content.split("\n")
                    new_entries = lines[last_parsed_count:]
                    for line in new_entries:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            loss_val = entry.get("loss")
                            if loss_val is not None:
                                loss_val = float(loss_val)
                                accumulated_losses.append(loss_val)
                                current_step = entry.get("current_steps", len(accumulated_losses))
                                epoch_val = entry.get("epoch", 0)
                                # epoch is fractional (e.g. 0.018 = 1.8% through epoch 1)
                                epoch_num = int(epoch_val) + 1

                                if sync_redis_client:
                                    emit_training_loss_sync(
                                        job_id, epoch_num, current_step, loss_val,
                                        accumulated_losses, sync_redis_client
                                    )

                            last_parsed_count += 1
                        except json.JSONDecodeError:
                            last_parsed_count += 1  # skip malformed lines
                            continue

                    if accumulated_losses and attempt % 6 == 0:  # Log every ~30s
                        logger.info(
                            f"[Training] Polling attempt {attempt+1}: "
                            f"process_alive={process_alive}, "
                            f"losses_parsed={len(accumulated_losses)}, "
                            f"latest_loss={accumulated_losses[-1]:.4f}"
                        )
                except ValueError:
                    pass
        except Exception:
            # Reconnect on SSH failure
            try:
                poll_client.close()
            except Exception:
                pass
            poll_client = paramiko.SSHClient()
            poll_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            poll_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=10)

        # If process exited, break the loop
        if not process_alive:
            logger.info(f"[Training] Process {pid} has exited (attempt {attempt+1})")

            # Determine exit code from the log file
            try:
                stdin, stdout, stderr = poll_client.exec_command(
                    f"grep 'EXIT_CODE' {log_path} 2>/dev/null || echo 'no_exit_code'", timeout=10
                )
                # No EXIT_CODE in background mode — check if LoRA output exists
                stdin, stdout, stderr = poll_client.exec_command(
                    f"ls {output_dir}/adapter_model.safetensors 2>/dev/null && echo SUCCESS || echo FAILED",
                    timeout=10
                )
                exit_check = stdout.read().decode("utf-8", errors="replace").strip()
                exit_code = 0 if "SUCCESS" in exit_check else 1
            except Exception:
                exit_code = 1

            poll_client.close()

            # Final parse of trainer_log.jsonl for any remaining entries
            final_losses = _parse_loss_from_trainer_state(output_dir)
            if len(final_losses) > len(accumulated_losses):
                # Push any remaining loss entries not yet streamed
                for i in range(len(accumulated_losses), len(final_losses)):
                    loss_val = final_losses[i]
                    accumulated_losses.append(loss_val)
                    if sync_redis_client:
                        emit_training_loss_sync(
                            job_id, 1, i + 1, loss_val,
                            accumulated_losses, sync_redis_client
                        )

            return exit_code, accumulated_losses, was_stopped

    # Timeout — 30 minutes exceeded
    logger.error(f"[Training] Timeout after {max_poll_attempts * 5}s, killing process {pid}")
    try:
        poll_client.exec_command(f"kill {pid}", timeout=10)
    except Exception:
        pass
    poll_client.close()
    return -1, accumulated_losses, False


def get_vllm_inference_url() -> str:
    """Get the vLLM inference URL pointing to the remote server."""
    return f"http://{SSH_HOST}:{REMOTE_VLLM_PORT}/v1"


def get_vllm_model_name(base_model: str = DEFAULT_BASE_MODEL) -> str:
    """Get the model name for vLLM inference (with LoRA suffix if applicable)."""
    # vLLM serves base model name; LoRA is loaded as module
    return base_model.split("/")[-1] if "/" in base_model else base_model


def ensure_model_cached(base_model: str) -> bool:
    """Ensure the base model is available on the remote server.

    Strategy:
    1. Check if base_model is a local path (starts with /) — verify it exists on remote
    2. If local path missing, resolve the ModelScope model ID and download via modelscope
    3. For HuggingFace-style names (org/model), download via ModelScope
    """
    # Case 1: local path on remote server — just verify existence
    if base_model.startswith("/"):
        check_cmd = f"ls {base_model}/config.json 2>/dev/null && echo EXISTS || echo MISSING"
        out, _, _ = _run_remote_command(check_cmd, timeout=10)

        if "EXISTS" in out:
            logger.info(f"[Training] Model local path exists on remote: {base_model}")
            return True

        # Local path doesn't exist — try to determine ModelScope ID from path
        model_name = base_model.rstrip("/").split("/")[-1]
        modelscope_id = MODELSCOPE_MODEL_IDS.get(model_name)

        if not modelscope_id:
            logger.error(f"[Training] Local path {base_model} not found and no ModelScope ID for '{model_name}'")
            return False

        logger.info(f"[Training] Local path {base_model} missing, downloading via ModelScope: {modelscope_id}")
        return _download_from_modelscope(modelscope_id, base_model)

    # Case 2: HuggingFace-style name (e.g., Qwen/Qwen3-8B) — download via ModelScope
    modelscope_id = base_model  # ModelScope uses same naming convention
    local_dir = f"/home/user/workspace/wangshuo/Models/{base_model.split('/')[-1]}"

    check_cmd = f"ls {local_dir}/config.json 2>/dev/null && echo EXISTS || echo MISSING"
    out, _, _ = _run_remote_command(check_cmd, timeout=10)

    if "EXISTS" in out:
        logger.info(f"[Training] Model already exists at {local_dir}")
        return True

    logger.info(f"[Training] Downloading model {modelscope_id} via ModelScope...")
    return _download_from_modelscope(modelscope_id, local_dir)


def _download_from_modelscope(modelscope_id: str, local_dir: str) -> bool:
    """Download model from ModelScope to the remote server.

    ModelScope is China-based, more reliable than HuggingFace for network access.
    """
    download_cmd = (
        f"pip install modelscope 2>/dev/null; "
        f"python -c \"from modelscope import snapshot_download; "
        f"snapshot_download('{modelscope_id}', local_dir='{local_dir}')\""
    )
    out, err, code = _run_remote_command(download_cmd, timeout=600)

    if code == 0:
        logger.info(f"[Training] Model {modelscope_id} downloaded via ModelScope to {local_dir}")
        return True
    else:
        logger.error(f"[Training] ModelScope download failed: {err[:300]}")
        return False