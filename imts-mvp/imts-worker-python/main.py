import asyncio
import json
import logging
import os
from typing import Optional, Dict
from dotenv import load_dotenv

import redis.asyncio as aioredis
import redis as sync_redis
from fastapi import FastAPI

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import text

from message_types import MessageBuilder
from graph_engine import run_workflow_async
from nodes import save_report, set_redis_client, set_sync_redis_client, set_session_factory

# 加载 .env 文件
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    # 为 data_opt_agent (使用 OPENAI_*) 和 eval_agent 建立别名映射
    if not os.getenv("OPENAI_API_KEY") and os.getenv("EVAL_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("EVAL_API_KEY")
    if not os.getenv("OPENAI_BASE_URL") and os.getenv("EVAL_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.getenv("EVAL_BASE_URL")
    if not os.getenv("OPENAI_MODEL_NAME") and os.getenv("EVAL_MODEL_NAME"):
        os.environ["OPENAI_MODEL_NAME"] = os.getenv("EVAL_MODEL_NAME")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IMTS Worker API (Async)")

redis_client: Optional[aioredis.Redis] = None
sync_redis_client_global: Optional[sync_redis.Redis] = None
queue_name = "imts_task_queue"

async_engine = None
async_session_factory = None

PASS_THRESHOLD = 75.0

# MinIO 配置 (用于从 MinIO URI 下载数据集到本地)
MINIO_ENDPOINT_URL = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "local_data")


def resolve_dataset_path(dataset_path: str) -> str:
    """解析数据集路径：MinIO URI → 本地绝对路径。

    如果 dataset_path 是 minio://bucket/key 格式，
    从 MinIO 下载到 LOCAL_DATA_DIR 并返回本地绝对路径。
    如果已是本地路径，直接返回（并校验文件存在）。
    """
    if not dataset_path:
        raise ValueError("dataset_path is empty")

    if dataset_path.startswith("minio://"):
        # 解析 minio://imts-datasets/10/v1/abc.json → bucket=imts-datasets, key=10/v1/abc.json
        uri_body = dataset_path[8:]
        parts = uri_body.split("/", 1)
        if len(parts) < 2:
            raise ValueError(f"Invalid MinIO URI: {dataset_path}")

        bucket, object_key = parts[0], parts[1]
        filename = os.path.basename(object_key)
        local_path = os.path.join(LOCAL_DATA_DIR, bucket, object_key)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            logger.info(f"Dataset already cached locally: {local_path}")
            return local_path

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        import boto3
        s3 = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT_URL,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            region_name='us-east-1'
        )
        logger.info(f"Downloading dataset from MinIO: {bucket}/{object_key} -> {local_path}")
        s3.download_file(bucket, object_key, local_path)

        # Pre-flight: 验证下载文件
        file_size = os.path.getsize(local_path)
        assert file_size > 0, f"Downloaded file is empty: {local_path}"
        logger.info(f"Dataset downloaded: {local_path} ({file_size} bytes)")
        return local_path

    # 本地路径 — 校验存在
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return dataset_path

# 活跃任务注册表 — 防止 fire-and-forget 导致幽灵任务
_active_tasks: Dict[str, asyncio.Task] = {}


def get_sync_redis_client():
    """获取同步 Redis 客户端用于 Pub/Sub"""
    global sync_redis_client_global
    if sync_redis_client_global is None:
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        sync_redis_client_global = sync_redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    return sync_redis_client_global


async def get_redis_client():
    global redis_client
    if redis_client is None:
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_client = aioredis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        set_redis_client(redis_client)
    return redis_client


async def should_stop_job(job_id: str) -> bool:
    """检查 Redis 中是否存在停止标记"""
    try:
        r = await get_redis_client()
        stop_key = f"imts_stop:{job_id}"
        exists = await r.exists(stop_key)
        return bool(exists)
    except Exception:
        return False


async def _monitor_task(task: asyncio.Task, job_id: str):
    """Task 完成后清理注册表；未捕获异常时兜底更新状态。"""
    try:
        await task
    except asyncio.CancelledError:
        logger.info(f"Job {job_id} task was cancelled")
        await update_job_status(job_id, "FAILED")
    except Exception as e:
        logger.critical(f"Unhandled exception in job {job_id}: {e}", exc_info=True)
        await update_job_status(job_id, "FAILED")
        if redis_client:
            try:
                builder = MessageBuilder(job_id, redis_client, get_sync_redis_client())
                await builder.error(f"Internal error: {e}")
            except Exception as notify_err:
                logger.error(f"Failed to notify frontend for job {job_id}: {notify_err}")
    finally:
        _active_tasks.pop(job_id, None)


async def process_job(job_json: str):
    """处理单个任务 - 使用 LangGraph 工作流"""
    from checkpoint_manager import get_checkpoint, clear_checkpoint, is_job_running

    # Fix #8: 先解析 JSON，确保 job_id 可用，避免后续异常处理中 job_id 为 "unknown"
    try:
        job_data = json.loads(job_json)
    except json.JSONDecodeError as e:
        logger.error(f"Malformed job JSON, cannot parse: {e}")
        return

    job_id = job_data.get("jobId")
    if not job_id:
        logger.error(f"Job JSON missing 'jobId' field: {job_json[:200]}")
        return

    try:
        logger.info(f"Processing job: {job_id}")

        # 检查是否被停止
        if await should_stop_job(job_id):
            logger.info(f"Job {job_id} was stopped before starting")
            await update_job_status(job_id, "FAILED")
            if redis_client:
                builder = MessageBuilder(job_id, redis_client, get_sync_redis_client())
                await builder.job_status("FAILED", "Job stopped by user")
            return

        # === 断点恢复：检查是否有可恢复的 checkpoint ===
        checkpoint = await get_checkpoint(job_id)
        is_resume = False
        if checkpoint:
            # 检查任务是否处于 RUNNING 状态（可能是 Worker 重启导致的任务中断）
            job_is_running = await is_job_running(job_id)
            if job_is_running:
                logger.info(f"检测到 checkpoint 恢复: job_id={job_id}, completed_node={checkpoint.get('completed_node')}, iteration={checkpoint.get('current_iteration')}")
                is_resume = True
            else:
                # 任务已结束但有 checkpoint，清理后重新开始
                logger.info(f"任务已结束但存在旧 checkpoint，清理: job_id={job_id}")
                await clear_checkpoint(job_id)
                checkpoint = None

        if not checkpoint:
            await update_job_status(job_id, "RUNNING")
        # === 断点恢复检查结束 ===

        builder = MessageBuilder(job_id, redis_client, get_sync_redis_client())
        await builder.job_status("RUNNING", "Job started")
        await builder._emit("WORKFLOW_START", {
            "job_id": job_id,
            "message": "LangGraph 工作流开始执行"
        }, progress=0)

        # 准备 job_data 兼容格式
        job_data["job_id"] = job_id
        job_data["target_prompt"] = job_data.get("targetPrompt", "")
        # 解析数据集路径：MinIO URI → 本地绝对路径（含 pre-flight 校验）
        raw_dataset_path = job_data.get("datasetPath", "")
        try:
            resolved_path = resolve_dataset_path(raw_dataset_path)
            job_data["dataset_path"] = resolved_path
            logger.info(f"Dataset path resolved: {raw_dataset_path} -> {resolved_path}")
        except Exception as resolve_err:
            logger.error(f"Failed to resolve dataset path '{raw_dataset_path}': {resolve_err}")
            await builder.job_status("FAILED", f"Dataset not accessible: {resolve_err}")
            await update_job_status(job_id, "FAILED")
            return
        job_data["model_name"] = job_data.get("modelName", "Qwen3-8B")  # 要训练的模型
        job_data["max_iterations"] = job_data.get("maxIterations", 3)
        job_data["target_score"] = PASS_THRESHOLD
        job_data["should_stop"] = should_stop_job  # 传递给工作流用于检查停止

        # 控制智能体的 LLM 配置（从 env 或前端参数获取）
        job_data["llm_api_key"] = job_data.get("llmApiKey") or os.getenv("OPENAI_API_KEY") or os.getenv("EVAL_API_KEY")
        job_data["llm_base_url"] = job_data.get("llmBaseUrl") or os.getenv("OPENAI_BASE_URL") or os.getenv("EVAL_BASE_URL")
        job_data["llm_model_name"] = job_data.get("llmModelName") or os.getenv("OPENAI_MODEL_NAME") or os.getenv("EVAL_MODEL_NAME")

        # 使用 LangGraph 工作流执行
        result = await run_workflow_async(job_data)

        passed = result.get("passed", False)
        final_iteration = result.get("current_iteration", 1)
        fatal_error = result.get("fatal_error")
        final_status = result.get("status")

        if fatal_error or final_status == "FATAL_ERROR":
            logger.error(f"Job {job_id} terminated with fatal error: {fatal_error[:300] if fatal_error else 'unknown'}")
            await builder.job_status("FAILED", fatal_error or "任务因不可恢复错误终止")
            await update_job_status(job_id, "FAILED", final_iteration)
        elif passed:
            await builder.job_status("SUCCESS", f"Job completed successfully after {final_iteration} iterations")
            await update_job_status(job_id, "SUCCESS", final_iteration)
        else:
            await builder.job_status("FAILED", f"Max iterations ({job_data['max_iterations']}) reached, target score not achieved")
            await update_job_status(job_id, "FAILED", final_iteration)

        # === 断点恢复：任务正常结束时清除 checkpoint ===
        await clear_checkpoint(job_id)
        # === checkpoint 清除结束 ===

        logger.info(f"Job {job_id} completed: passed={passed}, iterations={final_iteration}")

    except asyncio.CancelledError:
        logger.info(f"Job {job_id} was cancelled")
        await update_job_status(job_id, "FAILED")
        if redis_client:
            builder = MessageBuilder(job_id, redis_client, get_sync_redis_client())
            await builder.job_status("FAILED", "Job stopped by user")
        raise

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
        await update_job_status(job_id, "FAILED")
        if redis_client:
            try:
                builder = MessageBuilder(job_id, redis_client, get_sync_redis_client())
                await builder.error(str(e))
            except Exception as notify_err:
                logger.error(f"Failed to notify frontend for job {job_id}: {notify_err}")


async def update_job_status(job_id: str, status: str, iteration: int = None):
    global async_session_factory
    if async_session_factory is None:
        logger.error("Database session factory not initialized")
        return

    async with async_session_factory() as session:
        try:
            query = "UPDATE t_job_instance SET status = :status, updated_at = NOW()"
            if iteration:
                query += ", current_iteration = :iteration"
            query += " WHERE job_id = :job_id"

            params = {"status": status, "job_id": job_id}
            if iteration:
                params["iteration"] = iteration

            await session.execute(text(query), params)
            await session.commit()
            logger.info(f"Updated job {job_id} status to {status}, iteration={iteration}")
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to update job status: {e}")


async def worker_loop():
    """Worker 主循环 — 加入 Redis 重连退避和 JSON 解析容错"""
    logger.info("Async worker loop started")
    r = await get_redis_client()
    consecutive_errors = 0

    while True:
        try:
            result = await r.brpop(queue_name, timeout=5)
            if result:
                _, job_json = result
                logger.info(f"Received job: {job_json[:80]}...")

                # 提前解析 JSON 提取 job_id，避免 fire-and-forget 无法追溯
                try:
                    job_preview = json.loads(job_json)
                    job_id = job_preview.get("jobId", "")
                except json.JSONDecodeError:
                    logger.error(f"Malformed job JSON in queue, skipping: {job_json[:200]}")
                    continue

                if not job_id:
                    logger.error(f"Job JSON missing jobId, skipping: {job_json[:200]}")
                    continue

                # Fix #1: 受监控的 Task 而非 fire-and-forget
                task = asyncio.create_task(process_job(job_json))
                _active_tasks[job_id] = task
                asyncio.create_task(_monitor_task(task, job_id))

            consecutive_errors = 0

        except (aioredis.ConnectionError, aioredis.TimeoutError) as e:
            consecutive_errors += 1
            backoff = min(1 * (2 ** min(consecutive_errors - 1, 5)), 30)
            logger.error(f"Redis connection error (attempt {consecutive_errors}): {e}, backing off {backoff}s")
            await asyncio.sleep(backoff)
            try:
                r = await get_redis_client()
            except Exception as reconnect_err:
                logger.error(f"Redis reconnection failed: {reconnect_err}")

        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Worker loop error: {e}")
            await asyncio.sleep(1)


async def _recover_stale_tasks():
    """扫描 MySQL 中状态为 RUNNING 但无活跃 Task 的 job，将其重新入队或标记 FAILED。

    场景：Worker 进程崩溃/重启后，MySQL 中残留 RUNNING 状态的 job，
    但 _active_tasks 注册表为空 — 这些 job 已无人处理。
    """
    if async_session_factory is None:
        logger.warning("DB not initialized, skipping stale task recovery")
        return

    try:
        async with async_session_factory() as session:
            result = await session.execute(
                text("SELECT job_id FROM t_job_instance WHERE status = 'RUNNING'")
            )
            stale_jobs = [row[0] for row in result.fetchall()]

        if not stale_jobs:
            logger.info("No stale RUNNING jobs found")
            return

        # 过滤掉当前确实有活跃 Task 的 job
        truly_stale = [jid for jid in stale_jobs if jid not in _active_tasks]
        if not truly_stale:
            logger.info("All RUNNING jobs have active tasks, nothing to recover")
            return

        logger.warning(f"Found {len(truly_stale)} stale RUNNING jobs: {truly_stale}")

        r = await get_redis_client()
        for job_id in truly_stale:
            # 检查 Redis 中是否仍有 checkpoint 可恢复
            from checkpoint_manager import get_checkpoint, clear_checkpoint

            checkpoint = await get_checkpoint(job_id)
            if checkpoint:
                # 有 checkpoint — 重新入队让 Worker 从断点恢复
                job_payload = json.dumps({"jobId": job_id, "_recovery": True})
                await r.lpush(queue_name, job_payload)
                logger.info(f"Re-enqueued stale job {job_id} for checkpoint recovery")
            else:
                # 无 checkpoint — 无法恢复，标记为 FAILED
                await update_job_status(job_id, "FAILED")
                logger.info(f"Marked unrecoverable stale job {job_id} as FAILED")

    except Exception as e:
        logger.error(f"Stale task recovery failed: {e}", exc_info=True)


@app.on_event("startup")
async def startup():
    global async_engine, async_session_factory

    logger.info("Initializing database...")
    mysql_host = os.getenv('MYSQL_HOST', 'localhost')
    mysql_port = os.getenv('MYSQL_PORT', '3306')
    mysql_user = os.getenv('MYSQL_USER', 'root')
    mysql_password = os.getenv('MYSQL_PASSWORD', 'admin123456')
    mysql_database = os.getenv('MYSQL_DATABASE', 'imts_db')

    db_url = f"mysql+aiomysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database}"
    async_engine = create_async_engine(db_url, echo=False)
    async_session_factory = async_sessionmaker(async_engine, expire_on_commit=False)

    set_session_factory(async_session_factory)

    # 初始化同步 Redis 客户端（用于 Pub/Sub，与 Spring ReactiveRedisTemplate 兼容）
    sync_redis_host = os.getenv('REDIS_HOST', 'localhost')
    sync_redis_port = int(os.getenv('REDIS_PORT', 6379))
    sync_r = sync_redis.Redis(host=sync_redis_host, port=sync_redis_port, decode_responses=True)
    set_sync_redis_client(sync_r)
    logger.info(f"Initialized sync Redis client: {sync_redis_host}:{sync_redis_port}")

    # Fix #2: 启动时恢复因 Worker 重启而中断的 RUNNING job
    await _recover_stale_tasks()

    logger.info("Starting async worker task...")
    asyncio.create_task(worker_loop())


@app.get("/health")
async def health():
    return {"status": "UP", "active_tasks": len(_active_tasks)}


@app.get("/")
async def root():
    return {"message": "IMTS Worker API (Async)"}


@app.post("/stop/{job_id}")
async def stop_job(job_id: str):
    """停止指定的任务"""
    try:
        r = await get_redis_client()
        stop_key = f"imts_stop:{job_id}"
        await r.set(stop_key, "1", ex=3600)  # 1小时后过期
        logger.info(f"Stop flag set for job: {job_id}")

        # 更新数据库状态
        await update_job_status(job_id, "FAILED")

        # 通知前端
        builder = MessageBuilder(job_id, r, get_sync_redis_client())
        await builder.job_status("FAILED", "Job stopped by user")

        # Fix #1: 从注册表取消活跃 Task
        active_task = _active_tasks.pop(job_id, None)
        if active_task and not active_task.done():
            active_task.cancel()
            logger.info(f"Cancelled active task for job: {job_id}")

        return {"success": True, "jobId": job_id, "message": "Stop signal sent"}
    except Exception as e:
        logger.error(f"Failed to stop job {job_id}: {e}")
        return {"success": False, "jobId": job_id, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
