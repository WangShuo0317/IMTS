import os
import asyncio
import logging
import json
import time
import httpx
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
import paramiko
from dotenv import load_dotenv
from openai import AsyncOpenAI

from training_service import SSH_HOST, SSH_PORT, SSH_USER, SSH_PASSWORD, CONDA_ENV, REMOTE_WORKSPACE, _run_remote_command

logger = logging.getLogger(__name__)

load_dotenv()

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/home/user/workspace/wangshuo/Models/Qwen3-embeddings/")
REMOTE_EMBED_PORT = 8002

DASHSCOPE_API_KEY = os.getenv("EVAL_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DASHSCOPE_BASE_URL = os.getenv("EVAL_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
DASHSCOPE_MODEL = os.getenv("LLM_MODEL_NAME", "qwen-max")

LOCAL_SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "remote_embed_server.py")


def _start_embed_server() -> bool:
    """Deploy native transformers embedding server on remote GPU (not vLLM)."""
    logger.info("Deploying native embedding server on remote GPU...")
    # Kill any existing embed server
    _run_remote_command(f"pkill -f 'remote_embed_server' 2>/dev/null; pkill -f 'uvicorn remote_embed_server' 2>/dev/null", timeout=15)

    # Upload the server script
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=10)
    try:
        remote_script = f"{REMOTE_WORKSPACE}/remote_embed_server.py"
        sftp = client.open_sftp()
        sftp.put(LOCAL_SERVER_SCRIPT, remote_script)
        sftp.close()
    except Exception as e:
        logger.error(f"Failed to upload embed server script: {e}")
        client.close()
        return False

    # Launch via nohup
    script_content = (
        f"#!/bin/bash\n"
        f"source /home/user/miniconda3/etc/profile.d/conda.sh\n"
        f"conda activate {CONDA_ENV}\n"
        f"export EMBED_MODEL_PATH={EMBEDDING_MODEL_PATH}\n"
        f"export EMBED_PORT={REMOTE_EMBED_PORT}\n"
        f"nohup python {REMOTE_WORKSPACE}/remote_embed_server.py > {REMOTE_WORKSPACE}/embed_server.log 2>&1 &\n"
        f"echo $!\n"
    )
    script_path = f"{REMOTE_WORKSPACE}/start_embed.sh"
    try:
        sftp = client.open_sftp()
        with sftp.open(script_path, "w") as f:
            f.write(script_content)
        sftp.close()
        stdin, stdout, stderr = client.exec_command(f"chmod +x {script_path} && bash {script_path}")
        pid = stdout.read().decode().strip()
        logger.info(f"Native embedding server started with PID: {pid}")
    except Exception as e:
        logger.error(f"Failed to start embedding server: {e}")
        client.close()
        return False
    finally:
        client.close()

    # Poll until ready
    poll_client = paramiko.SSHClient()
    poll_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    poll_client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=10)
    for attempt in range(60):
        time.sleep(5)
        stdin, stdout, stderr = poll_client.exec_command(f"curl -sf http://localhost:{REMOTE_EMBED_PORT}/v1/models")
        out = stdout.read().decode()
        if '"data"' in out:
            logger.info("Embedding server is ready!")
            poll_client.close()
            return True
    logger.error("Embedding server failed to start within 300s")
    poll_client.close()
    return False


def _stop_embed_server():
    _run_remote_command(f"pkill -f 'remote_embed_server' 2>/dev/null; pkill -f 'uvicorn remote_embed_server' 2>/dev/null", timeout=15)
    logger.info("Embedding server stopped.")


async def get_embeddings(texts: list) -> list:
    """Fetch embeddings from remote embedding server in parallel batches."""
    os.environ["NO_PROXY"] = "10.242.33.21,127.0.0.1,localhost"
    os.environ["no_proxy"] = "10.242.33.21,127.0.0.1,localhost"
    url = f"http://10.242.33.21:{REMOTE_EMBED_PORT}/v1/embeddings"

    batch_size = 32
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    async def fetch_batch(client: httpx.AsyncClient, batch: list, batch_idx: int):
        payload = {"model": "qwen3-embeddings", "input": batch}
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return batch_idx, [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.error(f"Error fetching embeddings for batch {batch_idx}: {e}")
            return batch_idx, [[0.0] * 1024 for _ in batch]

    async with httpx.AsyncClient(timeout=120.0, limits=httpx.Limits(max_connections=50)) as client:
        tasks = [fetch_batch(client, batch, idx) for idx, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda x: x[0])
    embeddings = []
    for _, batch_embeds in results:
        embeddings.extend(batch_embeds)

    return embeddings


async def _dashscope_label_cluster(sample_texts: str) -> str:
    """Call DashScope qwen-max to label a cluster."""
    prompt = f"""以下是同一个类别下的几个数据样本：
{sample_texts}

请根据这些样本，用1-3个词总结它们所属的业务类别标签（例如：税务咨询、代码生成、日常闲聊等）。只需输出标签名，不要多余解释。"""

    client = AsyncOpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
    resp = await client.chat.completions.create(
        model=DASHSCOPE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
    )
    return resp.choices[0].message.content.strip()


async def cluster_and_label_dataset(dataset_path: str, llm_client=None) -> tuple:
    """
    1. Read dataset
    2. Start embed server, get vectors, stop server
    3. Cluster with HDBSCAN
    4. Call DashScope qwen-max to name clusters
    5. Save categorized dataset
    Returns (categorized_dataset_path, cluster_stats)
    """
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    else:
        df = pd.read_json(dataset_path)

    texts = []
    for _, row in df.iterrows():
        # Support multiple dataset formats
        q = str(row.get("instruction", row.get("question", row.get("title", row.get("input", "")))))
        a = str(row.get("output", row.get("answer", row.get("content", row.get("response", "")))))
        if q and a:
            texts.append(f"Q: {q} A: {a}")
        elif q:
            texts.append(q)
        elif a:
            texts.append(a)
        else:
            texts.append(str(row.get("domain", "unknown")))

    if not _start_embed_server():
        logger.error("Failed to start embedding server, aborting clustering")
        return dataset_path, {}

    try:
        embeddings = await get_embeddings(texts)
    finally:
        _stop_embed_server()

    # Check if embeddings are all zeros (failed fetch)
    X = np.array(embeddings)
    if np.allclose(X, 0):
        logger.error("All embeddings are zero — clustering impossible. Check embedding server logs.")
        return dataset_path, {}

    logger.info("Clustering embeddings...")
    n = len(df)
    # Use KMeans for high-dimensional normalized vectors (more reliable than HDBSCAN)
    from sklearn.cluster import KMeans
    # Estimate cluster count: aim for ~20-50 items per cluster
    n_clusters = max(3, min(20, n // 30))
    logger.info(f"KMeans params: n_clusters={n_clusters}")
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clusterer.fit_predict(X)
    df["cluster_id"] = labels

    # Label clusters via DashScope
    unique_labels = set(labels)
    cluster_names = {}

    for label in unique_labels:

        sample_rows = df[df["cluster_id"] == label].head(3)
        sample_examples = "\n\n".join([
            f"Question: {row.get('question', row.get('instruction', row.get('title', '')))}\nAnswer: {row.get('answer', row.get('output', row.get('content', '')))}"
            for _, row in sample_rows.iterrows()
        ])

        try:
            category_name = await _dashscope_label_cluster(sample_examples)
            cluster_names[label] = category_name
            logger.info(f"Cluster {label} labeled as: {category_name}")
        except Exception as e:
            logger.error(f"DashScope labeling failed for cluster {label}: {e}")
            cluster_names[label] = f"类别_{label}"

    df["category"] = df["cluster_id"].map(cluster_names)

    new_path = dataset_path.replace(".json", "_categorized.json").replace(".csv", "_categorized.csv")
    if new_path.endswith(".csv"):
        df.to_csv(new_path, index=False)
    else:
        df.to_json(new_path, orient="records", force_ascii=False)

    stats = df["category"].value_counts().to_dict()
    logger.info(f"Dataset categorized into {len(unique_labels)} clusters. Stats: {stats}")

    return new_path, stats