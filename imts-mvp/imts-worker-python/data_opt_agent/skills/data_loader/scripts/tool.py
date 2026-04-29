"""
DataLoader Tool - 数据加载工具
支持本地文件路径和 MinIO URI (minio://bucket/path)
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool

# MinIO 配置
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")


def parse_minio_uri(uri: str) -> Optional[tuple]:
    """解析 minio://bucket/path 格式的 URI"""
    if not uri.startswith("minio://"):
        return None
    path = uri[8:]  # 移除 "minio://"
    parts = path.split("/", 1)
    if len(parts) < 2:
        return None
    bucket = parts[0]
    object_path = parts[1]
    return bucket, object_path


async def load_from_minio(bucket: str, object_path: str) -> dict:
    """从 MinIO 加载数据"""
    from minio import Minio

    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    # 获取对象信息
    stat = client.stat_object(bucket, object_path)
    size_bytes = stat.size

    # 下载对象内容
    response = client.get_object(bucket, object_path)
    data = response.read()
    response.close()
    response.release_conn()

    # 解析数据格式
    content = data.decode("utf-8")

    # 根据扩展名判断格式
    if object_path.endswith(".jsonl"):
        lines = content.strip().split("\n")
        rows = len(lines)
        columns = list(json.loads(lines[0]).keys()) if lines else []
        file_type = "jsonl"
    elif object_path.endswith(".json"):
        parsed = json.loads(content)
        if isinstance(parsed, list):
            rows = len(parsed)
            columns = list(parsed[0].keys()) if parsed else []
        else:
            rows = 1
            columns = list(parsed.keys())
        file_type = "json"
    elif object_path.endswith(".csv"):
        lines = content.strip().split("\n")
        rows = len(lines) - 1  # 减去表头
        columns = lines[0].split(",")
        file_type = "csv"
    else:
        # 尝试自动检测
        file_type = "unknown"
        rows = 0
        columns = []

    return {
        "path": f"minio://{bucket}/{object_path}",
        "rows": rows,
        "columns": columns,
        "file_type": file_type,
        "encoding": "utf-8",
        "size_bytes": size_bytes,
        "bucket": bucket,
        "object_path": object_path
    }


async def load_from_local(path: str) -> dict:
    """从本地文件系统加载数据"""
    # 处理 Windows 路径
    path = path.replace("\\", "/")

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    size_bytes = file_path.stat().st_size

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 根据扩展名判断格式
    if str(path).endswith(".jsonl"):
        lines = content.strip().split("\n")
        rows = len(lines)
        columns = list(json.loads(lines[0]).keys()) if lines else []
        file_type = "jsonl"
    elif str(path).endswith(".json"):
        parsed = json.loads(content)
        if isinstance(parsed, list):
            rows = len(parsed)
            columns = list(parsed[0].keys()) if parsed else []
        else:
            rows = 1
            columns = list(parsed.keys())
        file_type = "json"
    elif str(path).endswith(".csv"):
        lines = content.strip().split("\n")
        rows = len(lines) - 1
        columns = lines[0].split(",")
        file_type = "csv"
    else:
        file_type = "unknown"
        rows = 0
        columns = []

    return {
        "path": path,
        "rows": rows,
        "columns": columns,
        "file_type": file_type,
        "encoding": "utf-8",
        "size_bytes": size_bytes
    }


@tool
async def data_loader(path: str) -> dict:
    """加载并解析各种格式（CSV、JSON、JSONL）的数据集。

    支持本地文件系统路径和 MinIO URI（minio://bucket/path）。

    当需要将数据集文件加载到内存中进行处理时使用此技能。

    Args:
        path: 数据集文件路径。可以是：
              - 本地路径：/data/train.csv, D:/data/train.json
              - MinIO URI：minio://datasets/12

    Returns:
        dict，包含 path、rows、columns、file_type、encoding、size_bytes。
    """
    await asyncio.sleep(0.5)  # 模拟加载延迟

    # 解析 MinIO URI
    minio_parsed = parse_minio_uri(path)

    if minio_parsed:
        # 从 MinIO 加载
        bucket, object_path = minio_parsed
        try:
            result = await load_from_minio(bucket, object_path)
            await asyncio.sleep(0.5)
            return result
        except Exception as e:
            raise ValueError(f"Failed to load from MinIO: {e}")
    else:
        # 从本地文件系统加载
        try:
            result = await load_from_local(path)
            await asyncio.sleep(0.5)
            return result
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except Exception as e:
            raise ValueError(f"Failed to load local file: {e}")
