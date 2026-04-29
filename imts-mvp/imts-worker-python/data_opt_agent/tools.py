"""
Custom filesystem tools that support MinIO URIs.

These tools override the deepagents built-in tools to handle MinIO paths.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from data_opt_agent.skills.data_augmenter.scripts.category_augmenter import cluster_and_find_weak_categories

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")


def parse_minio_uri(uri: str) -> Optional[tuple]:
    """Parse minio://bucket/path format URI."""
    if not uri.startswith("minio://"):
        return None
    path = uri[8:]
    parts = path.split("/", 1)
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


async def read_from_minio(bucket: str, object_path: str) -> str:
    """Read content from MinIO."""
    from minio import Minio

    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    try:
        response = client.get_object(bucket, object_path)
        content = response.read().decode("utf-8")
        response.close()
        response.release_conn()
        return content
    except Exception as e:
        return f"Error reading from MinIO: {str(e)}"


@tool
async def read_file(file_path: str, limit: int = 1000, offset: int = 0) -> Dict[str, Any]:
    """Read content from a file or MinIO URI.

    Supports:
    - Local file paths: /path/to/file.txt or D:/path/to/file.txt
    - MinIO URIs: minio://bucket/object-path

    Args:
        file_path: Path to the file or MinIO URI
        limit: Maximum number of lines/chars to read (default 1000)
        offset: Starting offset in lines/chars (default 0)

    Returns:
        Dict with content, line_count, and status
    """
    # Check if it's a MinIO URI
    minio_parsed = parse_minio_uri(file_path)
    if minio_parsed:
        bucket, object_path = minio_parsed
        content = await read_from_minio(bucket, object_path)
        if content.startswith("Error"):
            return {
                "status": "error",
                "content": content,
                "line_count": 0
            }
        lines = content.split("\n")
        return {
            "status": "success",
            "content": "\n".join(lines[offset:offset + limit]),
            "line_count": len(lines),
            "source": f"minio://{bucket}/{object_path}"
        }

    # Handle local file path
    # Convert Windows path to proper format
    file_path = file_path.replace("\\", "/")

    # Handle virtual paths (e.g., /IMTS/... which maps to D:/IMTS/...)
    if file_path.startswith("/"):
        # Check if it's a known virtual path prefix
        if file_path.startswith("/IMTS/"):
            file_path = "D:" + file_path
        elif file_path.startswith("/workspace/"):
            file_path = file_path.replace("/workspace/", "")

    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "status": "error",
                "content": f"File not found: {file_path}",
                "line_count": 0
            }

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        return {
            "status": "success",
            "content": "\n".join(lines[offset:offset + limit]),
            "line_count": len(lines),
            "source": str(path)
        }
    except Exception as e:
        return {
            "status": "error",
            "content": f"Error reading file: {str(e)}",
            "line_count": 0
        }


@tool
async def ls(path: str = ".") -> Dict[str, Any]:
    """List directory contents or MinIO bucket contents.

    Supports:
    - Local directory paths: /path/to/dir or D:/path/to/dir
    - MinIO URIs: minio://bucket/prefix

    Args:
        path: Directory path or MinIO URI

    Returns:
        Dict with file list and status
    """
    # Check if it's a MinIO URI
    minio_parsed = parse_minio_uri(path)
    if minio_parsed:
        bucket, prefix = minio_parsed
        from minio import Minio

        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )

        try:
            objects = client.list_objects(bucket, prefix=prefix, recursive=True)
            files = []
            for obj in objects:
                files.append({
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": str(obj.last_modified) if hasattr(obj, 'last_modified') else None
                })
            return {
                "status": "success",
                "files": files,
                "source": f"minio://{bucket}/{prefix}"
            }
        except Exception as e:
            return {
                "status": "error",
                "files": [],
                "error": str(e)
            }

    # Handle local path
    path = path.replace("\\", "/")

    # Handle virtual paths
    if path.startswith("/") and not Path(path).exists():
        if path.startswith("/IMTS/"):
            path = "D:" + path
        elif path.startswith("/workspace/"):
            path = path.replace("/workspace/", "")

    try:
        p = Path(path)
        if not p.exists():
            return {
                "status": "error",
                "files": [],
                "error": f"Path not found: {path}"
            }

        if p.is_file():
            return {
                "status": "success",
                "files": [{"name": p.name, "size": p.stat().st_size, "type": "file"}],
                "source": str(p)
            }

        files = []
        for item in p.iterdir():
            files.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0
            })
        return {
            "status": "success",
            "files": files,
            "source": str(p)
        }
    except Exception as e:
        return {
            "status": "error",
            "files": [],
            "error": str(e)
        }


@tool
async def write_file(file_path: str, content: str) -> Dict[str, Any]:
    """Write content to a file or MinIO URI.

    Supports:
    - Local file paths: /path/to/file.txt
    - MinIO URIs: minio://bucket/object-path (writes to MinIO)

    Args:
        file_path: Path to the file or MinIO URI
        content: Content to write

    Returns:
        Dict with status and message
    """
    # Check if it's a MinIO URI
    minio_parsed = parse_minio_uri(file_path)
    if minio_parsed:
        bucket, object_path = minio_parsed
        from minio import Minio

        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )

        try:
            data = content.encode("utf-8")
            client.put_object(bucket, object_path, data, len(data))
            return {
                "status": "success",
                "message": f"Written to minio://{bucket}/{object_path}",
                "bytes": len(data)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    # Handle local path
    file_path = file_path.replace("\\", "/")

    # Handle virtual paths
    if file_path.startswith("/"):
        if file_path.startswith("/IMTS/"):
            file_path = "D:" + file_path
        elif file_path.startswith("/workspace/"):
            file_path = file_path.replace("/workspace/", "")

    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "status": "success",
            "message": f"Written to {file_path}",
            "bytes": len(content)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# Export all custom tools
CUSTOM_TOOLS = [read_file, ls, write_file, cluster_and_find_weak_categories]
