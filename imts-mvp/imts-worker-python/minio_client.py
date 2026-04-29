"""
MinIO Storage Adapter - 数据集存储到 MinIO (S3兼容)

使用 MinIO 作为数据集的持久化存储：
- original: 原始数据集
- split: 分割后的训练/测试集
- optimized: 每次迭代的优化数据
- versions: 版本元数据 (JSON)

连接信息：
- Endpoint: http://localhost:9000 (本机 Docker)
- Access Key: minioadmin
- Secret Key: minioadmin123
"""

import os
import json
import io
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

# MinIO 连接配置
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")

# Bucket 名称
BUCKET_ORIGINAL = "imts-original"
BUCKET_SPLIT = "imts-split"
BUCKET_OPTIMIZED = "imts-optimized"
BUCKET_VERSIONS = "imts-versions"


class MinIOClient:
    """MinIO S3 客户端封装"""

    _instance = None

    def __init__(self):
        self.endpoint = MINIO_ENDPOINT
        self.access_key = MINIO_ACCESS_KEY
        self.secret_key = MINIO_SECRET_KEY
        self._client = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_client(self):
        """获取或创建 S3 客户端"""
        if self._client is None:
            import boto3
            self._client = boto3.client(
                's3',
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name='us-east-1'
            )
            # 确保所有 bucket 存在
            for bucket in [BUCKET_ORIGINAL, BUCKET_SPLIT, BUCKET_OPTIMIZED, BUCKET_VERSIONS]:
                try:
                    self._client.head_bucket(Bucket=bucket)
                except:
                    try:
                        self._client.create_bucket(Bucket=bucket)
                        logger.info(f"Created bucket: {bucket}")
                    except Exception as bucket_err:
                        logger.error(f"Failed to create bucket {bucket}: {bucket_err}")
        return self._client

    def upload_file(self, bucket: str, key: str, file_path: str):
        """上传本地文件到 MinIO"""
        client = self._get_client()
        client.upload_file(file_path, bucket, key)
        logger.info(f"Uploaded {file_path} -> {bucket}/{key}")

    def upload_dataframe(self, bucket: str, key: str, df: pd.DataFrame, format: str = "csv"):
        """上传 DataFrame 到 MinIO"""
        client = self._get_client()
        if format == "csv":
            content = df.to_csv(index=False, encoding='utf-8')
            content_type = "text/csv"
        elif format == "json":
            content = df.to_json(orient='records', force_ascii=False, indent=2)
            content_type = "application/json"
        else:
            raise ValueError(f"Unsupported format: {format}")

        client.put_object(Bucket=bucket, Key=key, Body=content.encode('utf-8'), ContentType=content_type)
        logger.info(f"Uploaded DataFrame -> {bucket}/{key}")

    def download_file(self, bucket: str, key: str, dest_path: str):
        """从 MinIO 下载文件到本地"""
        client = self._get_client()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        client.download_file(bucket, key, dest_path)
        logger.info(f"Downloaded {bucket}/{key} -> {dest_path}")

    def download_dataframe(self, bucket: str, key: str) -> pd.DataFrame:
        """从 MinIO 下载为 DataFrame"""
        client = self._get_client()
        response = client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')

        if key.endswith('.csv'):
            return pd.read_csv(io.StringIO(content))
        elif key.endswith('.json'):
            return pd.read_json(io.StringIO(content))
        else:
            raise ValueError(f"Unsupported file format: {key}")

    def list_objects(self, bucket: str, prefix: str = "") -> List[str]:
        """列出 bucket 中的对象"""
        client = self._get_client()
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]

    def object_exists(self, bucket: str, key: str) -> bool:
        """检查对象是否存在"""
        import botocore.exceptions
        client = self._get_client()
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except botocore.exceptions.ClientError:
            return False

    def upload_json(self, bucket: str, key: str, data: dict):
        """上传 JSON 到 MinIO"""
        client = self._get_client()
        content = json.dumps(data, ensure_ascii=False, indent=2)
        client.put_object(Bucket=bucket, Key=key, Body=content.encode('utf-8'), ContentType='application/json')
        logger.info(f"Uploaded JSON -> {bucket}/{key}")

    def download_json(self, bucket: str, key: str) -> dict:
        """从 MinIO 下载 JSON"""
        client = self._get_client()
        response = client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)

    def get_object_url(self, bucket: str, key: str) -> str:
        """获取对象的预签名 URL (7天有效期)"""
        client = self._get_client()
        return client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=7 * 24 * 3600
        )


def get_minio_client() -> MinIOClient:
    """获取 MinIO 客户端单例"""
    return MinIOClient.get_instance()