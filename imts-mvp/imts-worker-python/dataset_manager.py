"""
Dataset Manager - 版本管理、数据分割、数据集追踪 (MinIO 存储)

数据集版本管理策略：
1. 原始数据集：imts-original/{job_id}/v1/{filename}
2. 分割后的测试集：imts-split/{job_id}/test_{seed}.csv
3. 优化后的数据集：imts-optimized/{job_id}/iter_{iteration}.csv
4. 数据集版本元数据：imts-versions/{job_id}/metadata.json

数据分割：
- 首次迭代：随机划分 10% 作为测试集，90% 用于训练
- 后续迭代：复用已划分的测试集，训练数据经过优化后替换
"""

import os
import json
import random
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

# MinIO 存储配置
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")

# Bucket 名称
BUCKET_ORIGINAL = "imts-original"
BUCKET_SPLIT = "imts-split"
BUCKET_OPTIMIZED = "imts-optimized"
BUCKET_VERSIONS = "imts-versions"

# 本地临时目录（用于暂存下载的文件供处理）
LOCAL_TEMP_DIR = os.path.join(os.path.dirname(__file__), "..", "temp_datasets")


class DatasetVersion:
    """数据集版本信息"""
    def __init__(self, version_id: str, job_id: str, iteration: int, dataset_type: str,
                 minio_key: str, original_rows: int, final_rows: int,
                 test_size: float = 0.1, seed: int = 42):
        self.version_id = version_id
        self.job_id = job_id
        self.iteration = iteration
        self.dataset_type = dataset_type  # "original", "train", "test", "optimized"
        self.minio_key = minio_key
        self.original_rows = original_rows
        self.final_rows = final_rows
        self.test_size = test_size
        self.seed = seed
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "job_id": self.job_id,
            "iteration": self.iteration,
            "dataset_type": self.dataset_type,
            "minio_key": self.minio_key,
            "original_rows": self.original_rows,
            "final_rows": self.final_rows,
            "test_size": self.test_size,
            "seed": self.seed,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetVersion":
        v = cls(
            version_id=d["version_id"],
            job_id=d["job_id"],
            iteration=d["iteration"],
            dataset_type=d["dataset_type"],
            minio_key=d["minio_key"],
            original_rows=d["original_rows"],
            final_rows=d["final_rows"],
            test_size=d.get("test_size", 0.1),
            seed=d.get("seed", 42)
        )
        v.created_at = d.get("created_at", datetime.now().isoformat())
        return v


class DatasetManager:
    """数据集版本管理器 (MinIO 存储)"""

    _client = None

    def __init__(self, job_id: str):
        self.job_id = job_id
        os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

    @classmethod
    def _get_client(cls):
        """获取 MinIO S3 客户端"""
        if cls._client is None:
            import boto3
            cls._client = boto3.client(
                's3',
                endpoint_url=MINIO_ENDPOINT,
                aws_access_key_id=MINIO_ACCESS_KEY,
                aws_secret_access_key=MINIO_SECRET_KEY,
                region_name='us-east-1'
            )
            # 确保 bucket 存在
            for bucket in [BUCKET_ORIGINAL, BUCKET_SPLIT, BUCKET_OPTIMIZED, BUCKET_VERSIONS]:
                try:
                    cls._client.head_bucket(Bucket=bucket)
                except:
                    try:
                        cls._client.create_bucket(Bucket=bucket)
                        logger.info(f"Created bucket: {bucket}")
                    except Exception as bucket_err:
                        logger.error(f"Failed to create bucket {bucket}: {bucket_err}")
        return cls._client

    def _get_filename(self, path: str) -> str:
        """从路径提取文件名"""
        return os.path.basename(path)

    def _generate_version_id(self, prefix: str = "v") -> str:
        """生成版本ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def _load_dataframe(self, path: str) -> pd.DataFrame:
        """加载数据集为 DataFrame"""
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.json'):
            return pd.read_json(path)
        elif path.endswith('.jsonl'):
            return pd.read_json(path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def _download_to_local(self, bucket: str, key: str) -> str:
        """下载 MinIO 对象到本地临时目录"""
        client = self._get_client()
        local_path = os.path.join(LOCAL_TEMP_DIR, key.replace('/', '_'))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client.download_file(bucket, key, local_path)
        return local_path

    # ========================
    # 数据集版本管理核心方法
    # ========================

    def save_original_dataset(self, source_path: str) -> DatasetVersion:
        """
        保存原始数据集到 MinIO

        Args:
            source_path: 原始数据集路径

        Returns:
            DatasetVersion 对象
        """
        client = self._get_client()
        filename = self._get_filename(source_path)
        version_id = self._generate_version_id("orig")
        minio_key = f"{self.job_id}/v1_{filename}"

        # 上传到 MinIO
        client.upload_file(source_path, BUCKET_ORIGINAL, minio_key)

        # 读取数据统计
        df = self._load_dataframe(source_path)
        original_rows = len(df)

        version = DatasetVersion(
            version_id=version_id,
            job_id=self.job_id,
            iteration=0,
            dataset_type="original",
            minio_key=minio_key,
            original_rows=original_rows,
            final_rows=original_rows
        )

        self._save_version_info(version)
        logger.info(f"Saved original dataset to MinIO: {BUCKET_ORIGINAL}/{minio_key}, rows={original_rows}")

        return version

    def split_train_test(self, source_path: str, test_size: float = 0.1,
                         seed: int = 42, force_resplit: bool = False) -> Tuple[DatasetVersion, DatasetVersion]:
        """
        划分训练集和测试集

        Args:
            source_path: 源数据集路径
            test_size: 测试集比例 (默认 10%)
            seed: 随机种子 (默认 42)
            force_resplit: 是否强制重新划分

        Returns:
            (train_version, test_version) 元组
        """
        # 检查是否已有划分
        if not force_resplit:
            existing = self.get_latest_split()
            if existing:
                logger.info(f"Using existing split for job {self.job_id}")
                train_v, test_v = existing
                return train_v, test_v

        # 加载数据
        df = self._load_dataframe(source_path)
        total_rows = len(df)

        # 随机划分
        random.seed(seed)
        indices = list(range(total_rows))
        random.shuffle(indices)

        test_count = max(1, int(total_rows * test_size))
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]

        train_df = df.iloc[train_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        # 保存到本地临时文件
        filename = self._get_filename(source_path)
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]

        train_local = os.path.join(LOCAL_TEMP_DIR, f"train_{seed}_{filename}")
        test_local = os.path.join(LOCAL_TEMP_DIR, f"test_{seed}_{filename}")

        train_df.to_csv(train_local, index=False, encoding='utf-8')
        test_df.to_csv(test_local, index=False, encoding='utf-8')

        # 上传到 MinIO
        client = self._get_client()
        train_key = f"{self.job_id}/train_{seed}{ext}"
        test_key = f"{self.job_id}/test_{seed}{ext}"

        client.upload_file(train_local, BUCKET_SPLIT, train_key)
        client.upload_file(test_local, BUCKET_SPLIT, test_key)

        # 删除本地临时文件
        try:
            os.remove(train_local)
            os.remove(test_local)
        except OSError as cleanup_err:
            logger.warning(f"Failed to clean up temp files: {cleanup_err}")

        # 创建版本信息
        train_version = DatasetVersion(
            version_id=self._generate_version_id("train"),
            job_id=self.job_id,
            iteration=0,
            dataset_type="train",
            minio_key=train_key,
            original_rows=total_rows,
            final_rows=len(train_df),
            test_size=test_size,
            seed=seed
        )

        test_version = DatasetVersion(
            version_id=self._generate_version_id("test"),
            job_id=self.job_id,
            iteration=0,
            dataset_type="test",
            minio_key=test_key,
            original_rows=total_rows,
            final_rows=len(test_df),
            test_size=test_size,
            seed=seed
        )

        self._save_version_info(train_version)
        self._save_version_info(test_version)

        # 保存分割元数据
        split_meta = {
            "job_id": self.job_id,
            "test_size": test_size,
            "seed": seed,
            "total_rows": total_rows,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_key": train_key,
            "test_key": test_key,
            "train_bucket": BUCKET_SPLIT,
            "test_bucket": BUCKET_SPLIT,
            "created_at": datetime.now().isoformat()
        }
        self._save_json_to_minio("split_meta.json", split_meta)

        logger.info(f"Split and uploaded: train={len(train_df)}, test={len(test_df)} to MinIO")
        return train_version, test_version

    def save_optimized_dataset(self, source_path: str, iteration: int) -> DatasetVersion:
        """
        保存优化后的训练数据集到 MinIO

        Args:
            source_path: 优化后的数据集路径
            iteration: 当前迭代号

        Returns:
            DatasetVersion 对象
        """
        client = self._get_client()

        # 确保源文件存在
        if not os.path.exists(source_path):
            logger.warning(f"Optimized dataset not found at {source_path}, skipping save")
            return None

        filename = self._get_filename(source_path)
        version_filename = f"iter_{iteration}_{filename}"
        minio_key = f"{self.job_id}/{version_filename}"

        # 上传到 MinIO
        client.upload_file(source_path, BUCKET_OPTIMIZED, minio_key)

        # 读取数据统计
        df = self._load_dataframe(source_path)
        final_rows = len(df)

        version = DatasetVersion(
            version_id=self._generate_version_id("opt"),
            job_id=self.job_id,
            iteration=iteration,
            dataset_type="optimized",
            minio_key=minio_key,
            original_rows=final_rows,
            final_rows=final_rows
        )

        self._save_version_info(version)
        logger.info(f"Saved optimized dataset to MinIO: iter={iteration}, rows={final_rows}")

        return version

    # ========================
    # 数据查询方法
    # ========================

    def get_latest_split(self) -> Optional[Tuple[DatasetVersion, DatasetVersion]]:
        """获取最新的划分结果"""
        try:
            split_meta = self._load_json_from_minio("split_meta.json")
        except Exception:
            logger.warning(f"No split metadata found for job {self.job_id}")
            return None

        train_key = split_meta.get("train_key", "")
        test_key = split_meta.get("test_key", "")

        if not train_key or not test_key:
            return None

        train_version = DatasetVersion(
            version_id="train_latest",
            job_id=self.job_id,
            iteration=0,
            dataset_type="train",
            minio_key=train_key,
            original_rows=split_meta["total_rows"],
            final_rows=split_meta["train_rows"],
            test_size=split_meta.get("test_size", 0.1),
            seed=split_meta.get("seed", 42)
        )

        test_version = DatasetVersion(
            version_id="test_latest",
            job_id=self.job_id,
            iteration=0,
            dataset_type="test",
            minio_key=test_key,
            original_rows=split_meta["total_rows"],
            final_rows=split_meta["test_rows"],
            test_size=split_meta.get("test_size", 0.1),
            seed=split_meta.get("seed", 42)
        )

        return train_version, test_version

    def get_latest_optimized(self) -> Optional[DatasetVersion]:
        """获取最新优化后的训练数据集"""
        versions_file = os.path.join(LOCAL_TEMP_DIR, f"versions_{self.job_id}.json")
        all_versions = []

        # 尝试从本地缓存加载
        if os.path.exists(versions_file):
            try:
                all_versions = self._load_json_local(versions_file)
            except Exception as e:
                logger.warning(f"Failed to load version cache locally for job {self.job_id}: {e}")

        # 如果本地没有加载到，尝试从 MinIO 恢复
        if not all_versions:
            try:
                all_versions = self._load_json_from_minio("versions.json")
                if all_versions:
                    logger.info(f"Recovered version metadata from MinIO for job {self.job_id}")
                    # 保存到本地缓存以便后续使用
                    self._save_json_local(versions_file, all_versions)
            except Exception as e:
                logger.warning(f"No version metadata in MinIO for job {self.job_id}: {e}")

        if all_versions:
            optimized_versions = [v for v in all_versions if v.get("dataset_type") == "optimized"]
            if optimized_versions:
                latest = sorted(optimized_versions, key=lambda x: x.get("iteration", 0))[-1]
                return DatasetVersion.from_dict(latest)

        return None

    def get_test_dataset_path(self) -> Optional[str]:
        """获取测试数据集路径（下载到本地临时目录）"""
        split = self.get_latest_split()
        if not split:
            return None

        test_version = split[1]
        local_path = os.path.join(LOCAL_TEMP_DIR, f"test_{self.job_id}.csv")
        os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

        try:
            client = self._get_client()
            client.download_file(BUCKET_SPLIT, test_version.minio_key, local_path)
            logger.info(f"Downloaded test dataset from MinIO: {test_version.minio_key} -> {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download test dataset: {e}")
            return None

    def get_latest_train_dataset_path(self) -> Optional[str]:
        """获取最新训练数据集路径（优先优化后的，下载到本地临时目录）"""
        # 优先返回优化后的数据
        optimized = self.get_latest_optimized()
        if optimized:
            local_path = os.path.join(LOCAL_TEMP_DIR, f"train_opt_{self.job_id}.csv")
            os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
            try:
                client = self._get_client()
                client.download_file(BUCKET_OPTIMIZED, optimized.minio_key, local_path)
                return local_path
            except Exception as dl_err:
                logger.warning(f"Failed to download optimized dataset: {dl_err}")

        # 其次返回原始训练集
        split = self.get_latest_split()
        if split:
            train_version = split[0]
            local_path = os.path.join(LOCAL_TEMP_DIR, f"train_{self.job_id}.csv")
            os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
            try:
                client = self._get_client()
                client.download_file(BUCKET_SPLIT, train_version.minio_key, local_path)
                return local_path
            except Exception as dl_err:
                logger.warning(f"Failed to download train dataset: {dl_err}")

        return None

    # ========================
    # 辅助方法
    # ========================

    def _save_version_info(self, version: DatasetVersion):
        """保存版本信息到本地缓存 + MinIO"""
        versions_file = os.path.join(LOCAL_TEMP_DIR, f"versions_{self.job_id}.json")
        all_versions = []

        if os.path.exists(versions_file):
            all_versions = self._load_json_local(versions_file)

        # 更新或添加版本
        existing_idx = None
        for i, v in enumerate(all_versions):
            if v["version_id"] == version.version_id:
                existing_idx = i
                break

        if existing_idx is not None:
            all_versions[existing_idx] = version.to_dict()
        else:
            all_versions.append(version.to_dict())

        self._save_json_local(versions_file, all_versions)
        self._save_json_to_minio("versions.json", all_versions)

    def _save_json_to_minio(self, key: str, data: dict):
        """保存 JSON 到 MinIO"""
        client = self._get_client()
        full_key = f"{self.job_id}/{key}"
        content = json.dumps(data, ensure_ascii=False, indent=2)
        client.put_object(Bucket=BUCKET_VERSIONS, Key=full_key, Body=content.encode('utf-8'),
                         ContentType='application/json')

    def _load_json_from_minio(self, key: str) -> dict:
        """从 MinIO 加载 JSON"""
        import botocore.exceptions
        client = self._get_client()
        full_key = f"{self.job_id}/{key}"
        try:
            response = client.get_object(Bucket=BUCKET_VERSIONS, Key=full_key)
            content = response['Body'].read().decode('utf-8')
            return json.loads(content)
        except botocore.exceptions.ClientError as e:
            logger.warning(f"MinIO object not found: {BUCKET_VERSIONS}/{full_key}: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MinIO object {full_key}: {e}")
            return {}

    def _save_json_local(self, path: str, data: dict):
        """保存 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_json_local(self, path: str) -> dict:
        """加载 JSON 文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_dataset_summary(self) -> dict:
        """获取数据集版本摘要"""
        summary = {
            "job_id": self.job_id,
            "minio_endpoint": MINIO_ENDPOINT,
            "buckets": {
                "original": BUCKET_ORIGINAL,
                "split": BUCKET_SPLIT,
                "optimized": BUCKET_OPTIMIZED,
                "versions": BUCKET_VERSIONS
            }
        }

        try:
            summary["split"] = self._load_json_from_minio("split_meta.json")
        except Exception:
            logger.warning(f"Could not load split metadata for job {self.job_id}")

        return summary


def create_dataset_manager(job_id: str) -> DatasetManager:
    """创建数据集管理器"""
    return DatasetManager(job_id)