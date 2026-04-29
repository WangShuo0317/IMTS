"""
Nodes 模块 - 工具函数和共享状态

注意：业务逻辑节点已移至 graph_engine.py
本模块只保留跨节点共享的工具函数和状态。
"""

import asyncio
import json
import logging
import os
import traceback
import platform
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis
import redis as sync_redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 全局状态
# ============================================================================

redis_client: Optional[aioredis.Redis] = None
sync_redis_client: Optional[sync_redis.Redis] = None
async_session_factory = None


def set_redis_client(client):
    """设置全局异步 Redis 客户端"""
    global redis_client
    redis_client = client


def set_sync_redis_client(client):
    """设置全局同步 Redis 客户端（用于 Pub/Sub，与 Spring ReactiveRedisTemplate 兼容）"""
    global sync_redis_client
    sync_redis_client = client


def set_session_factory(factory):
    """设置全局数据库会话工厂"""
    global async_session_factory
    async_session_factory = factory
    # 同步设置到 checkpoint_manager
    from checkpoint_manager import set_session_factory as cm_set_factory
    cm_set_factory(factory)


def get_redis_client():
    """获取全局异步 Redis 客户端"""
    return redis_client


def get_sync_redis_client():
    """获取全局同步 Redis 客户端"""
    return sync_redis_client


# ============================================================================
# 工具函数
# ============================================================================

async def save_report(job_id: str, iteration: int, stage: str, content: dict):
    """
    保存阶段报告到数据库

    Args:
        job_id: 任务 ID
        iteration: 当前迭代轮次
        stage: 阶段名称 (DATA_OPTIMIZATION, TRAINING, EVALUATION)
        content: 阶段结果内容 (dict)
    """
    from sqlalchemy import text

    global async_session_factory

    if async_session_factory is None:
        logger.warning("Session factory not set, skipping report save")
        return

    async with async_session_factory() as session:
        try:
            await session.execute(
                text("""
                    INSERT INTO t_job_report (job_id, iteration_round, stage, content_json, created_at)
                    VALUES (:job_id, :iteration, :stage, :content, NOW())
                """),
                {
                    "job_id": job_id,
                    "iteration": iteration,
                    "stage": stage,
                    "content": json.dumps(content, ensure_ascii=False)
                }
            )
            await session.commit()
            logger.info(f"Report saved: job={job_id}, stage={stage}, iteration={iteration}")
        except Exception as e:
            await session.rollback()
            logger.error(f"Save report failed: {e}")


async def update_job_status_in_db(job_id: str, status: str, iteration: int = None):
    """
    更新任务状态到数据库

    Args:
        job_id: 任务 ID
        status: 新状态 (RUNNING, SUCCESS, FAILED)
        iteration: 当前迭代轮次 (可选)
    """
    from sqlalchemy import text

    global async_session_factory

    if async_session_factory is None:
        logger.warning("Session factory not set, skipping job status update")
        return

    async with async_session_factory() as session:
        try:
            query = "UPDATE t_job_instance SET status = :status, updated_at = NOW()"
            params = {"status": status, "job_id": job_id}

            if iteration is not None:
                query += ", current_iteration = :iteration"
                params["iteration"] = iteration

            query += " WHERE job_id = :job_id"
            await session.execute(text(query), params)
            await session.commit()
            logger.info(f"Job status updated: job={job_id}, status={status}, iteration={iteration}")
        except Exception as e:
            await session.rollback()
            logger.error(f"Update job status failed: {e}")


async def save_dataset_to_mysql(
    job_id: str,
    user_id: int,
    name: str,
    storage_path: str,
    row_count: int,
    file_name: str = None,
    file_size: int = 0,
    file_type: str = "CSV",
    description: str = None
) -> Optional[int]:
    """
    保存数据集信息到 MySQL t_dataset 表

    Args:
        job_id: 任务ID
        user_id: 用户ID
        name: 数据集名称
        storage_path: MinIO 存储路径 (如 minio://bucket/key)
        row_count: 数据行数
        file_name: 原始文件名
        file_size: 文件大小(字节)
        file_type: 文件类型 CSV/JSONL/JSON
        description: 数据集描述

    Returns:
        数据集ID，失败返回None
    """
    from sqlalchemy import text

    global async_session_factory

    if async_session_factory is None:
        logger.warning("Session factory not set, skipping dataset save - user_id={}, name={}".format(user_id, name))
        return None

    async with async_session_factory() as session:
        try:
            result = await session.execute(
                text("""
                    INSERT INTO t_dataset
                    (user_id, name, description, file_name, file_size, file_type, storage_path, row_count, columns, status, created_at)
                    VALUES (:user_id, :name, :description, :file_name, :file_size, :file_type, :storage_path, :row_count, :columns, 'ACTIVE', NOW())
                """),
                {
                    "user_id": user_id,
                    "name": name,
                    "description": description or f"数据集优化迭代",
                    "file_name": file_name or f"dataset_{job_id}.csv",
                    "file_size": file_size,
                    "file_type": file_type,
                    "storage_path": storage_path,
                    "row_count": row_count,
                    "columns": json.dumps({"columns": []})  # 默认空列信息
                }
            )
            await session.commit()
            dataset_id = result.lastrowid
            logger.info(f"Dataset saved to MySQL: id={dataset_id}, name={name}")
            return dataset_id
        except Exception as e:
            await session.rollback()
            logger.error(f"Save dataset to MySQL failed: user_id={user_id}, name={name}, error={e}")
            return None


async def save_dataset_version_to_mysql(
    job_id: str,
    user_id: int,
    parent_id: int,
    version_tag: str,
    oss_url: str,
    row_count: int,
    description: str = None,
    prev_version_id: int = None,
) -> Optional[int]:
    """
    保存数据集版本到 MySQL t_dataset_version 表

    Args:
        job_id: 任务ID
        user_id: 用户ID
        parent_id: 父数据集ID (NULL表示初始版本)
        version_tag: 版本标签 (V0, V1, V2...)
        oss_url: MinIO 存储路径
        row_count: 数据行数
        description: 变更说明
        prev_version_id: P1-2 上一个版本ID，形成 V0→V1→V2 链路

    Returns:
        版本ID，失败返回None
    """
    from sqlalchemy import text

    global async_session_factory

    if async_session_factory is None:
        logger.warning("Session factory not set, skipping version save")
        return None

    async with async_session_factory() as session:
        try:
            result = await session.execute(
                text("""
                    INSERT INTO t_dataset_version
                    (user_id, job_id, parent_id, version_tag, oss_url,
                     row_count, description, prev_version_id, created_at)
                    VALUES (:user_id, :job_id, :parent_id, :version_tag, :oss_url,
                     :row_count, :description, :prev_version_id, NOW())
                """),
                {
                    "user_id": user_id,
                    "job_id": job_id,
                    "parent_id": parent_id,
                    "version_tag": version_tag,
                    "oss_url": oss_url,
                    "row_count": row_count,
                    "description": description or f"版本 {version_tag}",
                    "prev_version_id": prev_version_id,
                }
            )
            await session.commit()
            version_id = result.lastrowid
            logger.info(f"Dataset version saved to MySQL: id={version_id}, tag={version_tag}, prev={prev_version_id}")
            return version_id
        except Exception as e:
            await session.rollback()
            logger.error(f"Save dataset version to MySQL failed: {e}")
            return None


async def get_latest_dataset_id(job_id: str, user_id: int) -> Optional[int]:
    """
    获取指定任务的最新数据集ID

    Args:
        job_id: 任务ID
        user_id: 用户ID

    Returns:
        数据集ID，不存在返回None
    """
    from sqlalchemy import text

    global async_session_factory

    if async_session_factory is None:
        return None

    async with async_session_factory() as session:
        try:
            result = await session.execute(
                text("""
                    SELECT id FROM t_dataset
                    WHERE job_id = :job_id AND user_id = :user_id AND status = 'ACTIVE'
                    ORDER BY created_at DESC LIMIT 1
                """),
                {"job_id": job_id, "user_id": user_id}
            )
            row = result.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Get latest dataset ID failed: {e}")
            return None


# ============================================================================
# P0-4: 异常上下文快照
# ============================================================================

async def save_error_context(
    job_id: str, iteration: int, node: str, state: dict, exc: Exception
):
    """在节点异常时保存完整上下文快照到 t_job_report。

    记录 AgentState、异常类型/traceback、系统资源信息，
    使事后能够精确回溯崩溃时的完整现场。
    """
    global async_session_factory

    if async_session_factory is None:
        logger.warning("Session factory not set, skipping error context save")
        return

    context = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "state_snapshot": {
            k: str(v)[:500] if not isinstance(v, (dict, list, int, float, bool, str)) else v
            for k, v in state.items()
        },
        "system_info": _collect_system_info(),
        "timestamp": datetime.now().isoformat(),
    }

    async with async_session_factory() as session:
        try:
            await session.execute(
                text("""
                    INSERT INTO t_job_report (job_id, iteration_round, stage, content_json, created_at)
                    VALUES (:job_id, :iteration, :stage, :content, NOW())
                """),
                {
                    "job_id": job_id,
                    "iteration": iteration,
                    "stage": f"{node}_ERROR_CONTEXT",
                    "content": json.dumps(context, ensure_ascii=False),
                },
            )
            await session.commit()
            logger.info(f"Error context saved: job={job_id}, node={node}")
        except Exception as e:
            await session.rollback()
            logger.error(f"Save error context failed: {e}")


def _collect_system_info() -> dict:
    """收集当前系统资源信息。"""
    info = {"platform": platform.platform()}
    try:
        import psutil
        info["memory_available_mb"] = psutil.virtual_memory().available // (1024 * 1024)
        info["memory_total_mb"] = psutil.virtual_memory().total // (1024 * 1024)
        info["disk_free_gb"] = psutil.disk_usage(os.getcwd()).free // (1024 ** 3)
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    except Exception:
        info["psutil_not_available"] = True
    return info


# ============================================================================
# P1-1: 模型资产持久化
# ============================================================================

async def save_model_asset_to_mysql(
    job_id: str,
    user_id: int,
    iteration: int,
    base_model_name: str,
    lora_oss_path: str,
    evaluation_score: float,
    is_best: bool = False,
) -> Optional[int]:
    """保存模型资产到 t_model_asset 表，实现模型↔数据集的追溯。"""
    from sqlalchemy import text

    global async_session_factory

    if async_session_factory is None:
        logger.warning("Session factory not set, skipping model asset save")
        return None

    async with async_session_factory() as session:
        try:
            # 如果 is_best=True，先取消该 job 之前的 is_best
            if is_best:
                await session.execute(
                    text("UPDATE t_model_asset SET is_best = 0 WHERE job_id = :job_id"),
                    {"job_id": job_id},
                )

            result = await session.execute(
                text("""
                    INSERT INTO t_model_asset
                    (user_id, job_id, iteration_round, base_model_name,
                     lora_oss_path, evaluation_score, is_best, created_at)
                    VALUES (:user_id, :job_id, :iteration, :model_name,
                     :lora_path, :score, :is_best, NOW())
                """),
                {
                    "user_id": user_id,
                    "job_id": job_id,
                    "iteration": iteration,
                    "model_name": base_model_name,
                    "lora_path": lora_oss_path,
                    "score": evaluation_score,
                    "is_best": 1 if is_best else 0,
                },
            )
            await session.commit()
            asset_id = result.lastrowid
            logger.info(
                f"Model asset saved: id={asset_id}, job={job_id}, iter={iteration}, "
                f"score={evaluation_score:.1f}, is_best={is_best}"
            )
            return asset_id
        except Exception as e:
            await session.rollback()
            logger.error(f"Save model asset failed: {e}")
            return None

# 为了向后兼容，保留这些导入
# 新代码应直接从 graph_engine 导入
async def async_data_opt(state: dict) -> dict:
    """兼容性函数：已废弃，请使用 graph_engine 中的节点"""
    from graph_engine import data_optimization_node
    return await data_optimization_node(state)


async def async_train(state: dict) -> dict:
    """兼容性函数：已废弃，请使用 graph_engine 中的节点"""
    from graph_engine import training_node
    return await training_node(state)


async def async_eval(state: dict) -> dict:
    """兼容性函数：已废弃，请使用 graph_engine 中的节点"""
    from graph_engine import evaluation_node
    return await evaluation_node(state)


async def async_analyze_eval_feedback(state: dict) -> dict:
    """兼容性函数：反馈分析已集成到 evaluation_node"""
    return state


__all__ = [
    "redis_client",
    "sync_redis_client",
    "async_session_factory",
    "set_redis_client",
    "set_sync_redis_client",
    "set_session_factory",
    "get_redis_client",
    "get_sync_redis_client",
    "save_report",
    "update_job_status_in_db",
    "save_dataset_to_mysql",
    "save_dataset_version_to_mysql",
    "get_latest_dataset_id",
    "save_error_context",
    "save_model_asset_to_mysql",
    # 兼容性导出
    "async_data_opt",
    "async_train",
    "async_eval",
    "async_analyze_eval_feedback",
]
