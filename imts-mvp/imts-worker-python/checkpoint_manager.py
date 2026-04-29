"""
Checkpointer 模块 - 工作流状态快照管理

提供工作流断点恢复所需的持久化状态存储。
使用 MySQL 作为存储后端。
"""

import json
import logging
import os
import tempfile
from typing import Optional, List

from sqlalchemy import text

logger = logging.getLogger(__name__)

# 全局 session factory（由 main.py 初始化时设置）
_async_session_factory = None


def set_session_factory(factory):
    """设置全局数据库会话工厂"""
    global _async_session_factory
    _async_session_factory = factory


def get_session_factory():
    """获取全局数据库会话工厂"""
    return _async_session_factory


# ============================================================================
# 子步骤级别断点恢复
# ============================================================================

async def save_substep_checkpoint(
    job_id: str,
    iteration: int,
    node_name: str,
    substep_key: str,
    substep_value: any
) -> bool:
    """
    保存节点子步骤的断点状态

    Args:
        job_id: 任务 ID
        iteration: 当前迭代轮次
        node_name: 节点名称 (data_optimization / training / evaluation)
        substep_key: 子步骤标识 (如 "evaluated_sample_ids", "data_cleaning_steps")
        substep_value: 子步骤状态值

    Returns:
        True if successful, False otherwise
    """
    if _async_session_factory is None:
        logger.warning(f"Session factory not set, skipping substep checkpoint save")
        return False

    async with _async_session_factory() as session:
        try:
            # 获取现有 checkpoint
            result = await session.execute(
                text("SELECT checkpoint_json FROM t_workflow_checkpoint WHERE job_id = :job_id"),
                {"job_id": job_id}
            )
            row = result.fetchone()

            if row:
                # 更新现有 checkpoint
                existing = json.loads(row[0])
                if "substep_state" not in existing:
                    existing["substep_state"] = {}
                if node_name not in existing["substep_state"]:
                    existing["substep_state"][node_name] = {}
                existing["substep_state"][node_name][substep_key] = substep_value
                existing["substep_state"][node_name]["_iteration"] = iteration

                new_json = json.dumps(existing, ensure_ascii=False)
                await session.execute(
                    text("UPDATE t_workflow_checkpoint SET checkpoint_json = :json, updated_at = NOW() WHERE job_id = :job_id"),
                    {"json": new_json, "job_id": job_id}
                )
            else:
                # 创建新的 checkpoint
                new_checkpoint = {
                    "substep_state": {
                        node_name: {
                            substep_key: substep_value,
                            "_iteration": iteration
                        }
                    }
                }
                new_json = json.dumps(new_checkpoint, ensure_ascii=False)
                await session.execute(
                    text("""
                        INSERT INTO t_workflow_checkpoint (job_id, checkpoint_json, completed_node, current_iteration, updated_at)
                        VALUES (:job_id, :json, '', 1, NOW())
                    """),
                    {"job_id": job_id, "json": new_json}
                )

            await session.commit()
            logger.info(f"Substep checkpoint saved: job_id={job_id}, node={node_name}, {substep_key}={substep_value}")
            return True
        except Exception as e:
            await session.rollback()
            logger.error(f"Save substep checkpoint failed: {e}")
            return False


async def get_substep_checkpoint(job_id: str, node_name: str, iteration: int) -> Optional[dict]:
    """
    获取节点子步骤的断点状态

    Args:
        job_id: 任务 ID
        node_name: 节点名称
        iteration: 当前迭代轮次

    Returns:
        子步骤状态字典，如果不存在则返回 None
    """
    if _async_session_factory is None:
        return None

    async with _async_session_factory() as session:
        try:
            result = await session.execute(
                text("SELECT checkpoint_json FROM t_workflow_checkpoint WHERE job_id = :job_id"),
                {"job_id": job_id}
            )
            row = result.fetchone()

            if not row:
                return None

            checkpoint = json.loads(row[0])
            substep_state = checkpoint.get("substep_state", {}).get(node_name, {})

            # 检查 iteration 是否匹配
            if substep_state.get("_iteration") == iteration:
                # 返回时移除内部字段
                return {k: v for k, v in substep_state.items() if not k.startswith("_")}

            return None
        except Exception as e:
            logger.error(f"Get substep checkpoint failed: {e}")
            return None


async def filter_evaluated_samples(
    dataset_path: str,
    evaluated_sample_ids: List[str]
) -> Optional[str]:
    """
    从数据集中过滤掉已评估的样本，返回过滤后的临时文件路径

    Args:
        dataset_path: 原始数据集路径
        evaluated_sample_ids: 已评估的样本 ID 列表

    Returns:
        过滤后数据集的临时文件路径，失败返回 None
    """
    if not evaluated_sample_ids or not os.path.exists(dataset_path):
        return None

    try:
        import pandas as pd

        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        else:
            return None

        # 生成 sample_id
        df["_sample_id"] = [f"sample_{i}" for i in range(len(df))]

        # 过滤
        filtered_df = df[~df["_sample_id"].isin(evaluated_sample_ids)]

        if len(filtered_df) == 0:
            logger.info("All samples have been evaluated")
            return None

        # 保存到临时文件
        temp_fd, temp_path = tempfile.mkstemp(suffix='.csv')
        os.close(temp_fd)

        if dataset_path.endswith('.csv'):
            filtered_df.to_csv(temp_path, index=False)
        else:
            filtered_df.to_json(temp_path, index=False)

        logger.info(f"Filtered dataset saved: {len(filtered_df)}/{len(df)} samples remaining")
        return temp_path

    except Exception as e:
        logger.error(f"Filter evaluated samples failed: {e}")
        return None


async def save_checkpoint(job_id: str, state: dict, completed_node: str) -> bool:
    """
    保存工作流状态快照到数据库

    Args:
        job_id: 任务 ID (作为 thread_id 使用)
        state: AgentState 字典
        completed_node: 最后完成的节点名称 (data_optimization / training / evaluation)

    Returns:
        True if successful, False otherwise
    """
    if _async_session_factory is None:
        logger.warning(f"Session factory not set, skipping checkpoint save for job_id={job_id}")
        return False

    async with _async_session_factory() as session:
        try:
            checkpoint_json = json.dumps(state, ensure_ascii=False)
            current_iteration = state.get("current_iteration", 1)

            # 使用 REPLACE INTO 实现 upsert
            await session.execute(
                text("""
                    INSERT INTO t_workflow_checkpoint (job_id, checkpoint_json, completed_node, current_iteration, updated_at)
                    VALUES (:job_id, :checkpoint_json, :completed_node, :current_iteration, NOW())
                    ON DUPLICATE KEY UPDATE
                        checkpoint_json = VALUES(checkpoint_json),
                        completed_node = VALUES(completed_node),
                        current_iteration = VALUES(current_iteration),
                        updated_at = NOW()
                """),
                {
                    "job_id": job_id,
                    "checkpoint_json": checkpoint_json,
                    "completed_node": completed_node,
                    "current_iteration": current_iteration
                }
            )
            await session.commit()
            logger.info(f"Checkpoint saved: job_id={job_id}, completed_node={completed_node}, iteration={current_iteration}")
            return True
        except Exception as e:
            await session.rollback()
            logger.error(f"Save checkpoint failed: job_id={job_id}, error={e}")
            return False


async def get_checkpoint(job_id: str) -> Optional[dict]:
    """
    获取工作流状态快照

    Args:
        job_id: 任务 ID

    Returns:
        包含 checkpoint 数据的字典，如果不存在则返回 None
        返回格式: {
            "checkpoint_json": str,  # AgentState 的 JSON 字符串
            "completed_node": str,
            "current_iteration": int,
            "state": dict  # 解析后的 AgentState
        }
    """
    if _async_session_factory is None:
        logger.warning(f"Session factory not set, cannot get checkpoint for job_id={job_id}")
        return None

    async with _async_session_factory() as session:
        try:
            result = await session.execute(
                text("""
                    SELECT checkpoint_json, completed_node, current_iteration
                    FROM t_workflow_checkpoint
                    WHERE job_id = :job_id
                """),
                {"job_id": job_id}
            )
            row = result.fetchone()

            if row is None:
                return None

            checkpoint_json = row[0]
            completed_node = row[1]
            current_iteration = row[2]

            # 解析 state
            try:
                state = json.loads(checkpoint_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse checkpoint JSON for job_id={job_id}: {e}")
                return None

            return {
                "checkpoint_json": checkpoint_json,
                "completed_node": completed_node,
                "current_iteration": current_iteration,
                "state": state
            }
        except Exception as e:
            logger.error(f"Get checkpoint failed: job_id={job_id}, error={e}")
            return None


async def clear_checkpoint(job_id: str) -> bool:
    """
    清除工作流状态快照

    Args:
        job_id: 任务 ID

    Returns:
        True if successful, False otherwise
    """
    if _async_session_factory is None:
        logger.warning(f"Session factory not set, cannot clear checkpoint for job_id={job_id}")
        return False

    async with _async_session_factory() as session:
        try:
            await session.execute(
                text("DELETE FROM t_workflow_checkpoint WHERE job_id = :job_id"),
                {"job_id": job_id}
            )
            await session.commit()
            logger.info(f"Checkpoint cleared: job_id={job_id}")
            return True
        except Exception as e:
            await session.rollback()
            logger.error(f"Clear checkpoint failed: job_id={job_id}, error={e}")
            return False


async def is_job_running(job_id: str) -> bool:
    """
    检查任务是否处于 RUNNING 状态

    Args:
        job_id: 任务 ID

    Returns:
        True if job is RUNNING, False otherwise
    """
    if _async_session_factory is None:
        return False

    async with _async_session_factory() as session:
        try:
            result = await session.execute(
                text("SELECT status FROM t_job_instance WHERE job_id = :job_id"),
                {"job_id": job_id}
            )
            row = result.fetchone()
            if row is None:
                return False
            return row[0] == "RUNNING"
        except Exception as e:
            logger.error(f"Check job running failed: job_id={job_id}, error={e}")
            return False
