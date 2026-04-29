"""
LangGraph 核心引擎 - IMTS 工作流编排器

三个核心节点直接定义为 LangGraph 节点：
1. data_optimization - 数据优化智能体 (deepagents)
2. training - 模型训练
3. evaluation - 模型评估智能体 (AutoGen)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END

from training_service import DEFAULT_BASE_MODEL

logger = logging.getLogger(__name__)


def _convert_to_alpaca_format(dataset_path: str) -> str:
    """Convert a dataset to Alpaca format {instruction, input, output} for LLaMA-Factory.

    Handles various input formats:
    - {title, content} → {instruction=title, output=content}
    - {title, domain, content} → {instruction=title, input=domain, output=content}
    - {question, answer} → {instruction=question, output=answer}
    - Already Alpaca format → pass through

    Returns path to the converted file. If already in Alpaca format, returns original path.
    """
    import pandas as pd

    if dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith(".jsonl"):
        df = pd.read_json(dataset_path, lines=True)
    elif dataset_path.endswith(".json"):
        df = pd.read_json(dataset_path)
    else:
        logger.warning(f"Unsupported dataset format: {dataset_path}, skipping conversion")
        return dataset_path

    # Check if already in Alpaca format
    if "instruction" in df.columns and "output" in df.columns:
        logger.info(f"Dataset already in Alpaca format ({len(df)} rows), no conversion needed")
        return dataset_path

    # Convert based on available columns
    alpaca_data = []
    for _, row in df.iterrows():
        # Determine instruction field
        instruction = str(row.get("instruction", row.get("question", row.get("title", row.get("input", "")))))
        # Determine input field (optional context)
        input_text = str(row.get("input", row.get("domain", row.get("context", ""))))
        if input_text == "nan" or not input_text.strip():
            input_text = ""
        # Determine output field
        output = str(row.get("output", row.get("answer", row.get("content", row.get("response", "")))))

        if not instruction.strip() and not output.strip():
            continue

        alpaca_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output,
        })

    if not alpaca_data:
        logger.warning(f"No valid samples after conversion, returning original path")
        return dataset_path

    # Write converted file alongside original
    base, ext = os.path.splitext(dataset_path)
    converted_path = f"{base}_alpaca.json"
    with open(converted_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Converted {len(alpaca_data)} samples to Alpaca format: {converted_path}")
    return converted_path

# Fix #3: 定义节点执行顺序，用于断点恢复判断
NODE_ORDER = {"data_optimization": 0, "training": 1, "evaluation": 2}

# 每个节点最大重试次数
DEFAULT_MAX_NODE_RETRIES = 3

# 数据优化智能体总超时（秒）— 5步流程，每步约1-2分钟，留足余量
DATA_OPT_TIMEOUT = 600


def _classify_node_error(error_msg: str) -> str:
    """将节点错误分类为可重试或不可重试类型。

    不可重试的错误（如磁盘空间不足、模型下载失败）重试也不会成功，
    直接标记为 fatal_error 终止任务。
    可重试的错误（如网络超时、LLM API 限流）可以重试。
    """
    if not error_msg:
        return "RETRYABLE"

    # 不可重试：资源耗尽类错误
    non_retryable_patterns = [
        "ENOSPC", "No space left", "disk full",
        "OOM", "out of memory", "CUDA out of memory",
    ]
    for pattern in non_retryable_patterns:
        if pattern in error_msg:
            return "NON_RETRYABLE"

    return "RETRYABLE"


def _make_fatal(node_name: str, error_msg: str, max_retries: int) -> dict:
    """构造致命错误返回值，用于重试耗尽或不可重试错误。"""
    msg = f"[{node_name}]任务终止: {error_msg[:300]}"
    logger.error(msg)
    return {"fatal_error": msg, "error": error_msg}

# P0-1: 连续评分无提升的迭代次数阈值，超过即强制终止
PLATEAU_THRESHOLD = 2

# P0-2: 评估维度中文映射，用于构造针对性优化 prompt
DIM_CN = {
    "fact_accuracy": "事实准确性",
    "logic_consistency": "逻辑一致性",
    "completeness": "完整性",
    "relevance": "相关性",
}


def _build_optimization_prompt(state: AgentState, iteration: int, dataset_path: str) -> str:
    """P0-2: Construct dataset optimization prompt with feedback loop."""
    output_path_json = '{"output_path": 最终返回的path字段}'

    base = f"数据集路径: {dataset_path}\n"

    if iteration == 1:
        # 首轮：聚类打标、长尾发现与基础清洗
        base += (
            f"首轮优化目标：发现长尾数据，设置类别标签并清洗。\n"
            f"请按以下顺序执行：\n"
            f"1. 调用 cluster_and_find_weak_categories(dataset_path='{dataset_path}') 对数据进行向量化聚类、自动打标并发现稀缺数据类别。\n"
            f"2. 针对返回的稀缺类别 (scarce_categories_needing_augmentation)，请利用你自身的知识生成 5-10 条相关的新样本，补充到数据集中。\n"
            f"3. 对聚类并增强后的新数据集依次调用 data_loader -> exact_deduplicate -> batch_clean -> data_validator 进行流水线清洗。\n"
            f"4. 最后直接回复 JSON {output_path_json}，不需要调用 write_file。"
        )
    else:
        # 后续轮次：基于上一轮评估进行定向增强
        prev_eval = state.get("eval_result", {})
        prev_score = prev_eval.get("overall_score", 0)
        suggestions = prev_eval.get("suggestions", [])
        metrics = prev_eval.get("detailed_metrics", {})

        DIM_CN = {"fact_accuracy": "事实准确性", "logic_consistency": "逻辑一致性", "completeness": "完整性", "relevance": "相关性"}
        weak_dims = []
        for dim, val in metrics.items():
            if val < 75 and dim in DIM_CN:
                weak_dims.append(f"{DIM_CN[dim]}({val:.1f}分)")

        focus = "，重点关注：" + "、".join(weak_dims) if weak_dims else ""
        sug_str = json.dumps(suggestions[:3], ensure_ascii=False) if suggestions else "无"

        base += (
            f"本轮优化目标：根据评估结果补齐短板类别。\n"
            f"上一轮综合评分: {prev_score:.1f}/100{focus}。\n"
            f"评估建议: {sug_str}\n\n"
            f"请按以下顺序执行：\n"
            f"1. 仔细阅读评估建议中提到的“表现较差的类别 (weak_categories)”或其它逻辑缺陷。\n"
            f"2. 专门针对这些表现差的类别，调用相关数据生成/增强能力，生成一批高质量、高逻辑性的新训练数据并合并。\n"
            f"3. 调用 data_loader(path='{dataset_path}') -> exact_deduplicate -> batch_clean -> data_validator 清洗流水线。\n"
            f"4. 最后直接回复 JSON {output_path_json}。"
        )

    return base


def _should_skip_node(checkpoint: dict, node_name: str, current_iteration: int) -> bool:
    """基于 checkpoint 的 completed_node 和节点执行顺序判断是否应跳过。

    如果 checkpoint 的 completed_node 在执行顺序上 >= 当前节点，
    且当前迭代 <= checkpoint 迭代，则跳过（此节点的工作已完成）。
    """
    if not checkpoint:
        return False

    checkpoint_iteration = checkpoint.get("current_iteration", 1)
    completed_node = checkpoint.get("completed_node", "")

    if current_iteration > checkpoint_iteration:
        return False

    completed_order = NODE_ORDER.get(completed_node, -1)
    node_order = NODE_ORDER.get(node_name, -1)

    return completed_order >= node_order


class AgentState(TypedDict):
    """LangGraph 工作流状态"""
    job_id: str
    user_id: int
    mode: str
    target_prompt: str
    dataset_path: str
    augmented_dataset_path: str
    model_name: str  # 要训练的模型
    max_iterations: int
    current_iteration: int
    target_score: float

    # 控制智能体的 LLM 配置
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model_name: str | None

    # 各节点结果
    data_opt_result: dict
    train_result: dict
    eval_result: dict

    # 反馈分析
    weak_areas: dict
    augmentation_result: dict

    # P0-1: 评分历史与停滞检测
    score_history: list
    stagnation_count: int

    # 节点级重试与致命错误
    node_retry_count: int           # 当前节点已重试次数
    max_node_retries: int           # 每个节点最大重试次数（默认3）
    fatal_error: str | None         # 不可恢复的错误，任务直接终止

    # 状态
    status: str
    passed: bool
    error: str | None


# ============================================================================
# 三个核心节点实现
# ============================================================================

async def data_optimization_node(state: AgentState):
    """
    节点1: 数据优化智能体

    使用 deepagents 架构的 DataOptAgent 进行数据优化。
    包含数据集版本管理：原始数据保存、数据集分割（首轮迭代）、优化数据版本管理
    """
    from data_opt_agent import create_data_opt_agent
    from data_opt_agent.callback import ConsoleStreamHandler
    from message_types import MessageBuilder, Stage
    from langchain_core.messages import AIMessage
    from nodes import redis_client as global_redis_client, get_sync_redis_client, save_report, save_error_context, save_dataset_to_mysql, save_dataset_version_to_mysql, save_error_context
    from checkpoint_manager import get_checkpoint, save_checkpoint
    from dataset_manager import create_dataset_manager, BUCKET_OPTIMIZED

    job_id = state["job_id"]

    # === 上游错误检测：前一轮有致命错误则跳过 ===
    fatal_error = state.get("fatal_error")
    if fatal_error:
        logger.error(f"[DataOpt Node] 前一轮存在致命错误，跳过数据优化: {fatal_error[:200]}")
        return {"fatal_error": fatal_error, "error": fatal_error}

    # === Fix #3: 使用统一的节点跳过逻辑 ===
    checkpoint = await get_checkpoint(job_id)
    if _should_skip_node(checkpoint, "data_optimization", state.get("current_iteration", 1)):
        logger.info(f"[DataOpt Node] 跳过已完成的 iteration={checkpoint['current_iteration']}，恢复执行后续节点")
        checkpoint_state = checkpoint.get("state", {})
        return {
            "data_opt_result": checkpoint_state.get("data_opt_result", {}),
            "augmented_dataset_path": checkpoint_state.get("augmented_dataset_path", ""),
            "train_dataset_path": checkpoint_state.get("train_dataset_path", ""),
            "test_dataset_path": checkpoint_state.get("test_dataset_path", ""),
            "error": None
        }
    # === 断点恢复逻辑结束 ===
    user_id = state.get("user_id", state.get("userId", 1))
    iteration = state.get("current_iteration", 1)
    dataset_path = state.get("dataset_path", "/data/train.csv")
    llm_api_key = state.get("llm_api_key")
    llm_base_url = state.get("llm_base_url")
    llm_model_name = state.get("llm_model_name")
    logger.info(f"[DataOpt Node] job_id={job_id}, user_id={user_id}, iteration={iteration}, model={llm_model_name}, has_api_key={bool(llm_api_key)}")

    builder = MessageBuilder(job_id, global_redis_client, get_sync_redis_client())
    await builder.stage_start(Stage.DATA_OPTIMIZATION.value)

    logger.info(f"[迭代 {iteration}] 数据优化智能体开始执行...")

    # ========================
    # 数据集版本管理
    # ========================
    dm = create_dataset_manager(job_id)

    # 首次迭代：保存原始数据并分割训练/测试集
    if iteration == 1:
        logger.info(f"[迭代 1] 首次执行，进行数据集版本管理和数据分割...")

        # 保存原始数据集
        orig_version = dm.save_original_dataset(dataset_path)
        await builder.agent_thought("DataOptAgent", f"原始数据集已保存: {orig_version.original_rows} 条记录", progress=5)

        # 分割训练集和测试集 (10% 测试)
        train_version, test_version = dm.split_train_test(dataset_path, test_size=0.1, seed=42)
        await builder.agent_thought("DataOptAgent", f"数据集已分割: 训练集 {train_version.final_rows} 条, 测试集 {test_version.final_rows} 条", progress=10)
        logger.info(f"Dataset split: train={train_version.final_rows}, test={test_version.final_rows}")

    # 获取当前迭代的训练数据集路径（优先使用优化后的）
    train_dataset_path = dm.get_latest_train_dataset_path() or dataset_path
    test_dataset_path = dm.get_test_dataset_path()

    logger.info(f"[迭代 {iteration}] 训练数据: {train_dataset_path}")
    logger.info(f"[迭代 {iteration}] 测试数据: {test_dataset_path}")

    # 将数据集转换为 Alpaca 格式（LLaMA-Factory 所需格式）
    train_dataset_path = _convert_to_alpaca_format(train_dataset_path)
    logger.info(f"[迭代 {iteration}] Alpaca格式训练数据: {train_dataset_path}")

    try:
        max_retries = state.get("max_node_retries", DEFAULT_MAX_NODE_RETRIES)
        # 选择模型
        model = "anthropic:claude-sonnet-4-6"
        if llm_model_name:
            if llm_base_url:
                model = f"openai:{llm_model_name}"
            else:
                model = llm_model_name

        # 创建 deepagents 智能体
        agent = create_data_opt_agent(
            model=model,
            api_key=llm_api_key,
            base_url=llm_base_url
        )

        # 创建统一的流式回调处理器（同时输出到 Redis 和控制台）
        # 传入 stop_checker 让每次 LLM/工具调用前检查取消信号
        from main import should_stop_job
        stream_handler = ConsoleStreamHandler.for_redis(builder, stop_checker=lambda: should_stop_job(job_id))

        # 使用训练数据集（排除测试集）进行优化
        await builder.agent_thought("DataOptAgent", f"使用训练数据集进行优化: {train_dataset_path}", progress=15)

        # 设置 fallback dataset path，防止 LLM 并行工具调用时产生 placeholder
        from data_opt_agent.skills.state_utils import set_fallback_dataset_path
        set_fallback_dataset_path(train_dataset_path)

        # P0-2: 使用包含反馈闭环的针对性优化 prompt
        optimization_prompt = _build_optimization_prompt(state, iteration, train_dataset_path)

        # 带重试的 LLM 智能体调用
        result = None
        for attempt in range(1, max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    agent.ainvoke(
                        {"messages": [{"role": "user", "content": optimization_prompt}]},
                        config={"callbacks": [stream_handler], "recursion_limit": 25}
                    ),
                    timeout=DATA_OPT_TIMEOUT
                )
                break  # 成功，跳出重试循环
            except asyncio.TimeoutError:
                logger.error(f"[数据优化] 智能体执行超时({DATA_OPT_TIMEOUT}s, attempt={attempt}/{max_retries})")
                if attempt < max_retries:
                    await asyncio.sleep(2 * attempt)
                else:
                    await save_error_context(job_id, iteration, "data_optimization", state, TimeoutError(f"Agent timeout {DATA_OPT_TIMEOUT}s"))
                    return _make_fatal("data_optimization", f"智能体执行超时({DATA_OPT_TIMEOUT}s)", max_retries)
            except asyncio.CancelledError:
                logger.info(f"[数据优化] 任务被用户取消")
                return _make_fatal("data_optimization", "任务被用户取消", max_retries)
            except Exception as e:
                error_str = str(e)
                error_class = _classify_node_error(error_str)
                if error_class == "NON_RETRYABLE":
                    logger.error(f"[数据优化] 不可重试错误: {error_str[:200]}")
                    await save_error_context(job_id, iteration, "data_optimization", state, e)
                    return _make_fatal("data_optimization", error_str, max_retries)
                if attempt < max_retries:
                    logger.warning(f"[数据优化] 第{attempt}/{max_retries}次调用失败: {error_str[:200]}, 重试中...")
                    await asyncio.sleep(2 * attempt)  # 线性退避
                else:
                    logger.error(f"[数据优化] 重试耗尽({max_retries}次): {error_str[:200]}")
                    await save_error_context(job_id, iteration, "data_optimization", state, e)
                    return _make_fatal("data_optimization", f"重试耗尽({max_retries}次): {error_str}", max_retries)

        # 从结果中提取最终响应和工具执行统计
        final_message = ""
        extracted_output_path = None  # 添加：提取 output_path
        total_original_rows = 0
        total_final_rows = 0
        total_null_removed = 0
        total_duplicates_removed = 0
        total_augmented = 0
        validation_passed = False

        if result and "messages" in result:
            import json  # 确保 json 已导入
            # 从消息历史中提取工具执行结果
            for msg in result.get("messages", []):
                # 检查 ToolMessage 中的工具返回结果
                if hasattr(msg, 'type') and 'tool' in str(msg.type).lower():
                    content = getattr(msg, 'content', '') or ''
                    # 解析工具返回的 JSON
                    try:
                        if isinstance(content, list):
                            for item in content:
                                if hasattr(item, 'text'):
                                    text = item.text
                                    if 'original_rows' in text or 'output_path' in text:
                                        data = json.loads(text)
                                        total_original_rows += data.get('total_samples', data.get('original_rows', 0))
                                        total_final_rows += data.get('processed_samples', data.get('final_rows', 0))
                                        total_duplicates_removed += data.get('duplicates_removed', 0) + data.get('exact_duplicates_removed', 0) + data.get('semantic_duplicates_removed', 0)
                                        total_null_removed += data.get('null_values_removed', data.get('cleaned_samples', 0))
                                        if 'augment' in text.lower():
                                            total_augmented += data.get('augmented', 0)
                                        validation_passed = data.get('validation_passed', validation_passed) or data.get('status') == 'success'
                                        # 提取 output_path
                                        if 'output_path' in data and not extracted_output_path:
                                            extracted_output_path = data.get('output_path')
                    except:
                        pass
                # 提取最终 AI 消息
                if isinstance(msg, AIMessage):
                    tool_calls = getattr(msg, 'tool_calls', None)
                    if not tool_calls:
                        final_message = msg.content if hasattr(msg, 'content') else str(msg)
                    # 从 AI 消息中尝试提取 output_path JSON
                    if not extracted_output_path and final_message:
                        try:
                            # 尝试从最终消息中提取 output_path
                            if '{"output_path"' in final_message or '"output_path"' in final_message:
                                import re
                                match = re.search(r'"output_path"\s*:\s*"([^"]+)"', final_message)
                                if match:
                                    extracted_output_path = match.group(1)
                        except:
                            pass

        # 如果没有提取到真实数据，不要伪造数字
        if total_original_rows == 0:
            logger.warning(f"[迭代 {iteration}] 数据优化智能体未产出有效数据指标，将验证输出文件")

        # === RC3: 验证优化输出文件是否包含足够数据 ===
        validated_output_path = extracted_output_path
        used_fallback = False
        if validated_output_path and os.path.exists(validated_output_path):
            try:
                import pandas as pd
                opt_df = pd.read_csv(validated_output_path) if validated_output_path.endswith('.csv') else pd.read_json(validated_output_path)
                actual_rows = len(opt_df)
                logger.info(f"[迭代 {iteration}] 优化输出文件实际行数: {actual_rows}")

                # 如果优化输出行数太少（低于5行），回退到原始数据集
                MIN_ROWS = 5
                if actual_rows < MIN_ROWS:
                    logger.warning(f"[迭代 {iteration}] 优化输出仅{actual_rows}行，低于最低要求{MIN_ROWS}行，回退到原始数据集")
                    validated_output_path = train_dataset_path
                    used_fallback = True
                elif total_original_rows > 0 and actual_rows < total_original_rows * 0.1:
                    logger.warning(f"[迭代 {iteration}] 优化输出{actual_rows}行不足原始{total_original_rows}行的10%，回退到原始数据集")
                    validated_output_path = train_dataset_path
                    used_fallback = True
                else:
                    # 输出有效，更新行数统计
                    if total_original_rows == 0 or total_final_rows == 0:
                        total_final_rows = actual_rows
            except Exception as e:
                logger.warning(f"[迭代 {iteration}] 无法验证优化输出文件: {e}, 回退到原始数据集")
                validated_output_path = train_dataset_path
                used_fallback = True
        elif not validated_output_path or not os.path.exists(validated_output_path):
            logger.warning(f"[迭代 {iteration}] 无有效输出路径，回退到原始数据集: {train_dataset_path}")
            validated_output_path = train_dataset_path
            used_fallback = True
        # === RC3 验证结束 ===

        # 将提取的 output_path 加入 result_data
        quality_score = 0.3 if used_fallback else (0.85 if validation_passed else 0.5)
        result_data = {
            "original_rows": total_original_rows,
            "final_rows": total_final_rows,
            "quality_score": quality_score,
            "null_removed": total_null_removed,
            "duplicates_removed": total_duplicates_removed,
            "augmented": total_augmented,
            "validation_passed": validation_passed,
            "agent_response": final_message[:1000] if final_message else "",
            "output_path": validated_output_path,  # RC3: 使用验证后的路径
        }

        await builder.stage_end(Stage.DATA_OPTIMIZATION.value, result_data)

        # 保存数据优化报告到数据库
        await save_report(job_id, iteration, "DATA_OPTIMIZATION", result_data)

        # 保存优化后的数据集版本（如果有输出路径）
        optimized_path = result_data.get("output_path")
        if optimized_path and os.path.exists(optimized_path):
            try:
                # 1. 保存到 MinIO (已有)
                opt_version = dm.save_optimized_dataset(optimized_path, iteration)
                await builder.agent_thought("DataOptAgent", f"优化数据已保存为版本 iter_{iteration}", progress=95)

                # 2. 持久化到 MySQL t_dataset 表
                user_id = state.get("user_id", state.get("userId", 1))  # 兼容 user_id 和 userId
                logger.info(f"[数据集保存] job_id={job_id}, user_id={user_id}, iteration={iteration}")

                dataset_name = f"优化数据集-迭代{iteration}"
                storage_path = f"minio://{BUCKET_OPTIMIZED}/{job_id}/iter_{iteration}_optimized.csv"

                # 获取文件大小
                file_size = os.path.getsize(optimized_path) if os.path.exists(optimized_path) else 0

                dataset_id = await save_dataset_to_mysql(
                    job_id=job_id,
                    user_id=user_id,
                    name=dataset_name,
                    storage_path=storage_path,
                    row_count=total_final_rows,
                    file_name=f"optimized_iter_{iteration}.csv",
                    file_size=file_size,
                    file_type="CSV",
                    description=f"数据优化迭代 {iteration}，原始行数: {total_original_rows}, 最终行数: {total_final_rows}"
                )

                # 3. 持久化到 MySQL t_dataset_version 表
                if dataset_id:
                    version_tag = f"V{iteration}"
                    await save_dataset_version_to_mysql(
                        job_id=job_id,
                        user_id=user_id,
                        parent_id=dataset_id,  # 指向当前数据集作为父版本
                        version_tag=version_tag,
                        oss_url=storage_path,
                        row_count=total_final_rows,
                        description=f"迭代 {iteration} 优化版本，原始行数: {total_original_rows}, 变化: {total_final_rows - total_original_rows}"
                    )
                    await builder.agent_thought("DataOptAgent", f"数据集信息已写入MySQL: dataset_id={dataset_id}, version={version_tag}", progress=98)
                else:
                    logger.error(f"[数据集保存] save_dataset_to_mysql 返回 None, user_id={user_id}, name={dataset_name}")
            except Exception as e:
                logger.error(f"[数据集保存] 保存优化数据集失败: {e}", exc_info=True)
        else:
            logger.warning(f"[数据集保存] 无输出路径或文件不存在: output_path={optimized_path}")

        logger.info(f"[迭代 {iteration}] 数据优化完成: original_rows={total_original_rows}, final_rows={total_final_rows}")

        # === 保存断点：data_optimization 完成 ===
        await save_checkpoint(job_id, state, "data_optimization")
        # === 断点保存结束 ===

        return {
            "data_opt_result": result_data,
            "augmented_dataset_path": result_data.get("output_path", train_dataset_path),
            "train_dataset_path": train_dataset_path,
            "test_dataset_path": test_dataset_path,
            "error": None
        }

    except Exception as e:
        logger.error(f"[迭代 {iteration}] 数据优化失败: {e}", exc_info=True)
        await save_error_context(job_id, iteration, "data_optimization", state, e)
        error_msg = str(e)
        error_result = {"error": error_msg}
        await builder.stage_end(Stage.DATA_OPTIMIZATION.value, error_result)
        await save_report(job_id, iteration, "DATA_OPTIMIZATION", error_result)
        return _make_fatal("data_optimization", error_msg, max_retries)


MODEL_ALIAS = {
    "Qwen3-8B": "/home/user/workspace/wangshuo/Models/Qwen3-8B",
    "Qwen3-7B": "/home/user/workspace/wangshuo/Models/Qwen3-8B",  # alias: 7B -> 8B
    "Qwen3-0.6B": "/home/user/workspace/wangshuo/Models/Qwen/Qwen3-0.6B",
    "Qwen2.5-1.5B": "/home/user/workspace/wangshuo/Models/Qwen2.5-1.5B",
}


def _resolve_model_path(model_name: str) -> str:
    """Resolve short model name to remote server local path."""
    if model_name in MODEL_ALIAS:
        return MODEL_ALIAS[model_name]
    # If it looks like a remote absolute path, use directly
    if model_name.startswith("/home/user/") or model_name.startswith("/"):
        return model_name
    # Default fallback
    return DEFAULT_BASE_MODEL


async def training_node(state: AgentState):
    """
    节点2: 模型训练 — 使用 LLaMA-Factory 在远程 GPU 服务器上执行真实 LoRA 微调

    流程：
    1. 确保基座模型已在远程服务器缓存
    2. 将优化后的数据集转换为 Alpaca 格式并上传到远程服务器
    3. 生成 LLaMA-Factory YAML 训练配置 + DeepSpeed ZeRO-2 配置
    4. 通过 SSH 执行 deepspeed --num_gpus=2 llamafactory-cli train
    5. 训练完成后启动 vLLM 服务 LoRA adapter 用于后续评估
    """
    from message_types import MessageBuilder, Stage
    from nodes import redis_client as global_redis_client, get_sync_redis_client, save_report, save_error_context
    from checkpoint_manager import get_checkpoint, save_checkpoint
    from training_service import run_remote_training, ensure_model_cached, DEFAULT_BASE_MODEL

    job_id = state["job_id"]
    iteration = state.get("current_iteration", 1)

    # === 上游错误检测：数据优化失败则跳过训练 ===
    upstream_error = state.get("error") or state.get("fatal_error")
    if upstream_error:
        logger.error(f"[迭代 {iteration}] 数据优化节点存在错误，跳过训练: {upstream_error[:200]}")
        return {"fatal_error": upstream_error, "error": upstream_error}

    # === Fix #3: 使用统一的节点跳过逻辑 ===
    checkpoint = await get_checkpoint(job_id)
    if _should_skip_node(checkpoint, "training", state.get("current_iteration", 1)):
        logger.info(f"[Training Node] 跳过已完成的 iteration={checkpoint['current_iteration']}，恢复执行后续节点")
        checkpoint_state = checkpoint.get("state", {})
        return {
            "train_result": checkpoint_state.get("train_result", {}),
            "error": None
        }
    # === 断点恢复逻辑结束 ===

    # Resolve base model — convert short names to HuggingFace paths
    model_name = state.get("model_name", "Qwen3-8B")
    base_model = _resolve_model_path(model_name)

    builder = MessageBuilder(job_id, global_redis_client, get_sync_redis_client())
    await builder.stage_start(Stage.TRAINING.value)

    logger.info(f"[迭代 {iteration}] 模型训练开始 (base={base_model})...")

    # 获取优化后的数据集路径（由 data_optimization_node 产出）
    dataset_path = state.get("augmented_dataset_path", state.get("dataset_path", ""))
    if not dataset_path:
        logger.warning(f"[迭代 {iteration}] 无优化数据集路径，回退到原始数据集")
        dataset_path = state.get("dataset_path", "/data/train.csv")

    try:
        max_retries = state.get("max_node_retries", DEFAULT_MAX_NODE_RETRIES)

        # Step 1: 确保基座模型已缓存
        await builder.agent_thought("TrainingAgent", f"检查基座模型 {base_model} 是否已缓存...", progress=5)
        await asyncio.to_thread(ensure_model_cached, base_model)
        await builder.agent_thought("TrainingAgent", f"基座模型 {base_model} 已就绪", progress=10)

        # Step 2: 计算训练参数（根据迭代轮次调整）
        num_epochs = min(3, max(1, iteration))  # 随迭代增加 epochs
        learning_rate = 1e-4 / (1 + 0.3 * (iteration - 1))  # 逐轮降低 lr
        batch_size = 2  # 2x RTX 5090 32GB, conservative for LoRA
        gradient_accumulation = 4

        await builder.agent_thought(
            "TrainingAgent",
            f"训练配置: LoRA rank=8, lr={learning_rate:.6f}, epochs={num_epochs}, batch={batch_size}×{gradient_accumulation}",
            progress=15
        )

        # Step 3: 执行远程训练（带重试）
        await builder.agent_thought("TrainingAgent", "上传数据集到远程 GPU 服务器...", progress=20)

        training_result = None
        for attempt in range(1, max_retries + 1):
            training_result = await asyncio.to_thread(
                run_remote_training,
                job_id=job_id,
                iteration=iteration,
                dataset_path=dataset_path,
                base_model=base_model,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gradient_accumulation=gradient_accumulation,
                sync_redis_client=get_sync_redis_client(),
            )
            if training_result.success:
                break  # 训练成功，跳出重试

            error_msg = training_result.error or "Unknown training error"
            error_class = _classify_node_error(error_msg)
            if error_class == "NON_RETRYABLE":
                logger.error(f"[迭代 {iteration}] 不可重试训练错误: {error_msg[:200]}")
                await save_error_context(job_id, iteration, "training", state, Exception(error_msg))
                error_result = {"model_name": base_model, "error": error_msg}
                await builder.stage_end(Stage.TRAINING.value, error_result)
                await save_report(job_id, iteration, "TRAINING", error_result)
                return _make_fatal("training", error_msg, max_retries)
            if attempt < max_retries:
                logger.warning(f"[迭代 {iteration}] 第{attempt}/{max_retries}次训练失败: {error_msg[:200]}, 重试中...")
                await builder.agent_thought("TrainingAgent", f"训练失败(第{attempt}次)，重试中...", progress=20)
            else:
                logger.error(f"[迭代 {iteration}] 训练重试耗尽({max_retries}次): {error_msg[:200]}")
                await save_error_context(job_id, iteration, "training", state, Exception(error_msg))
                error_result = {"model_name": base_model, "error": error_msg}
                await builder.stage_end(Stage.TRAINING.value, error_result)
                await save_report(job_id, iteration, "TRAINING", error_result)
                return _make_fatal("training", f"重试耗尽({max_retries}次): {error_msg}", max_retries)

        # Step 4: 训练成功 — 构造结果数据
        # Loss was streamed in real-time during training via emit_training_loss_sync.
        # Only send a final completion summary here.
        await builder.agent_thought(
            "TrainingAgent",
            f"训练完成! Final loss: {training_result.final_loss:.4f}, "
            f"总步数: {len(training_result.loss_history)}, "
            f"耗时: {training_result.training_time_seconds:.1f}s",
            progress=90
        )

        await builder.agent_thought(
            "TrainingAgent",
            f"LoRA adapter 保存于: {training_result.lora_path}",
            progress=95
        )

        # vLLM 服务状态
        if training_result.vllm_ready:
            await builder.agent_thought(
                "TrainingAgent",
                "vLLM inference server 已成功加载训练模型，准备评估",
                progress=98
            )
        else:
            await builder.agent_thought(
                "TrainingAgent",
                "vLLM 未成功加载训练模型，评估将无法进行",
                progress=98
            )

        result_data = training_result.to_dict()
        # 添加 lora_oss_path 供 evaluation_node 的 save_model_asset_to_mysql 使用
        result_data["lora_oss_path"] = f"s3://imts-models/{job_id}/iter_{iteration}/lora"
        result_data["model_name"] = base_model
        # vLLM LoRA model name — evaluation will use this in inference requests
        result_data["lora_model_name"] = "imts-lora"

        await builder.stage_end(Stage.TRAINING.value, result_data)
        await save_report(job_id, iteration, "TRAINING", result_data)
        logger.info(
            f"[迭代 {iteration}] 训练完成: loss={training_result.final_loss:.4f}, "
            f"time={training_result.training_time_seconds:.1f}s, lora={training_result.lora_path}"
        )

        # === 保存断点：training 完成 ===
        await save_checkpoint(job_id, state, "training")
        # === 断点保存结束 ===

        return {"train_result": result_data, "error": None}

    except Exception as e:
        logger.error(f"[迭代 {iteration}] 训练节点异常: {e}", exc_info=True)
        await save_error_context(job_id, iteration, "training", state, e)
        error_msg = str(e)
        error_result = {"model_name": base_model, "error": error_msg}
        await builder.stage_end(Stage.TRAINING.value, error_result)
        await save_report(job_id, iteration, "TRAINING", error_result)
        return _make_fatal("training", error_msg, max_retries)


async def evaluation_node(state: AgentState):
    """
    节点3: 模型评估智能体

    使用 AutoGen 架构的评估系统。
    """
    from message_types import MessageBuilder, Stage
    from nodes import redis_client as global_redis_client, get_sync_redis_client, save_report, save_error_context, save_model_asset_to_mysql
    from checkpoint_manager import get_checkpoint, save_checkpoint, get_substep_checkpoint, save_substep_checkpoint, filter_evaluated_samples

    job_id = state["job_id"]
    iteration = state.get("current_iteration", 1)
    target_score = state.get("target_score", 75.0)

    # === 上游错误检测：训练失败则跳过评估 ===
    upstream_error = state.get("error") or state.get("fatal_error")
    if upstream_error:
        logger.error(f"[迭代 {iteration}] 训练节点存在错误，跳过评估: {upstream_error[:200]}")
        return {"fatal_error": upstream_error, "error": upstream_error, "passed": False}

    # 评估的模型必须是训练节点产出的模型，如果没有成功训练则无法评估
    train_result = state.get("train_result", {})
    if not train_result or not train_result.get("success", False):
        logger.error(f"[迭代 {iteration}] 训练未产出有效模型，无法启动评估")
        return {"fatal_error": f"[迭代 {iteration}]训练未产出有效模型，无法评估", "error": "训练未产出有效模型", "passed": False}

    # vLLM 必须成功加载训练后的 LoRA 模型，否则评估无法使用训练产出
    # vLLM readiness check: if vllm_ready=False, try connecting before giving up
    # (the polling check can fail even when vLLM is actually running)
    if not train_result.get("vllm_ready", False):
        logger.warning(f"[迭代 {iteration}] vllm_ready=False, but will attempt to connect to vLLM before aborting")
        vllm_base_url = os.getenv("INFERENCE_BASE_URL", f"http://{SSH_HOST}:{REMOTE_VLLM_PORT}/v1")
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{vllm_base_url}/models")
                if resp.status_code == 200 and '"data"' in resp.text:
                    logger.info(f"[迭代 {iteration}] vLLM is actually running! Proceeding with evaluation.")
                else:
                    logger.error(f"[迭代 {iteration}] vLLM connection check failed: status={resp.status_code}")
                    return {"fatal_error": f"[迭代 {iteration}]vLLM未加载训练模型，无法评估", "error": "vLLM未加载训练模型", "passed": False}
        except Exception as conn_err:
            logger.error(f"[迭代 {iteration}] vLLM connection attempt failed: {conn_err}")
            return {"fatal_error": f"[迭代 {iteration}]vLLM未加载训练模型，无法评估", "error": "vLLM未加载训练模型", "passed": False}

    # === Fix #3: 使用统一的节点跳过逻辑 ===
    checkpoint = await get_checkpoint(job_id)
    if _should_skip_node(checkpoint, "evaluation", state.get("current_iteration", 1)):
        logger.info(f"[Evaluation Node] 跳过已完成的 iteration={checkpoint['current_iteration']}，恢复执行后续节点")
        checkpoint_state = checkpoint.get("state", {})
        checkpoint_eval_result = checkpoint_state.get("eval_result", {})
        return {
            "eval_result": checkpoint_eval_result,
            "passed": checkpoint_eval_result.get("passed", False),
            "current_iteration": checkpoint["current_iteration"] + 1,
            "error": None
        }
    # === 节点级别断点恢复结束 ===

    # === 子步骤断点恢复：检查是否有已评估的样本 ===
    resume_from_substep = False
    evaluated_sample_ids = []
    original_eval_result = {}

    substep_checkpoint = await get_substep_checkpoint(job_id, "evaluation", iteration)
    if substep_checkpoint:
        evaluated_sample_ids = substep_checkpoint.get("evaluated_sample_ids", [])
        if evaluated_sample_ids:
            logger.info(f"[Evaluation Node] 从子步骤恢复：已有 {len(evaluated_sample_ids)} 个样本已评估")
            resume_from_substep = True
            # 保存原始评估结果用于最终合并
            if checkpoint:
                original_eval_result = checkpoint.get("state", {}).get("eval_result", {})
    # === 子步骤断点恢复检查结束 ===

    # 从状态中获取测试数据集路径（由 data_optimization_node 分割并存储）
    test_dataset_path = state.get("test_dataset_path")
    train_dataset_path = state.get("train_dataset_path")

    builder = MessageBuilder(job_id, global_redis_client, get_sync_redis_client())

    # Check stop signal before starting evaluation
    sync_redis = get_sync_redis_client()
    if sync_redis and sync_redis.exists(f"imts_stop:{job_id}"):
        logger.info(f"[Evaluation] Stop signal detected before evaluation for job {job_id}")
        return {"fatal_error": "Job stopped by user", "error": "Job stopped by user", "passed": False}

    await builder.stage_start(Stage.EVALUATION.value)

    logger.info(f"[迭代 {iteration}] 模型评估开始...")

    # 用于跟踪本次评估的样本 ID（用于后续合并）
    samples_evaluated_this_run = []
    temp_filtered_path = None

    try:
        # 调用两模型评估智能体 (vLLM推理 + DashScope评估)
        from eval_agent.simple_eval import run_two_model_evaluation, load_test_data
        from training_service import _stop_vllm_server

        async def progress_callback(progress, message, role=None, speaker=None, is_streaming=False, msg_type="CHAT_MESSAGE"):
            # Check stop signal on every callback (called frequently during eval)
            sync_redis = get_sync_redis_client()
            if sync_redis and sync_redis.exists(f"imts_stop:{job_id}"):
                logger.info(f"[Evaluation] Stop signal detected for job {job_id}, aborting evaluation")
                _stop_vllm_server()
                raise asyncio.CancelledError(f"Job {job_id} stopped by user during evaluation")
            await builder.chat_message(
                role or "ARBITER",
                speaker or "Arbiter",
                message,
                is_streaming=is_streaming
            )

        # === 子步骤恢复：过滤掉已评估的样本 ===
        if resume_from_substep and evaluated_sample_ids and test_dataset_path:
            logger.info(f"[Evaluation Node] 过滤已评估的 {len(evaluated_sample_ids)} 个样本...")
            # 加载完整数据集获取样本 ID 列表
            full_test_data = await load_test_data(test_dataset_path, state)
            if full_test_data:
                # 找出未评估的样本
                evaluated_set = set(evaluated_sample_ids)
                remaining_samples = [s for s in full_test_data if s.get("id") not in evaluated_set]
                samples_evaluated_this_run = [s.get("id") for s in remaining_samples]
                logger.info(f"[Evaluation Node] 剩余 {len(remaining_samples)}/{len(full_test_data)} 个样本需要评估")

                if remaining_samples:
                    # 创建临时文件存储未评估的样本（用于评估）
                    temp_filtered_path = await filter_evaluated_samples(test_dataset_path, evaluated_sample_ids)
                    if temp_filtered_path:
                        test_dataset_path = temp_filtered_path
                else:
                    # 所有样本都已评估，跳过评估
                    logger.info(f"[Evaluation Node] 所有样本已评估完成，跳过评估")
                    # 更新 checkpoint 标记 evaluation 完成
                    state_after = dict(state)
                    state_after["eval_result"] = original_eval_result
                    state_after["passed"] = original_eval_result.get("passed", False)
                    state_after["current_iteration"] = iteration + 1
                    await save_checkpoint(job_id, state_after, "evaluation")
                    await save_substep_checkpoint(job_id, iteration, "evaluation", "evaluated_sample_ids", evaluated_sample_ids)
                    await builder.stage_end(Stage.EVALUATION.value, original_eval_result)
                    await save_report(job_id, iteration, "EVALUATION", original_eval_result)
                    return {
                        "eval_result": original_eval_result,
                        "passed": original_eval_result.get("passed", False),
                        "current_iteration": iteration + 1,
                        "error": None
                    }
        # === 子步骤过滤结束 ===

        # 优先使用分割后的测试数据集路径（不要直接修改 state["dataset_path"] 以免污染后续迭代的 checkpoint）
        eval_state = dict(state)
        if test_dataset_path:
            eval_state["dataset_path"] = test_dataset_path
            logger.info(f"[迭代 {iteration}] 使用分割后的测试数据集: {test_dataset_path}")
        elif not train_dataset_path:
            # 如果没有分割信息，回退到 DatasetManager 获取
            from dataset_manager import create_dataset_manager
            dm = create_dataset_manager(job_id)
            fallback_test = dm.get_test_dataset_path()
            if fallback_test:
                eval_state["dataset_path"] = fallback_test
                logger.info(f"[迭代 {iteration}] 回退到 DatasetManager 获取测试数据: {fallback_test}")

        # 带重试的评估执行
        max_retries = state.get("max_node_retries", DEFAULT_MAX_NODE_RETRIES)
        eval_result = None
        for attempt in range(1, max_retries + 1):
            # Check stop signal before each attempt
            sync_redis = get_sync_redis_client()
            if sync_redis and sync_redis.exists(f"imts_stop:{job_id}"):
                logger.info(f"[Evaluation] Stop signal detected before attempt {attempt} for job {job_id}")
                _stop_vllm_server()
                return {"fatal_error": "Job stopped by user", "error": "Job stopped by user", "passed": False}

            try:
                eval_result = await run_two_model_evaluation(
                    state=eval_state,
                    progress_callback=progress_callback
                )
                # 检查评估是否报告了错误
                if eval_result.get("error"):
                    error_str = eval_result["error"]
                    error_class = _classify_node_error(error_str)
                    if error_class == "NON_RETRYABLE":
                        logger.error(f"[迭代 {iteration}] 评估不可重试错误: {error_str[:200]}")
                        await builder.stage_end(Stage.EVALUATION.value, eval_result)
                        await save_report(job_id, iteration, "EVALUATION", eval_result)
                        return _make_fatal("evaluation", error_str, max_retries)
                    if attempt < max_retries:
                        logger.warning(f"[迭代 {iteration}] 第{attempt}/{max_retries}次评估失败: {error_str[:200]}, 重试中...")
                        await asyncio.sleep(2 * attempt)
                        continue
                    else:
                        logger.error(f"[迭代 {iteration}] 评估重试耗尽({max_retries}次): {error_str[:200]}")
                        await builder.stage_end(Stage.EVALUATION.value, eval_result)
                        await save_report(job_id, iteration, "EVALUATION", eval_result)
                        return _make_fatal("evaluation", f"重试耗尽({max_retries}次): {error_str}", max_retries)
                break  # 评估成功，跳出重试循环
            except asyncio.CancelledError:
                logger.info(f"[Evaluation] Evaluation cancelled for job {job_id} (stop signal)")
                return {"fatal_error": "Job stopped by user", "error": "Job stopped by user", "passed": False}
            except Exception as e:
                error_str = str(e)
                error_class = _classify_node_error(error_str)
                if error_class == "NON_RETRYABLE":
                    logger.error(f"[迭代 {iteration}] 评估不可重试异常: {error_str[:200]}")
                    await save_error_context(job_id, iteration, "evaluation", state, e)
                    return _make_fatal("evaluation", error_str, max_retries)
                if attempt < max_retries:
                    logger.warning(f"[迭代 {iteration}] 第{attempt}/{max_retries}次评估异常: {error_str[:200]}, 重试中...")
                    await asyncio.sleep(2 * attempt)
                    continue
                else:
                    logger.error(f"[迭代 {iteration}] 评估重试耗尽({max_retries}次): {error_str[:200]}")
                    await save_error_context(job_id, iteration, "evaluation", state, e)
                    return _make_fatal("evaluation", f"重试耗尽({max_retries}次): {error_str}", max_retries)

        passed = eval_result.get("passed", False)
        score = eval_result.get("overall_score", 0)

        # === 子步骤断点恢复：合并评估结果 ===
        all_evaluated_ids = list(evaluated_sample_ids)  # 已评估的
        all_evaluated_ids.extend(samples_evaluated_this_run)  # 本次新评估的

        if resume_from_substep and original_eval_result:
            # 有子步骤恢复，需要合并结果
            logger.info(f"[Evaluation Node] 合并评估结果: 之前 {len(evaluated_sample_ids)} 个 + 本次 {len(samples_evaluated_this_run)} 个")

            # 合并 sample_results
            original_samples = original_eval_result.get("sample_results", [])
            new_samples = eval_result.get("sample_results", [])

            # 合并去重（按 sample_id）
            merged_samples = {s.get("sample_id"): s for s in original_samples}
            for s in new_samples:
                merged_samples[s.get("sample_id")] = s
            merged_samples = list(merged_samples.values())

            # 重新计算聚合指标（基于所有样本的 overall_score）
            all_scores = []
            for sample in merged_samples:
                if "overall_score" in sample:
                    all_scores.append(sample["overall_score"])
                elif "fact_accuracy" in sample:
                    # 如果没有 overall_score，从各维度计算
                    fact = sample.get("fact_accuracy", 0)
                    logic = sample.get("logic_consistency", 0)
                    comp = sample.get("completeness", 0)
                    rel = sample.get("relevance", 0)
                    overall = fact * 0.35 + logic * 0.25 + comp * 0.20 + rel * 0.20
                    all_scores.append(overall)

            if all_scores:
                import numpy as np
                new_overall_score = float(np.mean(all_scores))
                new_passed = new_overall_score >= 75.0

                eval_result = dict(eval_result)
                eval_result["overall_score"] = round(new_overall_score, 2)
                eval_result["passed"] = new_passed
                eval_result["total_samples"] = len(all_evaluated_ids)
                eval_result["sample_results"] = merged_samples[:10]  # 限制返回数量

                passed = new_passed
                score = new_overall_score

                logger.info(f"[Evaluation Node] 合并后 score={score:.2f}, total_samples={len(all_evaluated_ids)}")

        # 保存子步骤 checkpoint
        await save_substep_checkpoint(job_id, iteration, "evaluation", "evaluated_sample_ids", all_evaluated_ids)
        # === 子步骤合并结束 ===

        await builder.stage_end(Stage.EVALUATION.value, eval_result)
        await save_report(job_id, iteration, "EVALUATION", eval_result)

        logger.info(f"[迭代 {iteration}] 评估完成: score={score:.1f}, passed={passed}")

        # P1-1: 保存模型资产到 t_model_asset，实现模型↔数据集追溯
        score_history_for_best = state.get("score_history", [])
        best_score = max(score_history_for_best + [score]) if score_history_for_best else score
        is_best = (score >= best_score)
        await save_model_asset_to_mysql(
            job_id=job_id,
            user_id=state.get("user_id", state.get("userId", 1)),
            iteration=iteration,
            base_model_name=state.get("model_name", "Qwen3-8B"),
            lora_oss_path=f"s3://imts-models/{job_id}/iter_{iteration}/lora",
            evaluation_score=score,
            is_best=is_best,
        )

        # === 保存断点：evaluation 完成 ===
        # P0-1: 计算 score_history 和 stagnation_count
        prev_history = state.get("score_history", [])
        new_history = prev_history + [score]
        stagnation_count = 0
        if len(new_history) >= 2 and not passed:
            if new_history[-1] <= new_history[-2]:
                stagnation_count = state.get("stagnation_count", 0) + 1
            else:
                stagnation_count = 0
        elif passed:
            stagnation_count = 0

        # 创建更新后的 state 用于保存（包含 current_iteration + 1）
        state_after = dict(state)
        state_after["eval_result"] = eval_result
        state_after["passed"] = passed
        state_after["current_iteration"] = iteration + 1
        state_after["score_history"] = new_history
        state_after["stagnation_count"] = stagnation_count
        await save_checkpoint(job_id, state_after, "evaluation")
        # === 断点保存结束 ===

        return {
            "eval_result": eval_result,
            "passed": passed,
            "current_iteration": iteration + 1,  # 为下一轮做准备
            "score_history": new_history,
            "stagnation_count": stagnation_count,
            "error": None
        }

    except Exception as e:
        logger.error(f"[迭代 {iteration}] 评估失败: {e}", exc_info=True)
        await save_error_context(job_id, iteration, "evaluation", state, e)
        error_msg = str(e)
        max_retries = state.get("max_node_retries", DEFAULT_MAX_NODE_RETRIES)
        error_result = {"overall_score": 0, "passed": False, "error": error_msg}
        await builder.stage_end(Stage.EVALUATION.value, error_result)
        await save_report(job_id, iteration, "EVALUATION", error_result)
        return _make_fatal("evaluation", error_msg, max_retries)
    finally:
        # Always kill vLLM after evaluation to free GPU memory for next iteration
        _stop_vllm_server()


# ============================================================================
# LangGraph 工作流定义
# ============================================================================

def should_continue(state: AgentState) -> Literal["CONTINUE", "END", "FATAL_END"]:
    """决定是否继续迭代。

    跳出循环的三个条件：
    1. 达到最大迭代次数 → END
    2. 模型评估达到交付阈值 → END
    3. 任一节点重试耗尽仍失败 → FATAL_END
    """
    # 条件3: 致命错误 → 立即终止
    if state.get("fatal_error"):
        logger.error(f"任务因不可恢复错误终止: {state['fatal_error'][:300]}")
        return "FATAL_END"

    # 条件2: 评估达到阈值 → 正常终止闭环
    if state.get("passed", False):
        score = state.get("eval_result", {}).get("overall_score", 0)
        target = state.get("target_score", 75.0)
        logger.info(f"[迭代 {state.get('current_iteration', 1)}] 当前模型评估分数为 {score:.2f}，达到交付阈值 {target:.1f}，系统正常终止闭环")
        return "END"

    # 条件1: 达到最大迭代次数
    # current_iteration 在 evaluation 完成后递增为 iteration+1，
    # 表示"下一轮要执行的迭代编号"，所以 > max_iterations 才意味着已跑完所有轮次。
    if state.get("current_iteration", 1) > state.get("max_iterations", 3):
        logger.info(f"已完成 {state.get('max_iterations', 3)} 次迭代（current_iteration={state.get('current_iteration', 1)}），结束")
        return "END"

    # 死循环检测 — 连续 N 次评分未提升则强制终止
    stagnation = state.get("stagnation_count", 0)
    if stagnation >= PLATEAU_THRESHOLD:
        logger.warning(
            f"[迭代 {state.get('current_iteration', 1)}] "
            f"连续 {stagnation} 次评分未提升，强制终止"
        )
        return "END"

    return "CONTINUE"


def create_workflow_graph():
    """创建 LangGraph 工作流"""
    workflow = StateGraph(AgentState)

    # 注册三个核心节点
    workflow.add_node("data_optimization", data_optimization_node)
    workflow.add_node("training", training_node)
    workflow.add_node("evaluation", evaluation_node)

    # 设置入口
    workflow.set_entry_point("data_optimization")

    # 主流程
    workflow.add_edge("data_optimization", "training")
    workflow.add_edge("training", "evaluation")

    # 条件边：评估后决定继续、正常结束或致命终止
    workflow.add_conditional_edges(
        "evaluation",
        should_continue,
        {
            "CONTINUE": "data_optimization",
            "END": END,
            "FATAL_END": END,
        }
    )

    return workflow.compile()


# ============================================================================
# 工作流执行
# ============================================================================

def run_workflow(job_data: dict) -> dict:
    """同步运行工作流"""
    from langgraph.checkpoint.memory import MemorySaver

    graph = create_workflow_graph()
    checkpointer = MemorySaver()

    initial_state = AgentState(
        job_id=job_data.get("jobId", ""),
        user_id=job_data.get("userId", 0),
        mode=job_data.get("mode", "AUTO_LOOP"),
        target_prompt=job_data.get("targetPrompt", ""),
        dataset_path=job_data.get("dataset_path") or job_data.get("datasetPath", ""),
        augmented_dataset_path="",
        model_name=job_data.get("modelName", "Qwen3-8B"),
        max_iterations=job_data.get("maxIterations", 3),
        current_iteration=1,
        target_score=job_data.get("target_score", 75.0),
        data_opt_result={},
        train_result={},
        eval_result={},
        weak_areas={},
        augmentation_result={},
        score_history=[],
        stagnation_count=0,
        node_retry_count=0,
        max_node_retries=DEFAULT_MAX_NODE_RETRIES,
        fatal_error=None,
        status="RUNNING",
        passed=False,
        error=None
    )

    logger.info(f"开始执行工作流: job_id={initial_state['job_id']}")

    config = {"configurable": {"thread_id": initial_state["job_id"]}}

    try:
        result = graph.invoke(initial_state, config=config)
        fatal = result.get("fatal_error")
        if fatal:
            logger.error(f"工作流因致命错误终止: {fatal[:300]}")
            return {**result, "status": "FATAL_ERROR"}
        logger.info(f"工作流完成: passed={result.get('passed')}")
        return result
    except Exception as e:
        logger.error(f"工作流执行失败: {e}", exc_info=True)
        return {**initial_state, "status": "FAILED", "error": str(e)}


async def run_workflow_async(job_data: dict) -> dict:
    """异步运行工作流（使用 ainvoke 调用异步节点）"""
    from langgraph.checkpoint.memory import MemorySaver

    graph = create_workflow_graph()
    checkpointer = MemorySaver()

    initial_state = AgentState(
        job_id=job_data.get("jobId", ""),
        user_id=job_data.get("userId", 0),
        mode=job_data.get("mode", "AUTO_LOOP"),
        target_prompt=job_data.get("targetPrompt", ""),
        dataset_path=job_data.get("dataset_path") or job_data.get("datasetPath", ""),
        augmented_dataset_path="",
        model_name=job_data.get("modelName", "Qwen3-8B"),
        max_iterations=job_data.get("maxIterations", 3),
        current_iteration=1,
        target_score=job_data.get("target_score", 75.0),
        llm_api_key=job_data.get("llm_api_key"),
        llm_base_url=job_data.get("llm_base_url"),
        llm_model_name=job_data.get("llm_model_name"),
        data_opt_result={},
        train_result={},
        eval_result={},
        weak_areas={},
        augmentation_result={},
        score_history=[],
        stagnation_count=0,
        node_retry_count=0,
        max_node_retries=DEFAULT_MAX_NODE_RETRIES,
        fatal_error=None,
        status="RUNNING",
        passed=False,
        error=None
    )

    logger.info(f"开始执行工作流: job_id={initial_state['job_id']}")

    config = {"configurable": {"thread_id": initial_state["job_id"]}}

    try:
        result = await graph.ainvoke(initial_state, config=config)
        fatal = result.get("fatal_error")
        if fatal:
            logger.error(f"工作流因致命错误终止: {fatal[:300]}")
            return {**result, "status": "FATAL_ERROR"}
        logger.info(f"工作流完成: passed={result.get('passed')}")
        return result
    except Exception as e:
        logger.error(f"工作流执行失败: {e}", exc_info=True)
        return {**initial_state, "status": "FAILED", "error": str(e)}


__all__ = ["AgentState", "create_workflow_graph", "run_workflow", "run_workflow_async"]
