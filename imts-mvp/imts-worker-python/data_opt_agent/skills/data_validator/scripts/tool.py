"""
DataValidator Tool - 数据验证工具
Agent Skills 规范实现

全面验证数据集质量：
- 空值检查
- 重复检查
- 格式检查
- 长度分布检查
- 字段完整性检查
"""

import asyncio
import os
import pandas as pd
import json
from typing import Dict, List, Any, Union
from langchain_core.tools import tool

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from embedding_service import load_dataset_from_state
from data_opt_agent.skills.state_utils import parse_state


def validate_sample(sample: dict) -> List[Dict[str, Any]]:
    """验证单个样本，返回所有问题"""
    issues = []

    question = str(sample.get("question", sample.get("instruction", sample.get("input", ""))))
    answer = str(sample.get("answer", sample.get("output", "")))

    # 检查空值
    if not question.strip():
        issues.append({"field": "question", "type": "empty", "severity": "high"})
    if not answer.strip():
        issues.append({"field": "answer", "type": "empty", "severity": "high"})

    # 检查问题长度
    if len(question) > 5000:
        issues.append({"field": "question", "type": "too_long", "severity": "medium", "length": len(question)})
    if len(question) < 3:
        issues.append({"field": "question", "type": "too_short", "severity": "medium", "length": len(question)})

    # 检查答案长度
    if len(answer) > 10000:
        issues.append({"field": "answer", "type": "too_long", "severity": "medium", "length": len(answer)})
    if len(answer) < 3:
        issues.append({"field": "answer", "type": "too_short", "severity": "medium", "length": len(answer)})

    # 检查格式 - 问题应该有问号结尾（或中文问号）
    if question.strip() and not question.strip().endswith(('?', '？', '?')):
        issues.append({"field": "question", "type": "missing_question_mark", "severity": "low"})

    # 检查特殊字符
    if '\ufffd' in question or '\ufffd' in answer:
        issues.append({"field": "both", "type": "garbled_char", "severity": "high"})

    return issues


@tool
async def data_validator(state: Union[dict, str]) -> dict:
    """使用全面检查验证最终数据集质量。

    在数据优化流水线末尾使用此技能。
    对整个数据集进行全面质量检查，包括：
    - 空值检查 (null_check)
    - 重复检查 (dup_check)
    - 格式检查 (format_check)
    - 长度分布检查 (length_check)
    - 字段完整性检查

    Returns:
        dict，包含 quality_checks、overall_quality_score、validation_passed、detailed_issues
    """
    state = parse_state(state)
    df, dataset_path = load_dataset_from_state(state)

    if df is None:
        return {"status": "no_data", "message": "No dataset loaded"}

    total_samples = len(df)

    # Determine column format
    has_alpaca = "instruction" in df.columns or "output" in df.columns
    q_col = "instruction" if has_alpaca and "instruction" in df.columns else "question"
    a_col = "output" if has_alpaca and "output" in df.columns else "answer"

    # 1. 空值检查
    null_questions = 0
    null_answers = 0
    for _, row in df.iterrows():
        q = str(row.get(q_col, row.get("question", row.get("instruction", row.get("input", "")))))
        a = str(row.get(a_col, row.get("answer", row.get("output", ""))))
        if not q.strip():
            null_questions += 1
        if not a.strip():
            null_answers += 1

    null_check = 1.0 - ((null_questions + null_answers) / (total_samples * 2))

    # 2. 重复检查
    seen_pairs = set()
    duplicates = 0
    for _, row in df.iterrows():
        q = str(row.get(q_col, row.get("question", row.get("instruction", row.get("input", ""))))).strip().lower()
        a = str(row.get(a_col, row.get("answer", row.get("output", "")))).strip().lower()
        pair = f"{q}|||{a}"
        if pair in seen_pairs:
            duplicates += 1
        seen_pairs.add(pair)

    dup_check = 1.0 - (duplicates / total_samples) if total_samples > 0 else 1.0

    # 3. 格式检查 (问题有问号、答案有句号)
    format_issues = 0
    for _, row in df.iterrows():
        q = str(row.get(q_col, row.get("question", row.get("instruction", row.get("input", "")))))
        a = str(row.get(a_col, row.get("answer", row.get("output", ""))))
        if q.strip() and not q.strip().endswith(('?', '？')):
            format_issues += 1
        if len(a) > 5000:
            format_issues += 1

    format_check = 1.0 - (format_issues / total_samples) if total_samples > 0 else 1.0

    # 4. 长度分布检查
    length_issues = 0
    for _, row in df.iterrows():
        q = str(row.get(q_col, row.get("question", row.get("instruction", row.get("input", "")))))
        a = str(row.get(a_col, row.get("answer", row.get("output", ""))))
        # 异常短或异常长
        if len(q) < 3 or len(q) > 5000:
            length_issues += 1
        if len(a) < 3 or len(a) > 10000:
            length_issues += 1

    length_check = 1.0 - (length_issues / total_samples) if total_samples > 0 else 1.0

    # 5. 字段完整性
    required_cols = ["question", "answer"]
    # 检查是否有这些字段或别名
    cols = [c.lower() for c in df.columns]
    has_question = any(c in cols for c in ["question", "input", "title"])
    has_answer = any(c in cols for c in ["answer", "output", "content"])
    field_check = 1.0 if (has_question and has_answer) else 0.5

    # 计算综合分数
    quality_checks = {
        "null_check": round(null_check, 4),
        "dup_check": round(dup_check, 4),
        "format_check": round(format_check, 4),
        "length_check": round(length_check, 4),
        "field_check": round(field_check, 4)
    }

    overall_score = sum(quality_checks.values()) / len(quality_checks)

    passed = overall_score >= 0.85

    # 生成详细问题列表 (只取前20个)
    detailed_issues = []
    for idx, row in df.head(100).iterrows():
        issues = validate_sample(row.to_dict())
        if issues:
            detailed_issues.append({
                "sample_id": str(row.get("id", idx)),
                "issues": issues
            })

    warnings = []
    if null_check < 0.95:
        warnings.append(f"存在 {null_questions} 个空问题 和 {null_answers} 个空答案")
    if dup_check < 0.95:
        warnings.append(f"发现 {duplicates} 个重复样本")
    if format_check < 0.95:
        warnings.append(f"存在 {format_issues} 个格式问题")
    if not passed:
        warnings.append("质量分数低于阈值 0.85")

    return {
        "status": "success",
        "final_rows": total_samples,
        "quality_checks": quality_checks,
        "overall_quality_score": round(overall_score, 4),
        "validation_passed": passed,
        "detailed_issues": detailed_issues[:20],
        "warnings": warnings,
        "dataset_path": dataset_path,
        "path": dataset_path,
        "stats": {
            "null_questions": null_questions,
            "null_answers": null_answers,
            "duplicates": duplicates,
            "format_issues": format_issues,
            "length_issues": length_issues
        }
    }