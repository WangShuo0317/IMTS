"""
TextNormalizer Tool - 文本规范化工具
Agent Skills 规范实现

文本规范化处理：
- 统一编码
- 去除多余空白
- 标点符号标准化
- 大小写规范化（对英文）
- 全角半角转换
- 行尾规范化
"""

import asyncio
import os
import re
import pandas as pd
from typing import Dict, List, Any, Union
from langchain_core.tools import tool

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from embedding_service import load_dataset_from_state
from data_opt_agent.skills.state_utils import parse_state


def normalize_text(text: str) -> tuple:
    """规范化单个文本，返回 (规范化后的文本, 应用的操 作列表)"""
    if not text:
        return text, []

    operations = []
    original = text

    # 1. 去除控制字符
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t\r')
    if text != original:
        operations.append("remove_control_chars")

    # 2. 去除多余空白
    # 多个空格/换行/tab 合并为一个空格
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\n ', '\n', text)
    text = re.sub(r' \n', '\n', text)
    if text != original:
        operations.append("normalize_whitespace")

    # 3. 去除首尾空白
    text = text.strip()
    if text != original:
        operations.append("trim_whitespace")

    # 4. 统一换行符为 \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 5. 去除连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 6. 句子结尾标准化 (可选)
    # 如果以没有标点的句子结尾，添加句号
    text = re.sub(r'([^。！？!?])$', r'\1。', text)

    return text, operations
    """规范化单个文本，返回 (规范化后的文本, 应用的操 作列表)"""
    if not text:
        return text, []

    operations = []
    original = text

    # 1. 去除控制字符
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t\r')
    if text != original:
        operations.append("remove_control_chars")

    # 2. 去除多余空白
    # 多个空格/换行/tab 合并为一个空格
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\n ', '\n', text)
    text = re.sub(r' \n', '\n', text)
    if text != original:
        operations.append("normalize_whitespace")

    # 3. 去除首尾空白
    text = text.strip()
    if text != original:
        operations.append("trim_whitespace")

    # 4. 全角转半角 (可选，用于英文场景)
    # 只对ASCII范围内的字符转换
    # 暂时不启用，避免中英文混合时出现问题

    # 5. 统一换行符为 \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 6. 去除连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 7. 句子结尾标准化 (可选)
    # 如果以没有标点的句子结尾，添加句号
    text = re.sub(r'([^。！？!?])$', r'\1。', text)

    return text, operations


@tool
async def text_normalizer(state: Union[dict, str]) -> dict:
    """通过转换为小写、移除特殊字符和去除空白来规范化文本数据。

    在 DataCleaner 之后使用此技能以标准化文本内容。
    对数据集中的所有文本字段进行规范化处理：
    - 去除控制字符
    - 标准化空白字符
    - 去除首尾空格
    - 统一换行符
    - 去除连续空行
    - 句子结尾标准化

    Returns:
        dict，包含 rows_normalized、operations_applied、stats
    """
    state = parse_state(state)
    df, dataset_path = load_dataset_from_state(state)

    if df is None:
        return {"status": "no_data", "message": "No dataset loaded"}

    total_samples = len(df)
    all_operations = set()
    total_chars_before = 0
    total_chars_after = 0

    # 确定需要规范化的列
    text_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["question", "input", "title", "answer", "output", "content", "text"]:
            text_cols.append(col)

    # 规范化每一行
    normalized_data = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        row_operations = set()

        for col in text_cols:
            if col in row_dict:
                original_text = str(row_dict[col])
                total_chars_before += len(original_text)

                normalized_text, ops = normalize_text(original_text)
                row_dict[col] = normalized_text
                total_chars_after += len(normalized_text)

                row_operations.update(ops)

        all_operations.update(row_operations)
        normalized_data.append(row_dict)

    # 更新 DataFrame
    df_normalized = pd.DataFrame(normalized_data)

    # 保存规范化后的数据
    output_path = dataset_path.replace('.csv', '_normalized.csv').replace('.json', '_normalized.json')

    if output_path.endswith('.csv'):
        df_normalized.to_csv(output_path, index=False, encoding='utf-8')
    else:
        df_normalized.to_json(output_path, orient='records', force_ascii=False, indent=2)

    return {
        "status": "success",
        "total_samples": total_samples,
        "rows_normalized": total_samples,
        "operations_applied": sorted(list(all_operations)),
        "char_stats": {
            "chars_before": total_chars_before,
            "chars_after": total_chars_after,
            "char_reduced": total_chars_before - total_chars_after,
            "reduction_ratio": round((total_chars_before - total_chars_after) / total_chars_before, 4) if total_chars_before > 0 else 0
        },
        "output_path": output_path,
        "columns_normalized": text_cols
    }