---
name: data-deduplicator
description: "数据去重模块，支持精确去重和语义去重，使用 MD5 哈希和 Embedding 相似度识别重复样本。"
---

# DataDeduplicator Skill

## 1. 概述

数据去重（Deduplication）模块，通过精确匹配和语义相似度两种方式识别并移除重复样本，提高数据集质量和多样性。

## 2. 功能

### 2.1 精确去重 (Exact Deduplication)
- 使用 MD5 哈希对规范化文本进行指纹识别
- 识别完全相同的问题-答案对
- 支持增量去重处理

### 2.2 语义去重 (Semantic Deduplication)
- 基于 Embedding 模型的语义相似度计算
- 可配置相似度阈值（默认 0.95）
- 支持批量处理大规模数据集
- 使用余弦相似度判断语义重复

### 2.3 完整去重 (Full Deduplication)
- 先执行精确去重
- 再执行语义去重
- 生成完整去重报告

## 3. 工具

### exact_deduplicate
```
输入: state (dict with dataset_path)
输出: {
    status: "success" | "no_data" | "error",
    original_count: int,
    duplicates_removed: int,
    unique_samples: int,
    duplicate_ratio: float,
    removed_samples: list,
    output_path: str
}
```

### semantic_deduplicate
```
输入: state (dict with dataset_path), similarity_threshold (float, 默认 0.95)
输出: {
    status: "success" | "no_data" | "error",
    original_count: int,
    semantic_duplicates_removed: int,
    unique_samples: int,
    duplicate_ratio: float,
    similarity_threshold: float,
    removed_samples: list,
    output_path: str
}
```

### full_deduplicate
```
输入: state (dict with dataset_path), semantic_threshold (float, 默认 0.95)
输出: {
    status: "success" | "no_data" | "error",
    original_count: int,
    exact_duplicates_removed: int,
    semantic_duplicates_removed: int,
    total_removed: int,
    final_unique_count: int,
    retention_rate: float,
    output_path: str,
    steps: list
}
```

## 4. 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| similarity_threshold | 0.95 | 语义相似度阈值 |
| embedding_url | http://10.242.33.21:11434/v1 | Embedding 服务地址 |
| embedding_model | nomic-embed-text | Embedding 模型名称 |

## 5. 使用示例

```python
from data_opt_agent.skills.data_deduplicator.scripts.tool import (
    exact_deduplicate,
    semantic_deduplicate,
    full_deduplicate
)

# 精确去重
result = await exact_deduplicate({"dataset_path": "/path/to/data.csv"})

# 语义去重 (阈值 0.90)
result = await semantic_deduplicate(
    {"dataset_path": "/path/to/data.csv"},
    similarity_threshold=0.90
)

# 完整去重
result = await full_deduplicate(
    {"dataset_path": "/path/to/data.csv"},
    semantic_threshold=0.95
)
```

## 6. 输出文件

去重后的数据保存为:
- 精确去重: `{原文件名}_deduplicated.{csv|json}`
- 语义去重: `{原文件名}_semantic_dedup.{csv|json}`
- 完整去重: `{原文件名}_fully_dedup.{csv|json}`