---
name: data-analyzer
description: "数据分析模块，自动识别数据集的语义分布、长度分布、语言比例，提取元数据（主题、难度、情感倾向等），检测异常样本。"
---

# DataAnalyzer Skill

## 1. 概述

数据分析（Distribution Analysis）模块，自动识别数据集的语义分布、长度分布、语言比例，提取元数据（主题、难度、情感倾向等），检测异常样本。

## 2. 功能

### 2.1 分布分析 (Distribution Analysis)
- 长度分布（问题、答案、总长度）
- 语义分布（使用 Embedding 计算）
- 主题/关键词分布
- 生成分析建议

### 2.2 元数据提取 (Metadata Extraction)
- 难度级别（easy/medium/hard）
- 情感倾向（positive/neutral/negative）
- 领域分类
- 每样本标签

### 2.3 异常检测 (Anomaly Detection)
- 格式错误检测（HTML、空字段、乱码）
- 长度异常值检测（使用 IQR 方法）
- 按严重性分类

## 3. 工具

### analyze_distribution
```
输入: state (dict with dataset_path)
输出: {
    status: "success" | "no_data",
    total_samples: int,
    length_stats: {mean, std, min, max, p50, p95},
    semantic_stats: {diversity_score, note},
    topics: [{topic, count}, ...],
    suggestions: [string, ...]
}
```

### extract_metadata
```
输入: state (dict with dataset_path)
输出: {
    status: "success" | "no_data",
    total_samples: int,
    difficulty_distribution: {easy, medium, hard},
    sentiment_distribution: {positive, neutral, negative},
    domain_distribution: {domain: count, ...},
    sample_metadata: [...],
    suggestions: [string, ...]
}
```

### detect_anomalies
```
输入: state (dict with dataset_path)
输出: {
    status: "success" | "no_data",
    total_samples: int,
    total_errors: int,
    format_errors: {count, high_severity, medium_severity, samples},
    length_outliers: {count, samples},
    suggestions: [string, ...]
}
```

## 4. 输出字段说明

### length_stats
| 字段 | 说明 |
|------|------|
| mean | 平均长度 |
| std | 标准差 |
| min | 最小长度 |
| max | 最大长度 |
| p50 | 中位数长度 |
| p95 | 95百分位长度 |

### difficulty_distribution
基于问题复杂度、答案长度、领域专业度综合评估：
- **easy**: 简单事实性问题
- **medium**: 需要解释或推理的问题
- **hard**: 复杂推理或多步骤问题

### format_errors 严重性等级
- **high**: 空字段、严重乱码、HTML 错误
- **medium**: 格式不规范、长度异常

## 5. 使用示例

```python
from data_opt_agent.skills.data_analyzer.scripts.tool import (
    analyze_distribution,
    extract_metadata,
    detect_anomalies
)

# 分析分布
result = await analyze_distribution({"dataset_path": "/path/to/data.csv"})

# 提取元数据
result = await extract_metadata({"dataset_path": "/path/to/data.csv"})

# 检测异常
result = await detect_anomalies({"dataset_path": "/path/to/data.csv"})
```
