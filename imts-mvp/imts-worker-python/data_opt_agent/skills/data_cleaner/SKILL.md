---
name: data-cleaner
description: "数据清洗模块，自动剔除 HTML 标签、乱码，修复断句错误，敏感信息脱敏。"
---

# DataCleaner Skill

## 1. 概述

数据清洗（Data Cleaning）模块，包含去噪与修复（HTML标签、乱码、断句错误）和敏感信息脱敏（PII Masking）。

## 2. 功能

### 2.1 去噪与修复 (Denoising & Fixing)
- HTML 标签移除
- 乱码字符清除
- 空格规范化
- 断句错误修复
- 格式验证与修复

### 2.2 敏感信息脱敏 (PII Masking)
- 手机号码（中国大陆手机号）
- 电子邮箱
- 身份证号码
- 银行卡号码

## 3. 工具

### clean_text
```
输入: text (str)
输出: {
    original_length: int,
    cleaned_length: int,
    operations_applied: [string, ...],
    cleaned_text: string
}
```

### mask_pii
```
输入: text (str)
输出: {
    pii_found: [string, ...],
    pii_count: int,
    masked_text: string
}
```

### batch_clean
```
输入: state (dict with dataset_path)
输出: {
    status: "success" | "no_data" | "error",
    total_samples: int,
    cleaned_samples: [...],
    html_removed_count: int,
    garbled_fixed_count: int,
    pii_masked_count: int,
    format_fixed_count: int,
    output_path: string
}
```

### validate_and_fix
```
输入: state (dict with dataset_path)
输出: {
    status: "success" | "no_data" | "error",
    total_samples: int,
    issues_found: int,
    fixes_applied: int,
    high_severity_issues: int,
    issues: [...],
    output_path: string
}
```

## 4. PII 类型说明

| 类型 | 说明 | 示例 |
|------|------|------|
| phone | 中国大陆手机号 | 13812345678 |
| email | 电子邮箱 | user@example.com |
| id_card | 身份证号码 | 110101199001011234 |
| bank_card | 银行卡号 | 6222021234567890123 |

## 5. 验证规则

- **必需字段**: question 和 answer 不能为空
- **问题格式**: 必须以 ? 或 ？ 结尾
- **答案格式**: 必须以句号结尾（。！？.!?）
- **长度限制**: 问题 < 5000 字符，答案 < 10000 字符

## 6. 使用示例

```python
from data_opt_agent.skills.data_cleaner.scripts.tool import (
    clean_text,
    mask_pii,
    batch_clean,
    validate_and_fix
)

# 清洗单个文本
result = await clean_text("Some text with <html> tags")

# 脱敏 PII
result = await mask_pii("Contact me at 13812345678 or user@email.com")

# 批量清洗数据集
result = await batch_clean({"dataset_path": "/path/to/data.csv"})

# 验证并修复格式
result = await validate_and_fix({"dataset_path": "/path/to/data.csv"})
```
