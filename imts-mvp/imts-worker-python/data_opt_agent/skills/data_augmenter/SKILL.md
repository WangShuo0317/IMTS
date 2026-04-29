---
name: data-augmenter
description: "数据增强模块，支持多样性改写、反向翻译/推导、边缘案例生成、思维链（CoT）生成。"
---

# DataAugmenter Skill

## 1. 概述

数据增强（Data Augmentation）模块，通过多样性改写、反向翻译/推导、边缘案例合成等方式生成新的训练样本，提高模型泛化能力。

## 2. 功能

### 2.1 多样性改写 (Diversity Rewrite)
- 同义词替换、句式重写
- 保持语义不变
- 支持多种策略：paraphrase/expand/condense/formalize

### 2.2 反向翻译/推导 (Reverse Translation)
- 基于原问答对生成更复杂的变体
- 添加条件或约束
- 改变范围（更通用或更具体）
- 请求"为什么"的解释
- 请求比较或对比

### 2.3 边缘案例合成 (Edge Case Generation)
- 生成边界条件、极端场景
- 异常处理测试用例
- 增强模型对边缘情况的处理能力

### 2.4 思维链生成 (Chain-of-Thought)
- 为复杂问题生成步骤推理
- 展示中间结论
- 解释每一步的原因

## 3. 工具

### diversity_rewrite
```
输入: state (dict with dataset_path), strategy (str, 默认 paraphrase)
输出: {
    status: "success" | "no_data" | "error",
    original_count: int,
    augmented_count: int,
    strategy: str,
    output_path: str,
    augmented_samples: list
}
```

### reverse_translate
```
输入: state (dict with dataset_path)
输出: {
    status: "success" | "no_data" | "error",
    original_count: int,
    generated_count: int,
    output_path: str,
    new_samples: list
}
```

### generate_edge_cases
```
输入: state (dict with dataset_path), num_per_category (int, 默认 5)
输出: {
    status: "success",
    generated_count: int,
    categories: list,
    output_path: str,
    edge_case_samples: list
}
```

### generate_cot
```
输入: state (dict with dataset_path)
输出: {
    status: "success" | "no_data" | "error",
    original_count: int,
    generated_count: int,
    output_path: str,
    cot_samples: list
}
```

## 4. 配置

使用环境变量配置 API：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| EVAL_API_KEY | - | 评估 API 密钥 |
| EVAL_BASE_URL | https://dashscope.aliyuncs.com/compatible-mode/v1 | API 地址 |
| EVAL_MODEL_NAME | qwen-max | 模型名称 |

## 5. 输出文件

增强后的数据保存为:
- 多样性改写: `{原文件名}_augmented_{strategy}.{csv|json}`
- 反向翻译: `{原文件名}_reverse_translated.{csv|json}`
- 边缘案例: `{原文件名}_edge_cases.{csv|json}`
- 思维链: `{原文件名}_cot.{csv|json}`

## 6. 使用示例

```python
from data_opt_agent.skills.data_augmenter.scripts.tool import (
    diversity_rewrite,
    reverse_translate,
    generate_edge_cases,
    generate_cot
)

# 多样性改写
result = await diversity_rewrite({"dataset_path": "/path/to/data.csv"}, strategy="paraphrase")

# 反向翻译生成变体
result = await reverse_translate({"dataset_path": "/path/to/data.csv"})

# 生成边缘案例
result = await generate_edge_cases({"dataset_path": "/path/to/data.csv"}, num_per_category=5)

# 生成思维链推理
result = await generate_cot({"dataset_path": "/path/to/data.csv"})
```
