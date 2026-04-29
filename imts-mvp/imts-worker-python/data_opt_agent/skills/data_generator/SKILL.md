---
name: data-generator
description: "合成数据生成模块，针对特定领域、概念或边缘案例生成新的训练样本。"
---

# DataGenerator Skill

## 1. 概述

合成数据生成（Synthetic Data Generation）模块，通过 LLM 生成全新的训练数据，用于补充数据集中缺失的领域、概念或边缘案例。

## 2. 功能

### 2.1 领域合成数据生成 (Domain Synthesis)
- 为特定领域生成全新 Q&A 对
- 基于已有样本作为参考上下文
- 生成不同难度级别的样本

### 2.2 概念合成数据生成 (Concept Synthesis)
- 针对特定概念生成测试数据
- 包含定义、示例、应用类问题
- 覆盖概念的各个维度

### 2.3 对抗性合成数据生成 (Adversarial Synthesis)
- 生成挑战性测试用例
- 测试模型鲁棒性
- 包含歧义性、多跳推理、常见误解等

## 3. 工具

### generate_domain_synthetic
```
输入: state (dict), domain (str), num_samples (int, 默认 10)
输出: {
    status: "success" | "error",
    domain: str,
    generated_count: int,
    output_path: str,
    synthetic_samples: list
}
```

### generate_concept_synthetic
```
输入: state (dict), concept (str), num_samples (int, 默认 5)
输出: {
    status: "success" | "error",
    concept: str,
    generated_count: int,
    output_path: str,
    synthetic_samples: list
}
```

### generate_adversarial_synthetic
```
输入: state (dict), base_domain (str), num_samples (int, 默认 5)
输出: {
    status: "success" | "error",
    base_domain: str,
    generated_count: int,
    output_path: str,
    synthetic_samples: list
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

合成数据保存为 CSV 格式:
- 领域合成: `synthetic_{domain}.csv`
- 概念合成: `synthetic_concept_{concept}.csv`
- 对抗合成: `adversarial_{domain}.csv`

## 6. 使用示例

```python
from data_opt_agent.skills.data_generator.scripts.tool import (
    generate_domain_synthetic,
    generate_concept_synthetic,
    generate_adversarial_synthetic
)

# 为操作系统领域生成合成数据
result = await generate_domain_synthetic(
    {"dataset_path": "/path/to/data.csv"},
    domain="operating systems",
    num_samples=20
)

# 为特定概念生成合成数据
result = await generate_concept_synthetic(
    {},
    concept="memory management",
    num_samples=10
)

# 生成对抗性测试数据
result = await generate_adversarial_synthetic(
    {},
    base_domain="computer science",
    num_samples=5
)
```
