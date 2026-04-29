---
name: data-validator
description: "使用全面检查验证最终数据集质量。在数据优化流水线末尾使用此技能。"
---

# DataValidator 技能

## 何时使用

在数据优化流水线末尾调用此技能以验证最终数据集质量。此技能执行全面的质量检查。

## 输入参数

| 参数 | 类型 | 必填 | 描述 |
|-----------|------|----------|-------------|
| augmented_data | dict | 是 | DataAugmenter 技能的输出 |
| augmented_data.final_rows | int | 是 | 最终行数 |

## 输出

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| final_rows | int | 最终数据行数 |
| quality_checks | dict | 各质量检查得分 |
| overall_quality_score | float | 综合质量得分 (0-1) |
| validation_passed | bool | 验证是否通过 |
| warnings | list | 警告消息 |

## 质量检查

| 检查项 | 权重 | 描述 |
|-------|--------|-------------|
| null_check | 25% | 检查空值 |
| dup_check | 25% | 检查重复 |
| format_check | 25% | 检查格式合规性 |
| length_check | 25% | 检查文本长度分布 |

## 验证标准

- 通过：overall_quality_score >= 0.85
- 警告：任何单项检查 < 0.80

## 步骤

1. 接收增强数据上下文
2. 执行空值检查
3. 执行重复检查
4. 执行格式检查
5. 执行长度检查
6. 计算总分
7. 判定通过/失败
8. 返回验证报告

## 示例

```python
context = {"augmented_data": {"final_rows": 15000}}
result = await data_validator.execute(context)
```
