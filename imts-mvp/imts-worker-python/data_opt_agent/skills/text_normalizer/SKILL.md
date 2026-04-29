---
name: text-normalizer
description: "通过转换小写、移除特殊字符和去除空白来规范化文本数据。在 DataCleaner 之后使用。"
---

# TextNormalizer 技能

## 何时使用

在 DataCleaner 之后调用此技能以标准化文本内容。此技能应用各种文本规范化操作以确保一致的文本格式。

## 输入参数

| 参数 | 类型 | 必填 | 描述 |
|-----------|------|----------|-------------|
| cleaned_data | dict | 是 | DataCleaner 技能的输出 |
| cleaned_data.remaining_rows | int | 是 | 清洗后的行数 |

## 输出

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| rows_normalized | int | 规范化后的行数 |
| operations | list | 应用的操作列表 |
| char_count_before | int | 规范化前的字符数 |
| char_count_after | int | 规范化后的字符数 |

## 应用的操作

1. 转换为小写
2. 移除多余空白
3. 移除特殊字符
4. 去除首尾空格

## 步骤

1. 接收清洗后的数据上下文
2. 应用小写转换
3. 移除多余空白
4. 移除特殊字符
5. 去除空格
6. 返回规范化报告

## 示例

```python
context = {"cleaned_data": {"remaining_rows": 9000}}
result = await text_normalizer.execute(context)
```
