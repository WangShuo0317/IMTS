---
name: data-loader
description: "加载并解析各种格式（CSV、JSON、JSONL）的数据集。当需要将数据集文件加载到内存中进行处理时使用此技能。"
---

# DataLoader 技能

## 何时使用

当需要从文件路径加载数据集时调用此技能。DataLoader 技能支持多种文件格式并自动处理编码检测。

## 输入参数

| 参数 | 类型 | 必填 | 描述 |
|-----------|------|----------|-------------|
| path | string | 是 | 数据集文件路径 |
| format | string | 否 | 数据格式（csv/json/jsonl）。如未指定则自动检测 |

## 输出

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| path | string | 数据集路径 |
| rows | int | 数据行数 |
| columns | list | 列名 |
| file_type | string | 文件格式 |
| encoding | string | 字符编码 |
| size_bytes | int | 文件大小（字节） |

## 步骤

1. 从上下文获取文件路径
2. 从扩展名检测文件格式
3. 使用适当的解析器加载文件内容
4. 提取元数据（行数、列、大小）
5. 返回结构化结果

## 示例

```python
context = {"path": "/data/train.csv"}
result = await data_loader.execute(context)
```
