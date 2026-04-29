"""
DataCleaner Tool - Enhanced Data Cleaning
Agent Skills 规范实现

去噪与修复：自动剔除 HTML 标签、乱码，修复断句错误
敏感信息脱敏 (PII Masking)：自动识别并屏蔽姓名、电话、地址等隐私数据
"""

import asyncio
import os
import json
import pandas as pd
from typing import Dict, List, Any, Tuple, Union
from langchain_core.tools import tool

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from embedding_service import Denoiser, DataSample, load_dataset_from_state
from data_opt_agent.skills.state_utils import parse_state


@tool
async def clean_text(text: str) -> dict:
    """Clean a single text sample.

    Performs:
    - HTML tag removal
    - Garbled character removal
    - Spacing normalization
    - Sentence ending fixes

    Args:
        text: The text to clean

    Returns:
        dict with cleaned_text and operations_applied
    """
    denoiser = Denoiser()
    original = text

    # Apply cleaning pipeline
    cleaned = denoiser.denoise(text)

    operations = []
    if '<html' in original.lower() or '<div' in original.lower():
        operations.append("removed_html")
    if '�' in original or '\x00' in original:
        operations.append("removed_garbled")
    if '  ' in original:
        operations.append("fixed_spacing")
    if not original.rstrip().endswith(('。', '！', '？', '.', '!', '?')):
        operations.append("fixed_ending")

    return {
        "original_length": len(original),
        "cleaned_length": len(cleaned),
        "operations_applied": operations,
        "cleaned_text": cleaned
    }


@tool
async def mask_pii(text: str) -> dict:
    """Mask personally identifiable information (PII) in text.

    Detects and masks:
    - Phone numbers (Chinese mobile)
    - Email addresses
    - ID numbers
    - Bank card numbers

    Args:
        text: The text containing potential PII

    Returns:
        dict with masked_text and pii_found (list of types)
    """
    denoiser = Denoiser()
    masked_text, masks = denoiser.mask_pii(text)

    pii_types = list(set(m["type"] for m in masks))

    return {
        "pii_found": pii_types,
        "pii_count": len(masks),
        "masked_text": masked_text
    }


@tool
async def batch_clean(state: Union[dict, str]) -> dict:
    """Batch clean all samples in the dataset.

    Applies full cleaning pipeline to all samples:
    - HTML removal
    - Garbled text removal
    - Spacing normalization
    - PII masking
    - Format fixes

    Preserves column format: Alpaca {instruction, input, output} or QA {question, answer}.

    Returns:
        dict with cleaning_summary and cleaned_samples
    """
    state = parse_state(state)
    df, dataset_path = load_dataset_from_state(state)

    if df is None:
        return {
            "status": "no_data",
            "message": "No dataset loaded"
        }

    denoiser = Denoiser()

    # Determine column format: preserve Alpaca {instruction, input, output} or QA {question, answer}
    has_alpaca = "instruction" in df.columns or "output" in df.columns
    q_col = "instruction" if has_alpaca and "instruction" in df.columns else "question"
    a_col = "output" if has_alpaca and "output" in df.columns else "answer"

    results = {
        "total_samples": len(df),
        "cleaned_samples": [],
        "html_removed_count": 0,
        "garbled_fixed_count": 0,
        "pii_masked_count": 0,
        "format_fixed_count": 0
    }

    for idx, row in df.iterrows():
        original_q = str(row.get(q_col, row.get("question", row.get("instruction", row.get("input", "")))))
        original_a = str(row.get(a_col, row.get("answer", row.get("output", ""))))

        # Clean
        cleaned_q = denoiser.denoise(original_q)
        cleaned_a = denoiser.denoise(original_a)

        # Mask PII
        masked_q, masks_q = denoiser.mask_pii(cleaned_q)
        masked_a, masks_a = denoiser.mask_pii(cleaned_a)

        # Track stats
        if '<html' in original_q.lower() or '<div' in original_q.lower():
            results["html_removed_count"] += 1
        if masked_q != cleaned_q or masked_a != cleaned_a:
            results["pii_masked_count"] += 1
        if original_q != masked_q or original_a != masked_a:
            results["garbled_fixed_count"] += 1

        sample = {
            "id": str(row.get("id", idx)),
            "pii_found": list(set([m["type"] for m in masks_q + masks_a]))
        }
        sample[q_col] = masked_q
        sample[a_col] = masked_a
        if "input" in df.columns:
            sample["input"] = str(row.get("input", ""))
        results["cleaned_samples"].append(sample)

    # Save cleaned dataset
    output_path = dataset_path.replace('.csv', '_cleaned.csv').replace('.json', '_cleaned.json')
    output_df = pd.DataFrame(results["cleaned_samples"])

    if output_path.endswith('.csv'):
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_json(output_path, orient='records', force_ascii=False)

    results["output_path"] = output_path
    results["dataset_path"] = output_path
    results["path"] = output_path
    results["status"] = "success"

    return results


@tool
async def validate_and_fix(state: Union[dict, str]) -> dict:
    """Validate and fix format issues in dataset.

    Checks:
    - Required fields present
    - Question ends with ?
    - Answer has content
    - No excessive length

    Returns:
        dict with validation_results and fixes_applied
    """
    state = parse_state(state)
    df, dataset_path = load_dataset_from_state(state)

    if df is None:
        return {"status": "no_data"}

    issues = []
    fixes = []

    # Determine column format
    has_alpaca = "instruction" in df.columns or "output" in df.columns
    q_col = "instruction" if has_alpaca and "instruction" in df.columns else "question"
    a_col = "output" if has_alpaca and "output" in df.columns else "answer"

    for idx, row in df.iterrows():
        sample_id = str(row.get("id", idx))
        q = str(row.get(q_col, row.get("question", row.get("input", ""))))
        a = str(row.get(a_col, row.get("answer", row.get("output", ""))))

        sample_issues = []

        # Check required fields
        if not q.strip():
            sample_issues.append("empty_question")
            issues.append({"sample_id": sample_id, "issue": "empty_question", "severity": "high"})
        if not a.strip():
            sample_issues.append("empty_answer")
            issues.append({"sample_id": sample_id, "issue": "empty_answer", "severity": "high"})

        # Check question/instruction format
        if q and not q.strip().endswith(('?', '？')):
            fixes.append({"sample_id": sample_id, "fix": "added_question_mark"})
            df.at[idx, q_col] = q.rstrip() + '?'

        # Check answer/output ending
        if a and not a.strip().endswith(('。', '！', '？', '.', '!', '?')):
            fixes.append({"sample_id": sample_id, "fix": "fixed_answer_ending"})
            df.at[idx, a_col] = a.rstrip() + '。'

        # Check length
        if len(q) > 5000:
            issues.append({"sample_id": sample_id, "issue": "question_too_long", "severity": "medium"})
        if len(a) > 10000:
            issues.append({"sample_id": sample_id, "issue": "answer_too_long", "severity": "medium"})

    # Save fixed dataset
    output_path = dataset_path.replace('.csv', '_validated.csv').replace('.json', '_validated.json')

    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    else:
        df.to_json(output_path, orient='records', force_ascii=False)

    return {
        "status": "success",
        "total_samples": len(df),
        "issues_found": len(issues),
        "fixes_applied": len(fixes),
        "high_severity_issues": len([i for i in issues if i.get("severity") == "high"]),
        "issues": issues[:50],
        "output_path": output_path,
        "dataset_path": output_path,
        "path": output_path
    }