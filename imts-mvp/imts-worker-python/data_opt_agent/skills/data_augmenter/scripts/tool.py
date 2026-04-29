"""
DataAugmenter Tool - Data Augmentation
Agent Skills 规范实现

多样性改写：对问题/答案进行同义词替换、句式重写，保持语义不变
反向翻译/推导：基于已有问答对生成新的问答对
合成数据生成：针对边缘case生成合成的训练数据
"""

import asyncio
import os
import json
import pandas as pd
from typing import Dict, List, Any, Tuple, Union
from langchain_core.tools import tool

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from embedding_service import DataSample, DataAugmenter, state_to_samples, load_dataset_from_state
from data_opt_agent.skills.state_utils import parse_state, get_output_dir


def get_augmenter_config() -> Tuple[str, str, str]:
    """Get configuration for DataAugmenter from environment."""
    # Use eval API for augmentation (qwen-max or similar)
    api_key = os.getenv('EVAL_API_KEY', 'dummy')
    base_url = os.getenv('EVAL_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    model_name = os.getenv('EVAL_MODEL_NAME', 'qwen-max')
    return api_key, base_url, model_name


@tool
async def diversity_rewrite(state: Union[dict, str], strategy: str = "paraphrase") -> dict:
    """Augment question/answer pairs using diverse rewriting strategies.

    Strategies:
    - paraphrase: Paraphrase while keeping meaning
    - expand: Expand abbreviations and formal language
    - condense: Condense verbose content
    - formalize: Convert casual to formal language

    Args:
        state: State dict with dataset_path
        strategy: Rewriting strategy (paraphrase/expand/condense/formalize)

    Returns:
        dict with original_count, augmented_count, augmented_samples
    """
    state = parse_state(state)
    df, dataset_path = load_dataset_from_state(state)
    if df is None:
        return {"status": "no_data"}

    api_key, base_url, model_name = get_augmenter_config()
    augmenter = DataAugmenter(api_key, base_url, model_name)

    augmented_samples = []
    strategy_prompts = {
        "paraphrase": "paraphrase the following Q&A with different wording while keeping the exact meaning",
        "expand": "expand abbreviations and make the content more detailed",
        "condense": "condense the content to be more concise while keeping key information",
        "formalize": "convert to a more formal tone"
    }

    for idx, row in df.iterrows():
        original_id = str(row.get("id", idx))
        question = str(row.get("question", row.get("input", "")))
        answer = str(row.get("answer", row.get("output", "")))

        # Call augmentation for each sample
        rewritten_q, rewritten_a = await augmenter.diversity_rewrite(question, answer)

        augmented_samples.append(DataSample(
            id=f"{original_id}_aug",
            question=rewritten_q,
            answer=rewritten_a,
            metadata={"original_id": original_id, "strategy": strategy}
        ))

    # Save augmented data
    output_path = dataset_path.replace('.csv', f'_augmented_{strategy}.csv').replace('.json', f'_augmented_{strategy}.json')

    augmented_data = [{"id": s.id, "question": s.question, "answer": s.answer, "original_id": s.metadata.get("original_id", "")} for s in augmented_samples]
    output_df = pd.DataFrame(augmented_data)

    if output_path.endswith('.csv'):
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_json(output_path, orient='records', force_ascii=False)

    return {
        "status": "success",
        "original_count": len(df),
        "augmented_count": len(augmented_samples),
        "strategy": strategy,
        "output_path": output_path,
        "augmented_samples": augmented_samples[:10]
    }


@tool
async def reverse_translate(state: Union[dict, str]) -> dict:
    """Generate new Q&A pairs through reverse translation/deduction.

    Generates more complex variations by:
    - Adding conditions or constraints
    - Changing scope (general/specific)
    - Asking for explanations of "why"
    - Requesting comparisons

    Args:
        state: State dict with dataset_path

    Returns:
        dict with original_count, generated_count, new_samples
    """
    state = parse_state(state)
    df, dataset_path = load_dataset_from_state(state)
    if df is None:
        return {"status": "no_data"}

    api_key, base_url, model_name = get_augmenter_config()
    augmenter = DataAugmenter(api_key, base_url, model_name)

    new_samples = []
    aug_id = len(df)

    for idx, row in df.iterrows():
        original_id = str(row.get("id", idx))
        question = str(row.get("question", row.get("input", "")))
        answer = str(row.get("answer", row.get("output", "")))

        # Get variations
        variations = await augmenter.reverse_translation(question, answer)

        for var in variations:
            aug_id += 1
            new_samples.append(DataSample(
                id=f"{original_id}_var_{aug_id}",
                question=var.get("question", question),
                answer=var.get("answer", answer),
                metadata={"original_id": original_id, "variation_type": var.get("type", "derived")}
            ))

    # Save generated data
    output_path = dataset_path.replace('.csv', '_reverse_translated.csv').replace('.json', '_reverse_translated.json')

    new_data = [{"id": s.id, "question": s.question, "answer": s.answer} for s in new_samples]
    output_df = pd.DataFrame(new_data)

    if output_path.endswith('.csv'):
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_json(output_path, orient='records', force_ascii=False)

    return {
        "status": "success",
        "original_count": len(df),
        "generated_count": len(new_samples),
        "output_path": output_path,
        "new_samples": new_samples[:10]
    }


@tool
async def generate_edge_cases(state: Union[dict, str], num_per_category: int = 5) -> dict:
    """Generate synthetic edge case data for improved model robustness.

    Generates Q&A pairs for:
    - Edge cases (borderline conditions)
    - Atypical examples
    - Extreme scenarios
    - Exception handling

    Args:
        state: State dict with dataset_path (optional base data)
        num_per_category: Number of samples per edge case category

    Returns:
        dict with generated_count, edge_case_samples
    """
    state = parse_state(state)
    base_path = state.get("dataset_path", "")

    # Get domain info from base data if available
    domain = "general knowledge"
    if base_path and os.path.exists(base_path):
        try:
            if base_path.endswith('.csv'):
                df = pd.read_csv(base_path)
            elif base_path.endswith('.json'):
                df = pd.read_json(base_path)

            # Extract domain from first few samples
            sample_texts = []
            for idx, row in df.head(5).iterrows():
                q = str(row.get("question", row.get("input", "")))
                a = str(row.get("answer", row.get("output", "")))
                sample_texts.append(f"{q} {a}")

            # Use a simple heuristic for domain
            combined = " ".join(sample_texts).lower()
            if any(kw in combined for kw in ["操作系统", "OS", "linux", "windows", "系统"]):
                domain = "operating systems"
            elif any(kw in combined for kw in ["数学", "math", "计算", "算法"]):
                domain = "mathematics"
            elif any(kw in combined for kw in ["物理", "physics", "化学", "science"]):
                domain = "science"
        except:
            pass

    api_key, base_url, model_name = get_augmenter_config()
    augmenter = DataAugmenter(api_key, base_url, model_name)

    edge_cases = []
    aug_id = 0

    # Generate edge cases for different categories
    categories = ["borderline", "extreme", "exception", "boundary"]
    for category in categories:
        result = await augmenter.generate_synthetic_for_edge_cases(
            domain=domain,
            concept=category,
            num_samples=num_per_category
        )

        for ec in result:
            aug_id += 1
            edge_cases.append(DataSample(
                id=f"ec_{aug_id}",
                question=ec.get("question", ""),
                answer=ec.get("answer", ""),
                metadata={
                    "category": ec.get("edge_case_type", category),
                    "is_synthetic": True
                }
            ))

    # Save generated data
    output_path = base_path.replace('.csv', '_edge_cases.csv').replace('.json', '_edge_cases.json') if base_path else os.path.join(get_output_dir(), 'edge_cases_generated.csv')
    if not output_path or output_path.endswith('.csv') or output_path.endswith('.json'):
        output_path = os.path.join(get_output_dir(), 'edge_cases_generated.csv')

    edge_data = [{"id": s.id, "question": s.question, "answer": s.answer, "category": s.metadata.get("category", "edge_case")} for s in edge_cases]
    output_df = pd.DataFrame(edge_data)

    if output_path.endswith('.csv'):
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_json(output_path, orient='records', force_ascii=False)

    return {
        "status": "success",
        "generated_count": len(edge_cases),
        "categories": categories,
        "output_path": output_path,
        "edge_case_samples": edge_cases[:10]
    }


@tool
async def generate_cot(state: Union[dict, str]) -> dict:
    """Generate Chain-of-Thought reasoning examples.

    Creates Q&A pairs with step-by-step reasoning for complex problems.

    Args:
        state: State dict with dataset_path

    Returns:
        dict with original_count, generated_count, cot_samples
    """
    state = parse_state(state)
    df, dataset_path = load_dataset_from_state(state)
    if df is None:
        return {"status": "no_data"}

    api_key, base_url, model_name = get_augmenter_config()
    augmenter = DataAugmenter(api_key, base_url, model_name)

    cot_samples = []

    for idx, row in df.iterrows():
        original_id = str(row.get("id", idx))
        question = str(row.get("question", row.get("input", "")))
        answer = str(row.get("answer", row.get("output", "")))

        # Generate CoT reasoning
        cot_answer = await augmenter.generate_cot(question, answer)

        cot_samples.append(DataSample(
            id=original_id,
            question=question,
            answer=cot_answer,
            metadata={"original_answer": answer, "has_cot": True}
        ))

    # Save generated data
    output_path = dataset_path.replace('.csv', '_cot.csv').replace('.json', '_cot.json')

    cot_data = [{"id": s.id, "question": s.question, "answer": s.answer} for s in cot_samples]
    output_df = pd.DataFrame(cot_data)

    if output_path.endswith('.csv'):
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_json(output_path, orient='records', force_ascii=False)

    return {
        "status": "success",
        "original_count": len(df),
        "generated_count": len(cot_samples),
        "output_path": output_path,
        "cot_samples": cot_samples[:10]
    }
