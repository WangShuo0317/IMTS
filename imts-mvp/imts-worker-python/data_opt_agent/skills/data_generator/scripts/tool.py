"""
DataGenerator Tool - Synthetic Data Generation
Agent Skills 规范实现

合成数据生成：针对特定领域或概念生成全新的训练数据
用于补充数据集中缺失的领域或边缘案例
"""

import asyncio
import os
import json
import re
import pandas as pd
from typing import Dict, List, Any, Tuple, Union
from langchain_core.tools import tool

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from embedding_service import DataSample, DataAugmenter
from data_opt_agent.skills.state_utils import parse_state, get_output_dir


def get_generator_config() -> Tuple[str, str, str]:
    """Get configuration for data generation from environment."""
    api_key = os.getenv('EVAL_API_KEY', 'dummy')
    base_url = os.getenv('EVAL_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    model_name = os.getenv('EVAL_MODEL_NAME', 'qwen-max')
    return api_key, base_url, model_name


@tool
async def generate_domain_synthetic(state: Union[dict, str], domain: str, num_samples: int = 10) -> dict:
    """Generate synthetic Q&A data for a specific domain.

    Creates entirely new training samples for a given domain,
    useful for filling gaps in the dataset.

    Args:
        state: State dict (optional base dataset for context)
        domain: Domain to generate data for (e.g., "operating systems", "math")
        num_samples: Number of samples to generate

    Returns:
        dict with generated_count, synthetic_samples
    """
    state = parse_state(state)
    dataset_path = state.get("dataset_path", "")

    # Load existing samples for context if available
    existing_samples = []
    if dataset_path and os.path.exists(dataset_path):
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)

            for idx, row in df.head(20).iterrows():
                existing_samples.append({
                    "question": str(row.get("question", row.get("input", ""))),
                    "answer": str(row.get("answer", row.get("output", "")))
                })
        except:
            pass

    api_key, base_url, model_name = get_generator_config()
    augmenter = DataAugmenter(api_key, base_url, model_name)

    prompt = f"""请为 {domain} 领域生成 {num_samples} 个高质量的问答对。

要求：
1. 每个问答对应唯一且覆盖 {domain} 的不同方面
2. 问题应该清晰、具体、可回答
3. 答案应该准确、详细、有帮助
4. 难度要有变化（简单、中等、困难）
5. 包含事实/概念型和实践型问题

"""

    if existing_samples:
        prompt += f"\n该领域的现有样本供参考：\n"
        for i, sample in enumerate(existing_samples[:5]):
            prompt += f"- 问：{sample['question'][:100]}\n  答：{sample['answer'][:100]}\n"

    prompt += """
请以JSON格式回复：
{
    "synthetic_data": [
        {
            "question": "详细的问题",
            "answer": "全面的答案",
            "difficulty": "easy|medium|hard",
            "type": "factual|conceptual|practical"
        }
    ]
}
"""

    response = await augmenter._call_llm(prompt)

    synthetic_samples = []
    try:
        result = json.loads(response)
        data = result.get("synthetic_data", [])
    except json.JSONDecodeError as e:
        # Try to extract valid JSON by finding the synthetic_data array
        # Find JSON-like content
        match = re.search(r'(\{[^{}]*"synthetic_data"\s*:\s*\[)', response, re.DOTALL)
        if match:
            start_pos = match.start()
            try:
                # Try to parse from the start of JSON object
                truncated = response[start_pos:]
                # Try to fix common issues - find the last complete object
                result = json.loads(truncated)
                data = result.get("synthetic_data", [])
            except:
                data = []
        else:
            data = []

        if not data:
            return {"status": "error", "message": f"Failed to parse generated data: {str(e)}. The LLM returned malformed JSON."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse generated data: {str(e)}"}

    for i, item in enumerate(data):
        synthetic_samples.append(DataSample(
            id=f"synth_{domain[:5]}_{i}",
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            metadata={
                "domain": domain,
                "difficulty": item.get("difficulty", "medium"),
                "type": item.get("type", "factual"),
                "is_synthetic": True
            }
        ))

    # Save generated data
    output_dir = get_output_dir()
    output_path = os.path.join(output_dir, f"synthetic_{domain.replace(' ', '_')}.csv")

    synthetic_data = [{
        "id": s.id,
        "question": s.question,
        "answer": s.answer,
        "domain": domain,
        "difficulty": s.metadata.get("difficulty", "medium")
    } for s in synthetic_samples]

    output_df = pd.DataFrame(synthetic_data)
    output_df.to_csv(output_path, index=False)

    return {
        "status": "success",
        "domain": domain,
        "generated_count": len(synthetic_samples),
        "output_path": output_path,
        "synthetic_samples": synthetic_samples[:10]
    }


@tool
async def generate_concept_synthetic(state: Union[dict, str], concept: str, num_samples: int = 5) -> dict:
    """Generate synthetic data for a specific concept.

    Creates Q&A pairs that test understanding of a specific concept,
    including definitions, examples, and edge cases.

    Args:
        state: State dict (optional context)
        concept: Specific concept to generate data for
        num_samples: Number of samples to generate

    Returns:
        dict with generated_count, synthetic_samples
    """
    state = parse_state(state)
    api_key, base_url, model_name = get_generator_config()
    augmenter = DataAugmenter(api_key, base_url, model_name)

    prompt = f"""Generate {num_samples} Q&A pairs that test the concept of: {concept}

Requirements:
1. Include definition questions
2. Include example-based questions
3. Include application/practice questions
4. Include edge case or tricky questions
5. Answers should be educational and explain the reasoning

Respond in JSON format:
{{
    "synthetic_data": [
        {{
            "question": "question about {concept}",
            "answer": "detailed answer with explanation",
            "question_type": "definition|example|application|edge_case"
        }}
    ]
}}
"""

    response = await augmenter._call_llm(prompt)

    synthetic_samples = []
    try:
        result = json.loads(response)
        data = result.get("synthetic_data", [])
    except json.JSONDecodeError:
        data = []
        if not data:
            return {"status": "error", "message": "Failed to parse generated data: LLM returned malformed JSON"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse generated data: {str(e)}"}

    for i, item in enumerate(data):
        synthetic_samples.append(DataSample(
            id=f"synth_c_{i}",
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            metadata={
                "concept": concept,
                "question_type": item.get("question_type", "definition"),
                "is_synthetic": True
            }
        ))

    # Save generated data
    output_dir = get_output_dir()
    output_path = os.path.join(output_dir, f"synthetic_concept_{concept.replace(' ', '_')[:20]}.csv")

    synthetic_data = [{
        "id": s.id,
        "question": s.question,
        "answer": s.answer,
        "concept": concept,
        "question_type": s.metadata.get("question_type", "definition")
    } for s in synthetic_samples]

    output_df = pd.DataFrame(synthetic_data)
    output_df.to_csv(output_path, index=False)

    return {
        "status": "success",
        "concept": concept,
        "generated_count": len(synthetic_samples),
        "output_path": output_path,
        "synthetic_samples": synthetic_samples[:10]
    }


@tool
async def generate_adversarial_synthetic(state: Union[dict, str], base_domain: str, num_samples: int = 5) -> dict:
    """Generate adversarial/special case synthetic data.

    Creates challenging Q&A pairs designed to test model robustness:
    - Ambiguous questions
    - Counterfactual scenarios
    - Multi-hop reasoning
    - Common misconceptions

    Args:
        state: State dict (optional context)
        base_domain: Base domain to create adversarial examples for
        num_samples: Number of samples to generate per type

    Returns:
        dict with generated_count, synthetic_samples
    """
    state = parse_state(state)
    api_key, base_url, model_name = get_generator_config()
    augmenter = DataAugmenter(api_key, base_url, model_name)

    prompt = f"""Generate adversarial/special case Q&A pairs for {base_domain}.

Create challenging examples that test model robustness:
1. Ambiguous questions with multiple interpretations
2. Counterfactual scenarios ("what if...")
3. Multi-hop reasoning questions
4. Common misconceptions to correct
5. Edge cases that are often misunderstood

Generate {num_samples * 5} total samples covering these categories.

Respond in JSON format:
{
    "synthetic_data": [
        {
            "question": "challenging question",
            "answer": "helpful answer addressing the challenge",
            "adversarial_type": "ambiguous|counterfactual|multi_hop|misconception|edge_case",
            "challenge_description": "what makes this challenging"
        }
    ]
}
"""

    response = await augmenter._call_llm(prompt)

    synthetic_samples = []
    try:
        result = json.loads(response)
        data = result.get("synthetic_data", [])
    except json.JSONDecodeError:
        data = []
        if not data:
            return {"status": "error", "message": "Failed to parse generated data: LLM returned malformed JSON"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse generated data: {str(e)}"}

    for i, item in enumerate(data):
        synthetic_samples.append(DataSample(
            id=f"adv_{i}",
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            metadata={
                "base_domain": base_domain,
                "adversarial_type": item.get("adversarial_type", "edge_case"),
                "challenge": item.get("challenge_description", ""),
                "is_synthetic": True,
                "is_adversarial": True
            }
        ))

    # Save generated data
    output_dir = get_output_dir()
    output_path = os.path.join(output_dir, f"adversarial_{base_domain.replace(' ', '_')[:15]}.csv")

    synthetic_data = [{
        "id": s.id,
        "question": s.question,
        "answer": s.answer,
        "adversarial_type": s.metadata.get("adversarial_type", "edge_case"),
        "challenge": s.metadata.get("challenge", "")
    } for s in synthetic_samples]

    output_df = pd.DataFrame(synthetic_data)
    output_df.to_csv(output_path, index=False)

    return {
        "status": "success",
        "base_domain": base_domain,
        "generated_count": len(synthetic_samples),
        "output_path": output_path,
        "synthetic_samples": synthetic_samples[:10]
    }
