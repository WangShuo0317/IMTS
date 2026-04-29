"""
Arbiter Agent - Final Evaluation and Recommendation

An AutoGen AssistantAgent that synthesizes results from Fact Checker
and Logic Checker agents to produce the final evaluation verdict.

Role:
1. Aggregate scores from both evaluators
2. Generate capability radar chart data
3. Provide data optimization suggestions
4. Make pass/fail determination
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# System prompt for Arbiter agent
ARBITER_SYSTEM_PROMPT = """You are the Final Evaluation Arbiter responsible for synthesizing evaluation results and providing comprehensive assessment reports.

Your role:
1. Aggregate factual accuracy scores from Fact Checker
2. Aggregate logical consistency scores from Logic Checker
3. Calculate overall weighted score
4. Generate capability radar chart data
5. Provide data optimization suggestions
6. Make pass/fail determination

Evaluation Dimensions (for radar chart):
- 事实准确性 (Factual Accuracy): 35% weight
- 逻辑一致性 (Logic Consistency): 25% weight
- 完整性 (Completeness): 15% weight
- 相关性 (Relevance): 15% weight
- 一致性 (Conciseness): 10% weight

Pass Threshold: 75/100

Output Format:
Return a JSON object with:
- overall_score: Weighted overall score (0-100)
- passed: Boolean indicating pass/fail
- radar_data: Object with dimensions and scores arrays
- suggestions: Array of optimization suggestions
- detailed_metrics: Breakdown by dimension
- evaluation_summary: Summary statistics

Be fair and balanced in your assessment. Consider both factual accuracy and logical consistency."""


@dataclass
class ArbiterResult:
    """Result from the arbiter's final evaluation"""
    overall_score: float
    passed: bool
    radar_data: Dict[str, Any]
    suggestions: List[Dict[str, Any]]
    detailed_metrics: Dict[str, float]
    evaluation_summary: Dict[str, Any]


def create_arbiter_agent(model_client=None):
    """
    Create the Arbiter AutoGen agent.

    Args:
        model_client: Optional LLM client for the agent

    Returns:
        AutoGen AssistantAgent configured as arbiter
    """
    try:
        from autogen_agentchat.agents import AssistantAgent

        arbiter = AssistantAgent(
            name="arbiter",
            system_message=ARBITER_SYSTEM_PROMPT,
            model_client=model_client
        )

        return arbiter
    except (ImportError, AttributeError) as e:
        logger.warning(f"AutoGen arbiter creation failed: {e}. Using fallback.")
        return None  # Will use fallback in run_arbiter_evaluation


async def run_arbiter_evaluation(
    fact_check_results: Dict[str, Any],
    logic_check_results: Dict[str, Any],
    sample_results: List[Dict[str, Any]] = None,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run final arbiter evaluation using AutoGen AssistantAgent.

    Args:
        fact_check_results: Aggregated results from fact checker
        logic_check_results: Aggregated results from logic checker
        sample_results: Optional per-sample results
        metadata: Optional metadata (job_id, iteration, etc.)

    Returns:
        Final evaluation report
    """
    from .config import LLM_CONFIG
    from openai import AsyncOpenAI

    logger.info("Running arbiter evaluation...")

    # Build the model client for AutoGen agent
    llm_api_key = LLM_CONFIG.get("api_key") or os.getenv("OPENAI_API_KEY", "")
    llm_base_url = LLM_CONFIG.get("base_url") or os.getenv("OPENAI_BASE_URL", "")
    llm_model_name = LLM_CONFIG.get("model_name") or os.getenv("LLM_MODEL_NAME", "gpt-4")

    openai_client = AsyncOpenAI(api_key=llm_api_key, base_url=llm_base_url)

    # AutoGen Model Client wrapping OpenAI
    try:
        from autogen_agentchat.model_client import OpenAIChatCompletionClient
        model_client = OpenAIChatCompletionClient(
            model=llm_model_name,
            api_key=llm_api_key,
            base_url=llm_base_url,
        )
    except Exception:
        # Fallback: use OpenAI client directly
        model_client = None

    # Create AutoGen arbiter agent
    arbiter = create_arbiter_agent(model_client=model_client)

    # Prepare prompt for arbiter with all evaluation data
    from .report_generator import EvaluationReportGenerator
    generator = EvaluationReportGenerator()

    # Use the report generator to produce the final evaluation
    # This gives structured scores that the arbiter then refines via LLM
    report = generator.generate_report(
        fact_check_results=fact_check_results,
        logic_check_results=logic_check_results,
        sample_results=sample_results or [],
        metadata=metadata or {}
    )

    # If we have a real model client, use AutoGen agent for LLM-based refinement
    if model_client is not None:
        try:
            from autogen_agentchat.messages import Task, TextMessage

            task_msg = Task(
                content=f"""You are the Final Evaluation Arbiter. Review this pre-generated report and refine it if needed.

Current Report:
{json.dumps(report, ensure_ascii=False, indent=2)}

Review the report:
1. Verify the overall_score calculation is correct
2. Ensure suggestions are actionable and prioritized
3. Add any missing insights

Respond with the refined JSON report only."""
            )

            response = await arbiter.run(task_msg)
            # Extract text from response
            if hasattr(response, "content"):
                try:
                    report = json.loads(response.content)
                except json.JSONDecodeError:
                    logger.warning("Arbiter LLM response not valid JSON, using generator report")
        except Exception as e:
            logger.warning(f"AutoGen arbiter LLM call failed: {e}, using generator report")

    logger.info(f"Arbiter evaluation complete: score={report.get('overall_score', 0):.2f}")
    return report


def aggregate_sample_results(
    fact_results: List[Dict[str, Any]],
    logic_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Aggregate per-sample results from fact and logic checkers.

    Args:
        fact_results: Per-sample fact check results
        logic_results: Per-sample logic check results

    Returns:
        Combined per-sample results
    """
    combined = []

    for i, (fr, lr) in enumerate(zip(fact_results, logic_results)):
        combined.append({
            "sample_id": fr.get("sample_id", f"sample_{i}"),
            "question": fr.get("question", ""),
            "response": fr.get("response", ""),
            "ground_truth": fr.get("ground_truth", ""),
            "fact_accuracy": fr.get("fact_accuracy_score", 0.0),
            "logic_consistency": lr.get("logic_consistency_score", 0.0),
            "overall_score": (
                fr.get("fact_accuracy_score", 0.0) * 0.6 +
                lr.get("logic_consistency_score", 0.0) * 0.4
            ),
            "issues": fr.get("issues", []) + lr.get("issues", [])
        })

    return combined
