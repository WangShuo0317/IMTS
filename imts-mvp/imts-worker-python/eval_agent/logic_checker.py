"""
Logic Checker Agent - Logical Consistency Evaluation using NLI + LLM

An AutoGen AssistantAgent that evaluates logical consistency and reasoning
capabilities using a hybrid approach:
- Fast NLI model for quick checks
- LLM judge for complex logical reasoning
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LogicCheckResult:
    """Result of logical consistency check for a single sample"""
    sample_id: str
    question: str
    ground_truth: str
    model_response: str
    logic_consistency_score: float  # 0.0 - 1.0
    nli_label: str
    nli_confidence: float
    contradictions: List[str]
    logical_gaps: List[str]
    issues: List[str]


class LogicCheckerTools:
    """Tools available to the Logic Checker agent"""

    def __init__(self, nli_analyzer, llm_judge):
        self.nli = nli_analyzer
        self.llm = llm_judge

    async def fast_nli_check(
        self,
        premise: str,
        hypothesis: str
    ) -> Dict[str, Any]:
        """
        Fast NLI check using lightweight model.

        Returns:
            Dictionary with NLI label and confidence
        """
        result = self.nli.analyze_single(premise, hypothesis)

        return {
            "label": result.label.value if result else "unknown",
            "confidence": result.confidence if result else 0.0,
            "entailment_prob": result.entailment_prob if result else 0.0,
            "contradiction_prob": result.contradiction_prob if result else 0.0,
            "neutral_prob": result.neutral_prob if result else 0.0
        }

    async def batch_nli_check(
        self,
        premise_hypothesis_pairs: List[Tuple[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Batch NLI check for efficiency.

        Returns:
            List of NLI results
        """
        result = self.nli.analyze(premise_hypothesis_pairs)

        return [
            {
                "label": sample.label.value if sample else "unknown",
                "confidence": sample.confidence if sample else 0.0
            }
            for sample in result.samples
        ]

    async def deep_logic_analysis(
        self,
        premise: str,
        hypothesis: str,
        context: str = None
    ) -> Dict[str, Any]:
        """
        Deep logical analysis using LLM for complex cases.

        Returns:
            Dictionary with LLM-based logical analysis
        """
        result = await self.llm.evaluate(premise, hypothesis, context)

        return {
            "label": result.label.value if result else "uncertain",
            "confidence": result.confidence if result else 0.0,
            "reasoning": result.reasoning if result else "",
            "key_logical_relations": result.key_logical_relations if result else [],
            "potential_issues": result.potential_issues if result else []
        }


# System prompt for Logic Checker agent
LOGIC_CHECKER_SYSTEM_PROMPT = """You are a Logical Consistency Evaluator specializing in assessing the logical coherence and reasoning quality of AI model responses.

Your role:
1. Evaluate logical consistency between model responses and ground truth
2. Detect logical contradictions and inconsistencies
3. Assess the quality of logical reasoning chains
4. Use fast NLI for quick checks, LLM judge for complex reasoning

Evaluation Approach:
1. First, use NLI model for quick consistency checks on sentence pairs
2. For complex cases, use LLM judge for deeper analysis
3. Identify specific contradictions and logical gaps
4. Calculate logical consistency score (0-100)

Output Format:
Return a JSON object with:
- logic_consistency_score: Overall consistency score (0-1)
- contradiction_count: Number of contradictions found
- logic_issues: List of specific logical issues
- nli_summary: Summary of NLI analysis

Be thorough in detecting subtle logical inconsistencies."""


def create_logic_checker_agent(nli_analyzer, llm_judge, model_client=None):
    """
    Create the Logic Checker AutoGen agent.

    Args:
        nli_analyzer: NLIFastAnalyzer instance
        llm_judge: LLMJudge instance
        model_client: Optional LLM client for the agent

    Returns:
        AutoGen AssistantAgent configured for logic checking
    """
    try:
        from autogen_agentchat.agents import AssistantAgent

        tools = LogicCheckerTools(nli_analyzer, llm_judge)

        logic_checker = AssistantAgent(
            name="logic_checker",
            system_message=LOGIC_CHECKER_SYSTEM_PROMPT,
            model_client=model_client,
            tools=[
                tools.fast_nli_check,
                tools.batch_nli_check,
                tools.deep_logic_analysis,
            ]
        )

        return logic_checker

    except (ImportError, AttributeError) as e:
        logger.warning(f"AutoGen agent creation failed: {e}. Using fallback.")
        return create_fallback_logic_checker(nli_analyzer, llm_judge)


def create_fallback_logic_checker(nli_analyzer, llm_judge):
    """
    Create a fallback logic checker when AutoGen is not available.
    """
    logger.info("Creating fallback logic checker (non-AutoGen)")

    async def logic_check(sample: Dict[str, Any]) -> LogicCheckResult:
        """Perform logic checking on a single sample"""
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth", "")
        model_response = sample.get("response", "")
        sample_id = sample.get("id", str(hash(question)))

        # Break into sentences for NLI analysis
        gt_sentences = [s.strip() for s in ground_truth.split('.') if s.strip()]
        response_sentences = [s.strip() for s in model_response.split('.') if s.strip()]

        contradictions = []
        logical_gaps = []
        issues = []
        nli_results = []

        # Check each response sentence against each ground truth sentence
        for resp_sent in response_sentences[:5]:  # Limit checks
            for gt_sent in gt_sentences[:3]:
                if len(resp_sent) > 10 and len(gt_sent) > 10:
                    result = nli_analyzer.analyze_single(resp_sent, gt_sent)
                    nli_results.append(result)

                    if result and hasattr(result, 'label') and result.label and result.label.value == "contradiction":
                        contradictions.append(f"{resp_sent[:50]}... vs {gt_sent[:50]}...")
                        issues.append("Logical contradiction detected")

        # Use LLM judge for complex cases (if many contradictions found)
        if len(contradictions) > 2:
            llm_result = await llm_judge.evaluate(model_response, ground_truth, question)
            if llm_result and hasattr(llm_result, 'label') and llm_result.label and llm_result.label.value == "contradiction":
                contradictions.append(f"LLM detected: {llm_result.reasoning[:100]}...")

        # Calculate score
        total_pairs = len(nli_results) if nli_results else 1
        consistency = max(0.0, 1.0 - (len(contradictions) * 0.2))
        consistency = min(1.0, consistency)

        # Additional scoring based on NLI confidence
        if nli_results:
            avg_confidence = sum(r.confidence for r in nli_results if r) / len(nli_results)
            consistency = consistency * (0.7 + 0.3 * avg_confidence)

        return LogicCheckResult(
            sample_id=sample_id,
            question=question,
            ground_truth=ground_truth,
            model_response=model_response,
            logic_consistency_score=consistency,
            nli_label="contradiction" if contradictions else "entailment",
            nli_confidence=avg_confidence if nli_results else 0.5,
            contradictions=contradictions[:3],
            logical_gaps=logical_gaps[:3],
            issues=issues
        )

    class FallbackLogicChecker:
        """Fallback logic checker class"""

        name = "logic_checker"
        system_message = LOGIC_CHECKER_SYSTEM_PROMPT

        async def run(self, task):
            """Run logic checking on the task"""
            samples = task.get("samples", [])
            results = []
            total_contradictions = 0
            total_issues = []

            for sample in samples:
                result = await logic_check(sample)
                results.append({
                    "sample_id": result.sample_id,
                    "logic_consistency_score": result.logic_consistency_score,
                    "nli_label": result.nli_label,
                    "contradictions": result.contradictions,
                    "issues": result.issues
                })
                total_contradictions += len(result.contradictions)
                total_issues.extend(result.issues)

            # Aggregate results
            total = len(results) if results else 1
            avg_consistency = sum(r["logic_consistency_score"] for r in results) / total

            return {
                "logic_consistency_score": avg_consistency,
                "total_samples": total,
                "avg_logic_consistency": avg_consistency,
                "contradiction_count": total_contradictions,
                "logic_issues": list(set(total_issues))[:10],
                "sample_results": results,
                "contradiction_examples": [c for r in results for c in r["contradictions"][:1]],
                "relevance_score": 0.85,  # Placeholder
                "conciseness_score": 0.75,  # Placeholder
            }

    return FallbackLogicChecker()


async def run_logic_check(
    samples: List[Dict[str, Any]],
    nli_analyzer,
    llm_judge,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Run logic checking on a batch of samples.

    Args:
        samples: List of test samples with question, ground_truth, response
        nli_analyzer: NLIFastAnalyzer instance
        llm_judge: LLMJudge instance
        progress_callback: Optional callback for progress updates

    Returns:
        Aggregated logic check results
    """
    logger.info(f"Running logic check on {len(samples)} samples")

    # Create logic checker (AutoGen or fallback)
    checker = create_logic_checker_agent(nli_analyzer, llm_judge)

    # Run evaluation
    task = {"samples": samples}
    results = await checker.run(task)

    if progress_callback:
        progress_callback(100, "Logic checking complete")

    return results
