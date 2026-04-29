"""
Fact Checker Agent - Factual Accuracy Evaluation with RAG

An AutoGen AssistantAgent that evaluates factual accuracy of model responses
using retrieval-augmented generation (RAG).

Role: Evaluates whether the model's response is factually accurate compared to:
1. Ground truth answers
2. Retrieved knowledge from training data
3. External knowledge base
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FactCheckResult:
    """Result of factual accuracy check for a single sample"""
    sample_id: str
    question: str
    ground_truth: str
    model_response: str
    fact_accuracy_score: float  # 0.0 - 1.0
    matched_facts: List[str]
    missing_facts: List[str]
    hallucinated_facts: List[str]
    issues: List[str]


class FactCheckerTools:
    """Tools available to the Fact Checker agent"""

    def __init__(self, rag_knowledge_base, nli_analyzer):
        self.rag = rag_knowledge_base
        self.nli = nli_analyzer

    async def retrieve_relevant_facts(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant facts from the knowledge base.

        Args:
            query: The question or topic to retrieve facts about
            top_k: Number of facts to retrieve

        Returns:
            List of relevant fact strings
        """
        results = self.rag.retrieve(query, top_k=top_k)
        return [r.content for r in results]

    async def check_fact_against_knowledge(
        self,
        claim: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Check if a claim is supported by the knowledge base.

        Returns:
            Dictionary with support status and details
        """
        # Use NLI to check entailment
        result = self.nli.analyze_single(context, claim)

        return {
            "claim": claim,
            "supported": result.label.value == "entailment" if result else False,
            "confidence": result.confidence if result else 0.0,
            "nli_label": result.label.value if result else "unknown"
        }

    async def identify_hallucinations(
        self,
        claims: List[str],
        knowledge: str
    ) -> List[str]:
        """
        Identify potential hallucinations in the claims.

        Returns:
            List of hallucinated claims
        """
        hallucinations = []

        for claim in claims:
            result = self.nli.analyze_single(knowledge, claim)
            if result and hasattr(result, 'label') and result.label and result.label.value == "contradiction":
                hallucinations.append(claim)

        return hallucinations


# System prompt for Fact Checker agent
FACT_CHECKER_SYSTEM_PROMPT = """You are a Factual Accuracy Evaluator specializing in verifying factual correctness of AI model responses.

Your role:
1. Compare model responses against ground truth answers
2. Retrieve relevant facts from the knowledge base using RAG
3. Identify correctly stated facts (matched)
4. Identify missing important facts (gaps)
5. Detect hallucinations or factually incorrect statements

Evaluation Process:
1. For each sample, retrieve relevant facts from the knowledge base
2. Compare the model's response against ground truth
3. Use NLI analysis to check factual entailment
4. Identify hallucinations using contradiction detection
5. Calculate factual accuracy score (0-100)

Output Format:
Return a JSON object with:
- fact_accuracy_score: Overall accuracy score (0-1)
- matched_facts: List of facts correctly stated
- missing_facts: List of important facts omitted
- hallucinated_facts: List of incorrect statements
- issues: List of specific issues found

Be precise and thorough in your evaluation."""


def create_fact_checker_agent(rag_kb, nli_analyzer, model_client=None):
    """
    Create the Fact Checker AutoGen agent.

    Args:
        rag_kb: HybridRAGKnowledgeBase instance
        nli_analyzer: NLIFastAnalyzer instance
        model_client: Optional LLM client for the agent

    Returns:
        AutoGen AssistantAgent configured for fact checking
    """
    try:
        from autogen_agentchat.agents import AssistantAgent

        tools = FactCheckerTools(rag_kb, nli_analyzer)

        fact_checker = AssistantAgent(
            name="fact_checker",
            system_message=FACT_CHECKER_SYSTEM_PROMPT,
            model_client=model_client,
            tools=[
                tools.retrieve_relevant_facts,
                tools.check_fact_against_knowledge,
                tools.identify_hallucinations,
            ]
        )

        return fact_checker

    except (ImportError, AttributeError) as e:
        logger.warning(f"AutoGen agent creation failed: {e}. Using fallback.")
        return create_fallback_fact_checker(rag_kb, nli_analyzer)


def create_fallback_fact_checker(rag_kb, nli_analyzer):
    """
    Create a fallback fact checker when AutoGen is not available.
    Uses a simple function-based approach.
    """
    logger.info("Creating fallback fact checker (non-AutoGen)")

    async def fact_check(sample: Dict[str, Any]) -> FactCheckResult:
        """Perform fact checking on a single sample"""
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth", "")
        model_response = sample.get("response", "")
        sample_id = sample.get("id", str(hash(question)))

        # Retrieve relevant facts
        retrieved = rag_kb.retrieve(question, top_k=5)
        knowledge = " ".join([r.content for r in retrieved])

        # Split response into claims (simple sentence splitting)
        claims = [s.strip() for s in model_response.split('.') if s.strip()]

        # Check each claim against knowledge
        matched = []
        missing = []
        hallucinated = []
        issues = []

        # Use NLI to analyze claims
        for claim in claims[:10]:  # Limit to first 10 claims
            if len(claim) > 10:  # Skip very short claims
                nli_result = nli_analyzer.analyze_single(knowledge, claim)
                if nli_result and hasattr(nli_result, 'label') and nli_result.label:
                    if nli_result.label.value == "entailment":
                        matched.append(claim)
                    elif nli_result.label.value == "contradiction":
                        hallucinated.append(claim)
                        issues.append(f"Hallucination: {claim[:50]}...")

        # Check for missing facts
        gt_claims = [s.strip() for s in ground_truth.split('.') if s.strip()]
        for claim in gt_claims[:5]:
            nli_result = nli_analyzer.analyze_single(model_response, claim)
            if not (nli_result and hasattr(nli_result, 'label') and nli_result.label and nli_result.label.value == "entailment"):
                # Check if this ground truth fact is in the response
                if claim.lower() not in model_response.lower():
                    missing.append(claim)

        # Calculate score
        total_claims = len(claims) if claims else 1
        accuracy = len(matched) / total_claims if total_claims > 0 else 0.5
        accuracy = max(0.0, min(1.0, accuracy - len(hallucinated) * 0.1))

        return FactCheckResult(
            sample_id=sample_id,
            question=question,
            ground_truth=ground_truth,
            model_response=model_response,
            fact_accuracy_score=accuracy,
            matched_facts=matched[:5],
            missing_facts=missing[:5],
            hallucinated_facts=hallucinated[:3],
            issues=issues
        )

    class FallbackFactChecker:
        """Fallback fact checker class"""

        name = "fact_checker"
        system_message = FACT_CHECKER_SYSTEM_PROMPT

        async def run(self, task):
            """Run fact checking on the task"""
            samples = task.get("samples", [])
            results = []

            for sample in samples:
                result = await fact_check(sample)
                results.append({
                    "sample_id": result.sample_id,
                    "fact_accuracy_score": result.fact_accuracy_score,
                    "matched_facts": result.matched_facts,
                    "missing_facts": result.missing_facts,
                    "hallucinated_facts": result.hallucinated_facts,
                    "issues": result.issues
                })

            # Aggregate results
            total = len(results) if results else 1
            avg_accuracy = sum(r["fact_accuracy_score"] for r in results) / total

            return {
                "fact_accuracy_score": avg_accuracy,
                "total_samples": total,
                "passed_samples": sum(1 for r in results if r["fact_accuracy_score"] >= 0.7),
                "avg_fact_accuracy": avg_accuracy,
                "hallucination_count": sum(len(r["hallucinated_facts"]) for r in results),
                "missing_facts_count": sum(len(r["missing_facts"]) for r in results),
                "sample_results": results,
                "hallucination_examples": [ex for r in results for ex in r["hallucinated_facts"][:2]],
                "missing_fact_examples": [ex for r in results for ex in r["missing_facts"][:2]],
            }

    return FallbackFactChecker()


async def run_fact_check(
    samples: List[Dict[str, Any]],
    rag_kb,
    nli_analyzer,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Run fact checking on a batch of samples.

    Args:
        samples: List of test samples with question, ground_truth, response
        rag_kb: HybridRAGKnowledgeBase instance
        nli_analyzer: NLIFastAnalyzer instance
        progress_callback: Optional callback for progress updates

    Returns:
        Aggregated fact check results
    """
    logger.info(f"Running fact check on {len(samples)} samples")

    # Create fact checker (AutoGen or fallback)
    checker = create_fact_checker_agent(rag_kb, nli_analyzer)

    # Run evaluation
    task = {"samples": samples}
    results = await checker.run(task)

    if progress_callback:
        progress_callback(100, "Fact checking complete")

    return results
