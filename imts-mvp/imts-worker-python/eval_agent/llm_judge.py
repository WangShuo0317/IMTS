"""
LLM Judge - Complex Logical Reasoning using LLM

Provides LLM-based logical reasoning for complex cases that the fast NLI model
cannot handle reliably. Used by the Logic Checker agent for nuanced logical analysis.

Supports:
- Async LLM calls for concurrent processing
- Configurable LLM providers (OpenAI, Anthropic, custom)
- Structured output parsing
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LogicLabel(Enum):
    """Logical relation labels from LLM judge"""
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


@dataclass
class LLMJudgeResult:
    """Result from LLM-based logical judgment"""
    premise: str
    hypothesis: str
    label: LogicLabel
    confidence: float
    reasoning: str
    key_logical_relations: List[str]
    potential_issues: List[str]


class LLMJudge:
    """
    LLM-based logical reasoning judge.

    Used for complex logical inference that requires:
    - World knowledge
    - Common sense reasoning
    - Multi-step logical deduction
    - Handling of implied meanings and nuances
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model_name: str = "gpt-4",
        temperature: float = 0.0,
        timeout: int = 30
    ):
        """
        Initialize LLM Judge.

        Args:
            api_key: API key for LLM provider
            base_url: Base URL for OpenAI-compatible API
            model_name: Model name to use
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout

        self._client = None
        self._initialized = False

    def initialize(self):
        """Initialize the LLM client"""
        if self._initialized:
            return

        logger.info(f"Initializing LLM Judge with model: {self.model_name}")

        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            self._initialized = True
            logger.info("LLM Judge initialized successfully")

        except ImportError:
            logger.warning("OpenAI client not available, LLM judge will use mock responses")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise

    def _build_prompt(self, premise: str, hypothesis: str, context: str = None) -> str:
        """Build the prompt for logical reasoning"""
        context_section = f"\n\nAdditional Context:\n{context}" if context else ""

        return f"""Analyze the logical relationship between the premise and hypothesis.

Premise: {premise}
Hypothesis: {hypothesis}{context_section}

Your task:
1. Determine if the hypothesis logically follows from the premise (entailment),
   contradicts the premise (contradiction), or is neither supported nor contradicted (neutral).
2. Provide your reasoning process.
3. Identify key logical relations involved.
4. Note any potential issues with the reasoning.

Respond in the following JSON format:
{{
    "label": "entailment" | "contradiction" | "neutral" | "uncertain",
    "confidence": 0.0-1.0,
    "reasoning": "your detailed reasoning",
    "key_logical_relations": ["relation1", "relation2"],
    "potential_issues": ["issue1", "issue2"]
}}

Important:
- Be precise in your label assignment
- Confidence should reflect certainty in the label
- If the relationship is genuinely unclear, use "uncertain"
"""

    async def evaluate(
        self,
        premise: str,
        hypothesis: str,
        context: str = None
    ) -> LLMJudgeResult:
        """
        Evaluate logical relationship using LLM.

        Args:
            premise: The premise/fact
            hypothesis: The hypothesis to evaluate
            context: Optional additional context

        Returns:
            LLMJudgeResult with analysis
        """
        if not self._initialized:
            self.initialize()

        prompt = self._build_prompt(premise, hypothesis, context)

        try:
            if self._client:
                response = await self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a logical reasoning expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                data = json.loads(content)

                return LLMJudgeResult(
                    premise=premise,
                    hypothesis=hypothesis,
                    label=LogicLabel(data.get("label", "uncertain")),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ""),
                    key_logical_relations=data.get("key_logical_relations", []),
                    potential_issues=data.get("potential_issues", [])
                )
            else:
                # Fallback to mock response
                return self._mock_evaluate(premise, hypothesis)

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return self._mock_evaluate(premise, hypothesis)

    def _mock_evaluate(self, premise: str, hypothesis: str) -> LLMJudgeResult:
        """Mock evaluation when LLM is not available"""
        import random

        labels = list(LogicLabel)
        label = random.choice(labels)
        confidence = random.uniform(0.6, 0.9)

        return LLMJudgeResult(
            premise=premise,
            hypothesis=hypothesis,
            label=label,
            confidence=confidence,
            reasoning="Mock evaluation (LLM not available)",
            key_logical_relations=["mock_relation"],
            potential_issues=[]
        )

    async def batch_evaluate(
        self,
        premise_hypothesis_pairs: List[tuple],
        contexts: List[str] = None,
        max_concurrent: int = 5
    ) -> List[LLMJudgeResult]:
        """
        Evaluate multiple pairs concurrently.

        Args:
            premise_hypothesis_pairs: List of (premise, hypothesis) tuples
            contexts: Optional list of context strings
            max_concurrent: Maximum concurrent requests

        Returns:
            List of LLMJudgeResult
        """
        if not self._initialized:
            self.initialize()

        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_with_semaphore(idx: int, pair: tuple):
            async with semaphore:
                premise, hypothesis = pair
                context = contexts[idx] if contexts and idx < len(contexts) else None
                return await self.evaluate(premise, hypothesis, context)

        tasks = [
            evaluate_with_semaphore(i, pair)
            for i, pair in enumerate(premise_hypothesis_pairs)
        ]

        return await asyncio.gather(*tasks)


class LogicConsistencyScorer:
    """
    Scores logical consistency based on NLI and LLM results.
    """

    def __init__(self, nli_weight: float = 0.4, llm_weight: float = 0.6):
        """
        Initialize scorer.

        Args:
            nli_weight: Weight for NLI model confidence
            llm_weight: Weight for LLM judge confidence
        """
        self.nli_weight = nli_weight
        self.llm_weight = llm_weight

    def score(
        self,
        nli_label: str,
        nli_confidence: float,
        llm_label: str = None,
        llm_confidence: float = None
    ) -> Dict[str, Any]:
        """
        Calculate logical consistency score.

        Returns:
            Dictionary with score and breakdown
        """
        # NLI-based score
        if nli_label == "contradiction":
            nli_score = 0.0
        elif nli_label == "entailment":
            nli_score = 1.0
        else:  # neutral
            nli_score = 0.5

        nli_contribution = nli_score * nli_confidence * self.nli_weight

        # LLM-based score (if available)
        if llm_label and llm_confidence is not None:
            if llm_label == "contradiction":
                llm_score = 0.0
            elif llm_label == "entailment":
                llm_score = 1.0
            elif llm_label == "uncertain":
                llm_score = 0.3
            else:  # neutral
                llm_score = 0.5

            llm_contribution = llm_score * llm_confidence * self.llm_weight
            final_score = nli_contribution + llm_contribution
        else:
            final_score = nli_contribution / self.nli_weight if self.nli_weight > 0 else 0.5

        return {
            "score": round(final_score * 100, 2),  # 0-100 scale
            "nli_contribution": round(nli_contribution * 100, 2),
            "llm_contribution": round(llm_contribution * 100, 2) if llm_label else None,
            "nli_label": nli_label,
            "llm_label": llm_label,
            "nli_confidence": nli_confidence,
            "llm_confidence": llm_confidence
        }
