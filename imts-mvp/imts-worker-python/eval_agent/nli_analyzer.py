"""
NLI Analyzer - Fast Natural Language Inference

Provides fast NLI (Natural Language Inference) analysis using lightweight models.
Used by the Logic Checker agent for quick logical consistency checks.

Supports:
- Batch processing for efficiency
- Three-way classification: entailment, contradiction, neutral
- CPU-friendly models for deployment flexibility
"""

import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class NLILabel(Enum):
    """NLI classification labels"""
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


@dataclass
class NLISample:
    """Single NLI sample"""
    premise: str
    hypothesis: str
    label: NLILabel = None
    confidence: float = 0.0
    entailment_prob: float = 0.0
    contradiction_prob: float = 0.0
    neutral_prob: float = 0.0


@dataclass
class NLIResult:
    """NLI analysis result for a batch"""
    samples: List[NLISample]
    summary: Dict[str, Any]

    @property
    def avg_confidence(self) -> float:
        return np.mean([s.confidence for s in self.samples]) if self.samples else 0.0

    @property
    def contradiction_rate(self) -> float:
        count = sum(1 for s in self.samples if s.label == NLILabel.CONTRADICTION)
        return count / len(self.samples) if self.samples else 0.0

    @property
    def consistency_score(self) -> float:
        """Calculate consistency score (1 - contradiction_rate)"""
        return 1.0 - self.contradiction_rate


class NLIFastAnalyzer:
    """
    Fast NLI analyzer using DeBERTa-v3-xsmall model.

    Optimized for:
    - Batch processing to maximize GPU/CPU utilization
    - Low memory footprint (xsmall model)
    - Quick inference for real-time evaluation
    """

    def __init__(
        self,
        model_name: str = "DeBERTa-v3-xsmall",
        batch_size: int = 32,
        device: str = None
    ):
        """
        Initialize NLI analyzer.

        Args:
            model_name: HuggingFace model name for NLI
            batch_size: Batch size for inference
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if self._has_cuda() else "cpu")

        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _has_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def initialize(self):
        """Load the NLI model and tokenizer"""
        if self._initialized:
            return

        logger.info(f"Loading NLI model: {self.model_name} on {self.device}")

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            self._initialized = True
            logger.info(f"NLI model loaded successfully (device: {self.device})")

        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise

    def analyze(
        self,
        premise_hypothesis_pairs: List[Tuple[str, str]]
    ) -> NLIResult:
        """
        Analyze a batch of premise-hypothesis pairs for NLI.

        Args:
            premise_hypothesis_pairs: List of (premise, hypothesis) tuples

        Returns:
            NLIResult with analysis results
        """
        if not self._initialized:
            self.initialize()

        if not premise_hypothesis_pairs:
            return NLIResult(samples=[], summary={})

        logger.info(f"Analyzing {len(premise_hypothesis_pairs)} pairs with NLI")

        samples = []
        all_entailment = []
        all_contradiction = []
        all_neutral = []

        # Process in batches
        for i in range(0, len(premise_hypothesis_pairs), self.batch_size):
            batch = premise_hypothesis_pairs[i:i + self.batch_size]
            batch_samples = self._process_batch(batch)

            for sample in batch_samples:
                samples.append(sample)
                all_entailment.append(sample.entailment_prob)
                all_contradiction.append(sample.contradiction_prob)
                all_neutral.append(sample.neutral_prob)

        # Calculate summary statistics
        summary = {
            "total_samples": len(samples),
            "avg_entailment_prob": float(np.mean(all_entailment)) if all_entailment else 0.0,
            "avg_contradiction_prob": float(np.mean(all_contradiction)) if all_contradiction else 0.0,
            "avg_neutral_prob": float(np.mean(all_neutral)) if all_neutral else 0.0,
            "contradiction_count": sum(1 for s in samples if s.label == NLILabel.CONTRADICTION),
            "entailment_count": sum(1 for s in samples if s.label == NLILabel.ENTAILMENT),
            "neutral_count": sum(1 for s in samples if s.label == NLILabel.NEUTRAL),
            "model_name": self.model_name,
            "device": self.device,
        }

        return NLIResult(samples=samples, summary=summary)

    def _process_batch(self, batch: List[Tuple[str, str]]) -> List[NLISample]:
        """Process a single batch of pairs"""
        import torch
        from torch.nn.functional import softmax

        premises, hypotheses = zip(*batch)

        # Tokenize
        inputs = self.tokenizer(
            list(premises),
            list(hypotheses),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get probabilities
        probs = softmax(logits, dim=-1)
        probs_np = probs.cpu().numpy()

        # Parse results
        # Model label mapping: typically 0=entailment, 1=neutral, 2=contradiction
        samples = []
        for i, (premise, hypothesis) in enumerate(batch):
            # Try to determine label order from model config
            # Default: 0=entailment, 1=neutral, 2=contradiction (MNLI format)
            entailment_prob = float(probs_np[i][0])
            neutral_prob = float(probs_np[i][1])
            contradiction_prob = float(probs_np[i][2])

            # Determine label
            if contradiction_prob > entailment_prob and contradiction_prob > neutral_prob:
                label = NLILabel.CONTRADICTION
            elif entailment_prob > neutral_prob:
                label = NLILabel.ENTAILMENT
            else:
                label = NLILabel.NEUTRAL

            confidence = max(entailment_prob, neutral_prob, contradiction_prob)

            samples.append(NLISample(
                premise=premise,
                hypothesis=hypothesis,
                label=label,
                confidence=confidence,
                entailment_prob=entailment_prob,
                contradiction_prob=contradiction_prob,
                neutral_prob=neutral_prob
            ))

        return samples

    def analyze_single(self, premise: str, hypothesis: str) -> NLISample:
        """Analyze a single premise-hypothesis pair"""
        result = self.analyze([(premise, hypothesis)])
        return result.samples[0] if result.samples else None


class NLIFactory:
    """Factory for creating NLI analyzers with various configurations"""

    @staticmethod
    def create_fast_analyzer(batch_size: int = 32) -> NLIFastAnalyzer:
        """Create a fast NLI analyzer (DeBERTa-v3-xsmall)"""
        return NLIFastAnalyzer(
            model_name="DeBERTa-v3-xsmall",
            batch_size=batch_size
        )

    @staticmethod
    def create_balanced_analyzer(batch_size: int = 16) -> NLIFastAnalyzer:
        """Create a balanced NLI analyzer (DeBERTa-v3-small)"""
        return NLIFastAnalyzer(
            model_name="microsoft/deberta-v3-small",
            batch_size=batch_size
        )

    @staticmethod
    def create_accurate_analyzer(batch_size: int = 8) -> NLIFastAnalyzer:
        """Create an accurate NLI analyzer (DeBERTa-v3-base)"""
        return NLIFastAnalyzer(
            model_name="microsoft/deberta-v3-base",
            batch_size=batch_size
        )
