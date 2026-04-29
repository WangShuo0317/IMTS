"""
Evaluation Report Generator

Generates comprehensive evaluation reports including:
- Overall scores and pass/fail determination
- Radar chart data for capability visualization
- Data optimization suggestions
- Detailed metrics breakdown
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from .config import EVALUATION_CONFIG, DIMENSION_DESCRIPTIONS

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSuggestion:
    """Single optimization suggestion"""
    category: str
    description: str
    priority: str  # "high", "medium", "low"
    affected_samples: int
    examples: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class RadarChartData:
    """Radar chart visualization data"""
    dimensions: List[str]
    scores: List[float]
    max_score: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "scores": self.scores,
            "max_score": self.max_score
        }


class EvaluationReportGenerator:
    """
    Generates comprehensive evaluation reports.

    Produces:
    - Overall score and pass/fail status
    - Radar chart data for capability visualization
    - Data optimization suggestions
    - Detailed metrics breakdown
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or EVALUATION_CONFIG
        self.weights = self.config.get("scoring_weights", {})
        self.pass_threshold = self.config.get("pass_threshold", 75.0)

    def generate_report(
        self,
        fact_check_results: Dict[str, Any],
        logic_check_results: Dict[str, Any],
        sample_results: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            fact_check_results: Results from fact checker agent
            logic_check_results: Results from logic checker agent
            sample_results: Optional per-sample results
            metadata: Optional metadata (job_id, iteration, etc.)

        Returns:
            Complete evaluation report dictionary
        """
        logger.info("Generating evaluation report...")

        # Calculate dimension scores
        dimension_scores = self._calculate_dimension_scores(
            fact_check_results,
            logic_check_results
        )

        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)

        # Determine pass/fail
        passed = overall_score >= self.pass_threshold

        # Generate radar chart data
        radar_data = self._generate_radar_chart_data(dimension_scores)

        # Generate suggestions
        suggestions = self._generate_suggestions(
            fact_check_results,
            logic_check_results,
            sample_results
        )

        # Calculate summary statistics
        summary = self._generate_summary(
            fact_check_results,
            logic_check_results,
            sample_results,
            overall_score
        )

        report = {
            "overall_score": round(overall_score, 2),
            "passed": passed,
            "pass_threshold": self.pass_threshold,
            "radar_data": radar_data.to_dict(),
            "suggestions": [s.to_dict() for s in suggestions],
            "detailed_metrics": dimension_scores,
            "evaluation_summary": summary,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "metadata": metadata or {}
        }

        logger.info(f"Report generated: score={overall_score:.2f}, passed={passed}")
        return report

    def _calculate_dimension_scores(
        self,
        fact_check_results: Dict[str, Any],
        logic_check_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate scores for each dimension (0-100 scale)"""
        dimensions = self.config.get("radar_dimensions", [
            "事实准确性", "逻辑一致性", "完整性", "相关性", "一致性"
        ])

        # Extract component scores
        fact_accuracy = fact_check_results.get("fact_accuracy_score", 0.0) * 100
        logic_consistency = logic_check_results.get("logic_consistency_score", 0.0) * 100

        # Map to dimensions
        scores = {}
        scores["事实准确性"] = fact_accuracy
        scores["逻辑一致性"] = logic_consistency

        # Completeness: derived from fact completeness
        scores["完整性"] = fact_check_results.get("completeness_score", 0.8) * 100

        # Relevance: derived from fact and logic combined
        scores["相关性"] = (
            fact_check_results.get("relevance_score", 0.85) * 0.5 +
            logic_check_results.get("relevance_score", 0.85) * 0.5
        ) * 100

        # Conciseness: derived from logic checker
        scores["一致性"] = logic_check_results.get("conciseness_score", 0.75) * 100

        return scores

    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        total_weight = sum(self.weights.values())
        weighted_sum = 0.0

        for dimension, score in dimension_scores.items():
            weight = self.weights.get(dimension, 0.0)
            weighted_sum += score * weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _generate_radar_chart_data(
        self,
        dimension_scores: Dict[str, float]
    ) -> RadarChartData:
        """Generate radar chart data for visualization"""
        dimensions = self.config.get("radar_dimensions", [
            "事实准确性", "逻辑一致性", "完整性", "相关性", "一致性"
        ])

        scores = [dimension_scores.get(d, 0.0) for d in dimensions]

        return RadarChartData(
            dimensions=dimensions,
            scores=[round(s, 2) for s in scores],
            max_score=100.0
        )

    def _generate_suggestions(
        self,
        fact_check_results: Dict[str, Any],
        logic_check_results: Dict[str, Any],
        sample_results: List[Dict[str, Any]] = None
    ) -> List[EvaluationSuggestion]:
        """Generate data optimization suggestions based on results"""
        suggestions = []

        # Analyze fact check issues
        hallucination_count = fact_check_results.get("hallucination_count", 0)
        missing_facts_count = fact_check_results.get("missing_facts_count", 0)
        total_samples = fact_check_results.get("total_samples", 1)

        if hallucination_count > total_samples * 0.05:  # >5% hallucinations
            suggestions.append(EvaluationSuggestion(
                category="factual_hallucination",
                description="模型存在幻觉问题，在部分回答中产生了与事实不符的内容。建议增加事实核查类训练数据。",
                priority="high",
                affected_samples=hallucination_count,
                examples=fact_check_results.get("hallucination_examples", [])[:3]
            ))

        if missing_facts_count > total_samples * 0.1:  # >10% missing facts
            suggestions.append(EvaluationSuggestion(
                category="factual_gaps",
                description="模型回答存在知识缺口，部分重要事实未被覆盖。建议补充相关领域的训练数据。",
                priority="high",
                affected_samples=missing_facts_count,
                examples=fact_check_results.get("missing_fact_examples", [])[:3]
            ))

        # Analyze logic check issues
        contradiction_count = logic_check_results.get("contradiction_count", 0)
        logic_issues = logic_check_results.get("logic_issues", [])

        if contradiction_count > total_samples * 0.05:  # >5% contradictions
            suggestions.append(EvaluationSuggestion(
                category="logic_contradiction",
                description="模型回答存在逻辑矛盾，前后不一致。建议增加逻辑一致性相关的训练数据。",
                priority="high",
                affected_samples=contradiction_count,
                examples=logic_check_results.get("contradiction_examples", [])[:3]
            ))

        if logic_issues:
            suggestions.append(EvaluationSuggestion(
                category="logic_weakness",
                description="模型在复杂逻辑推理方面存在不足，因果推理能力需加强。",
                priority="medium",
                affected_samples=len(logic_issues),
                examples=logic_issues[:3]
            ))

        # Analyze completeness issues
        low_completeness_count = fact_check_results.get("low_completeness_count", 0)
        if low_completeness_count > total_samples * 0.15:  # >15% incomplete
            suggestions.append(EvaluationSuggestion(
                category="completeness",
                description="模型回答完整性不足，部分问题未得到完整解答。建议增加需要全面回答的训练样本。",
                priority="medium",
                affected_samples=low_completeness_count
            ))

        # Default suggestion if everything looks good
        if not suggestions:
            suggestions.append(EvaluationSuggestion(
                category="general",
                description="模型表现良好，数据质量无需特殊优化。建议保持现有训练流程。",
                priority="low",
                affected_samples=0
            ))

        return suggestions

    def _generate_summary(
        self,
        fact_check_results: Dict[str, Any],
        logic_check_results: Dict[str, Any],
        sample_results: List[Dict[str, Any]] = None,
        overall_score: float = 0.0
    ) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_samples = fact_check_results.get("total_samples", 0)
        passed_samples = fact_check_results.get("passed_samples", total_samples)

        return {
            "total_samples": total_samples,
            "passed_samples": passed_samples,
            "failed_samples": total_samples - passed_samples,
            "pass_rate": round(passed_samples / total_samples * 100, 2) if total_samples > 0 else 0.0,
            "fact_accuracy_avg": fact_check_results.get("avg_fact_accuracy", 0.0),
            "logic_consistency_avg": logic_check_results.get("avg_logic_consistency", 0.0),
            "contradiction_count": logic_check_results.get("contradiction_count", 0),
            "hallucination_count": fact_check_results.get("hallucination_count", 0),
        }

    def generate_sample_report(self, sample: Dict[str, Any]) -> str:
        """Generate a human-readable report for a single sample"""
        fact_score = sample.get("fact_accuracy", 0.0) * 100
        logic_score = sample.get("logic_consistency", 0.0) * 100
        overall = sample.get("overall_score", 0.0)

        return f"""Sample Evaluation Report
=======================
Question: {sample.get('question', 'N/A')[:100]}...
Model Response: {sample.get('response', 'N/A')[:100]}...

Scores:
- 事实准确性: {fact_score:.1f}/100
- 逻辑一致性: {logic_score:.1f}/100
- Overall: {overall:.1f}/100

Issues Found: {', '.join(sample.get('issues', ['None']))}
"""


def generate_evaluation_report(
    fact_check_results: Dict[str, Any],
    logic_check_results: Dict[str, Any],
    sample_results: List[Dict[str, Any]] = None,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate evaluation report.

    Args:
        fact_check_results: Results from fact checker
        logic_check_results: Results from logic checker
        sample_results: Optional per-sample results
        metadata: Optional metadata

    Returns:
        Complete evaluation report
    """
    generator = EvaluationReportGenerator()
    return generator.generate_report(
        fact_check_results,
        logic_check_results,
        sample_results,
        metadata
    )
