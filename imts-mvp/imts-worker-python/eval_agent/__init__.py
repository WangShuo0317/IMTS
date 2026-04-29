"""
AutoGen Evaluation Agent Package

This package provides multi-agent evaluation capabilities using Microsoft AutoGen:
- Fact Checker: Evaluates factual accuracy using RAG
- Logic Checker: Evaluates logical consistency using NLI + LLM
- Arbiter: Generates final scores, radar chart data, and optimization suggestions
"""

from .autogen_eval import run_autogen_evaluation
from .config import EVALUATION_CONFIG

__all__ = ["run_autogen_evaluation", "EVALUATION_CONFIG"]
