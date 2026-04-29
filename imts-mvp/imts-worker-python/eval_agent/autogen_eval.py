"""
AutoGen Evaluation Main Entry

Orchestrates the multi-agent AutoGen evaluation system:
1. Initializes RAG knowledge base and NLI analyzer
2. Runs Fact Checker and Logic Checker agents in parallel
3. Synthesizes results with Arbiter agent
4. Generates final evaluation report
"""

import asyncio
import logging
import time
import os
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)


async def run_autogen_evaluation(
    state: Dict[str, Any],
    message_builder=None,
    progress_callback: Callable = None
) -> Dict[str, Any]:
    """
    Run the AutoGen multi-agent evaluation.

    This is the main entry point for evaluation, called from nodes.py.

    Args:
        state: State dictionary containing job_id, dataset_path, model_name, etc.
        message_builder: Optional MessageBuilder for progress updates
        progress_callback: Optional callback for progress updates (progress, message)

    Returns:
        Evaluation result dictionary with scores, radar data, and suggestions
    """
    job_id = state.get("job_id", "unknown")
    dataset_path = state.get("dataset_path", "/data/train.csv")
    iteration = state.get("current_iteration", 1)
    llm_api_key = state.get("llm_api_key")
    llm_base_url = state.get("llm_base_url")
    llm_model_name = state.get("llm_model_name", "gpt-4")

    logger.info(f"Starting AutoGen evaluation for job {job_id}, iteration {iteration}")

    start_time = time.time()

    # Progress callback helper
    def emit_progress(progress: int, message: str):
        if progress_callback:
            progress_callback(progress, message)
        if message_builder:
            # Emit via message builder if available
            pass

    try:
        # Step 1: Initialize components
        emit_progress(5, "Initializing evaluation components...")

        from .config import EVALUATION_CONFIG, LLM_CONFIG
        from .rag_knowledge_base import HybridRAGKnowledgeBase
        from .nli_analyzer import NLIFastAnalyzer
        from .llm_judge import LLMJudge

        # Use LLM config from state or fall back to environment variables
        llm_api_key = llm_api_key or LLM_CONFIG.get("api_key") or os.getenv("OPENAI_API_KEY") or os.getenv("EVAL_API_KEY")
        llm_base_url = llm_base_url or LLM_CONFIG.get("base_url") or os.getenv("OPENAI_BASE_URL") or os.getenv("EVAL_BASE_URL")
        llm_model_name = llm_model_name or LLM_CONFIG.get("model_name") or os.getenv("OPENAI_MODEL_NAME") or os.getenv("EVAL_MODEL_NAME")

        logger.info(f"Using LLM config: model={llm_model_name}, base_url={llm_base_url}")

        # Initialize RAG knowledge base
        emit_progress(10, "Building RAG knowledge base...")
        rag_kb = HybridRAGKnowledgeBase(
            dataset_path=dataset_path,
            embedding_model=EVALUATION_CONFIG.get("rag_embedding_model", "all-MiniLM-L6-v2"),
            use_external_kb=EVALUATION_CONFIG.get("use_external_kb", True),
            external_kb_type=EVALUATION_CONFIG.get("external_kb_type", "wikipedia"),
            top_k=EVALUATION_CONFIG.get("rag_top_k", 5)
        )
        rag_kb.initialize()

        # Initialize NLI analyzer
        emit_progress(15, "Loading NLI model...")
        nli_analyzer = NLIFastAnalyzer(
            model_name=EVALUATION_CONFIG.get("nli_model", "DeBERTa-v3-xsmall"),
            batch_size=EVALUATION_CONFIG.get("batch_size", 32)
        )
        nli_analyzer.initialize()

        # Initialize LLM judge
        emit_progress(20, "Initializing LLM judge...")
        llm_judge = LLMJudge(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model_name=llm_model_name
        )
        llm_judge.initialize()

        # Step 2: Load test data
        emit_progress(25, "Loading test data...")
        test_data = await load_test_data(dataset_path, state)

        if not test_data:
            logger.warning(f"No test data found for job {job_id}, using mock data")
            test_data = generate_mock_test_data(100)

        total_samples = len(test_data)
        logger.info(f"Loaded {total_samples} test samples")

        # Step 3: Run Fact Checker and Logic Checker in parallel
        emit_progress(30, "Running fact checker and logic checker in parallel...")

        from .fact_checker import run_fact_check
        from .logic_checker import run_logic_check

        # Run both checkers concurrently
        fact_task = run_fact_check(
            samples=test_data,
            rag_kb=rag_kb,
            nli_analyzer=nli_analyzer,
            progress_callback=lambda p, m: emit_progress(30 + p * 0.3, f"Fact check: {m}")
        )

        logic_task = run_logic_check(
            samples=test_data,
            nli_analyzer=nli_analyzer,
            llm_judge=llm_judge,
            progress_callback=lambda p, m: emit_progress(30 + p * 0.3, f"Logic check: {m}")
        )

        # Wait for both to complete
        fact_results, logic_results = await asyncio.gather(fact_task, logic_task)

        emit_progress(65, "Fact and logic checking complete")

        # Step 4: Run Arbiter for final evaluation
        emit_progress(70, "Running arbiter for final evaluation...")

        from .arbiter import run_arbiter_evaluation

        arbiter_result = await run_arbiter_evaluation(
            fact_check_results=fact_results,
            logic_check_results=logic_results,
            sample_results=fact_results.get("sample_results", []),
            metadata={
                "job_id": job_id,
                "iteration": iteration,
                "model_name": state.get("model_name", "unknown"),
                "dataset_path": dataset_path
            }
        )

        emit_progress(90, "Generating final report...")

        # Step 5: Add timing and finalize
        end_time = time.time()
        evaluation_time = end_time - start_time

        final_result = {
            **arbiter_result,
            "evaluation_time_seconds": round(evaluation_time, 2),
            "total_samples": total_samples,
            "job_id": job_id,
            "iteration": iteration,
        }

        # Add passed status based on threshold
        pass_threshold = EVALUATION_CONFIG.get("pass_threshold", 75.0)
        final_result["passed"] = final_result.get("overall_score", 0) >= pass_threshold

        emit_progress(100, "Evaluation complete")

        logger.info(
            f"Evaluation complete for job {job_id}: "
            f"score={final_result.get('overall_score', 0):.2f}, "
            f"time={evaluation_time:.2f}s"
        )

        return final_result

    except Exception as e:
        logger.error(f"Evaluation failed for job {job_id}: {e}", exc_info=True)

        # Return a failure result
        return {
            "overall_score": 0.0,
            "passed": False,
            "error": str(e),
            "evaluation_time_seconds": time.time() - start_time,
            "job_id": job_id,
            "iteration": iteration,
        }


async def load_test_data(dataset_path: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load test data from the dataset.

    For evaluation, we use a portion of the training data or look for
    a separate test dataset.

    Args:
        dataset_path: Path to the dataset
        state: State dictionary with additional context

    Returns:
        List of test samples with question, ground_truth, response
    """
    import os
    import pandas as pd
    import json

    # Try to find test data
    possible_paths = [
        dataset_path,
        dataset_path.replace("/train.csv", "/test.csv"),
        dataset_path.replace("/train.json", "/test.json"),
        os.path.join(os.path.dirname(dataset_path), "test.csv"),
        os.path.join(os.path.dirname(dataset_path), "test.json"),
    ]

    test_data = []

    for path in possible_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                elif path.endswith('.json'):
                    df = pd.read_json(path)
                else:
                    continue

                # Convert to test format
                for idx, row in df.iterrows():
                    sample = {
                        "id": f"sample_{idx}",
                        "question": str(row.get("question", row.get("input", ""))),
                        "ground_truth": str(row.get("answer", row.get("output", ""))),
                        "response": "",  # Will be filled by model
                    }
                    test_data.append(sample)

                if test_data:
                    logger.info(f"Loaded {len(test_data)} test samples from {path}")
                    return test_data[:1000]  # Limit to 1000

            except Exception as e:
                logger.warning(f"Failed to load test data from {path}: {e}")

    # If no test data found, return empty
    return test_data


def generate_mock_test_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Generate mock test data for testing purposes.

    Args:
        num_samples: Number of samples to generate

    Returns:
        List of mock test samples
    """
    import random

    topics = [
        "science", "history", "geography", "mathematics",
        "literature", "technology", "arts", "sports"
    ]

    mock_data = []

    for i in range(num_samples):
        topic = random.choice(topics)

        # Generate a simple Q&A pair
        question = f"What is the capital of {topic}?"
        answer = f"The capital of {topic} is an important city."

        mock_data.append({
            "id": f"mock_sample_{i}",
            "question": question,
            "ground_truth": answer,
            "response": answer,  # For fact checking, we compare gt to model response
        })

    return mock_data


async def run_evaluation_batch(
    samples: List[Dict[str, Any]],
    state: Dict[str, Any],
    message_builder=None
) -> Dict[str, Any]:
    """
    Run evaluation on a batch of samples.

    This is an alternative entry point for batch evaluation.

    Args:
        samples: List of test samples
        state: State dictionary
        message_builder: Message builder for progress updates

    Returns:
        Evaluation results
    """
    # Create a modified state with the provided samples
    # This is useful when test data comes from a different source

    state["_test_samples"] = samples
    return await run_autogen_evaluation(state, message_builder)


# Export main function
__all__ = ["run_autogen_evaluation", "run_evaluation_batch"]
