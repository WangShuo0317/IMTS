"""
DataDeduplicator Tool - Enhanced Deduplication
Agent Skills 规范实现

去重（Deduplication）：
- 精确去重 (Exact Deduplication)
- 语义去重 (Semantic Deduplication using Embeddings)
"""

import asyncio
import os
import pandas as pd
from typing import Dict, List, Any, Tuple, Union
from langchain_core.tools import tool

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from embedding_service import DataSample, Deduplicator, EmbeddingService, state_to_samples
from data_opt_agent.skills.state_utils import parse_state


def _parse(state: Union[dict, str]) -> dict:
    return parse_state(state)


@tool
async def exact_deduplicate(state: Union[dict, str]) -> dict:
    """Remove exact duplicates from dataset.

    Uses MD5 hash of normalized text to identify duplicates.

    Returns:
        dict with original_count, duplicates_removed, unique_samples
    """
    state = _parse(state)
    samples, dataset_path = state_to_samples(state)

    if not samples:
        return {"status": "no_data"}

    deduplicator = Deduplicator()
    unique_samples, removed = deduplicator.exact_deduplicate(samples)

    # Save unique samples
    output_path = dataset_path.replace('.csv', '_deduplicated.csv').replace('.json', '_deduplicated.json')

    unique_data = [{"id": s.id, "question": s.question, "answer": s.answer} for s in unique_samples]
    output_df = pd.DataFrame(unique_data)

    if output_path.endswith('.csv'):
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_json(output_path, orient='records', force_ascii=False)

    return {
        "status": "success",
        "original_count": len(samples),
        "duplicates_removed": len(removed),
        "unique_samples": len(unique_samples),
        "duplicate_ratio": round(len(removed) / len(samples), 4) if samples else 0,
        "removed_samples": removed[:20],  # Limit output
        "output_path": output_path,
        "dataset_path": output_path,
        "path": output_path
    }


@tool
async def semantic_deduplicate(state: Union[dict, str], similarity_threshold: float = 0.95) -> dict:
    """Remove semantic duplicates using embedding similarity.

    Uses embedding model to compute semantic similarity between samples.
    Samples with similarity >= threshold are considered duplicates.

    Args:
        state: State dict with dataset_path
        similarity_threshold: Cosine similarity threshold (0.0-1.0), default 0.95

    Returns:
        dict with original_count, semantic_duplicates_removed, unique_samples
    """
    state = _parse(state)
    # Get embedding config from environment
    embedding_url = os.getenv('EMBEDDING_BASE_URL', 'http://10.242.33.21:8002')
    embedding_model = os.getenv('EMBEDDING_MODEL_NAME', 'Qwen3-embeddings')

    samples, dataset_path = state_to_samples(state)

    if not samples:
        return {"status": "no_data", "message": "No dataset found"}

    # Initialize embedding service
    embedding_service = EmbeddingService(embedding_url, embedding_model)

    deduplicator = Deduplicator(embedding_service)

    # Note: This is a simplified version. For production, use batch processing
    # and approximate nearest neighbors (e.g., FAISS) for efficiency

    print(f"Getting embeddings for {len(samples)} samples...")
    texts = [f"{s.question} {s.answer}" for s in samples]
    embeddings = await embedding_service.get_embeddings_batch(texts)

    for s, emb in zip(samples, embeddings):
        s.embedding = emb

    print("Computing similarity and deduplicating...")
    unique_samples, removed = await deduplicator.semantic_deduplicate(samples, threshold=similarity_threshold)

    # Save unique samples
    output_path = dataset_path.replace('.csv', '_semantic_dedup.csv').replace('.json', '_semantic_dedup.json')

    unique_data = [{"id": s.id, "question": s.question, "answer": s.answer} for s in unique_samples]
    output_df = pd.DataFrame(unique_data)

    if output_path.endswith('.csv'):
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_json(output_path, orient='records', force_ascii=False)

    return {
        "status": "success",
        "original_count": len(samples),
        "semantic_duplicates_removed": len(removed),
        "unique_samples": len(unique_samples),
        "duplicate_ratio": round(len(removed) / len(samples), 4) if samples else 0,
        "similarity_threshold": similarity_threshold,
        "removed_samples": removed[:20],
        "output_path": output_path
    }


@tool
async def full_deduplicate(state: Union[dict, str], semantic_threshold: float = 0.95) -> dict:
    """Perform both exact and semantic deduplication.

    First removes exact duplicates, then semantic duplicates.

    Args:
        state: State dict with dataset_path
        semantic_threshold: Similarity threshold for semantic dedup

    Returns:
        dict with full deduplication report
    """
    state = _parse(state)
    # Get embedding config
    embedding_url = os.getenv('EMBEDDING_BASE_URL', 'http://10.242.33.21:8002')
    embedding_model = os.getenv('EMBEDDING_MODEL_NAME', 'Qwen3-embeddings')

    samples, dataset_path = state_to_samples(state)

    if not samples:
        return {"status": "no_data"}

    original_count = len(samples)

    # Step 1: Exact deduplication
    deduplicator = Deduplicator()
    samples, exact_removed = deduplicator.exact_deduplicate(samples)

    # Step 2: Semantic deduplication
    embedding_service = EmbeddingService(embedding_url, embedding_model)

    texts = [f"{s.question} {s.answer}" for s in samples]
    embeddings = await embedding_service.get_embeddings_batch(texts)

    for s, emb in zip(samples, embeddings):
        s.embedding = emb

    samples, semantic_removed = await deduplicator.semantic_deduplicate(samples, threshold=semantic_threshold)

    # Save final result
    output_path = dataset_path.replace('.csv', '_fully_dedup.csv').replace('.json', '_fully_dedup.json')

    final_data = [{"id": s.id, "question": s.question, "answer": s.answer} for s in samples]
    output_df = pd.DataFrame(final_data)

    if output_path.endswith('.csv'):
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_json(output_path, orient='records', force_ascii=False)

    total_removed = len(exact_removed) + len(semantic_removed)

    return {
        "status": "success",
        "original_count": original_count,
        "exact_duplicates_removed": len(exact_removed),
        "semantic_duplicates_removed": len(semantic_removed),
        "total_removed": total_removed,
        "final_unique_count": len(samples),
        "retention_rate": round(len(samples) / original_count, 4) if original_count else 0,
        "output_path": output_path,
        "steps": [
            f"Removed {len(exact_removed)} exact duplicates",
            f"Removed {len(semantic_removed)} semantic duplicates (threshold={semantic_threshold})"
        ]
    }
