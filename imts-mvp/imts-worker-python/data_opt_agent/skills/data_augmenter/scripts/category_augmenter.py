import asyncio
import logging
import pandas as pd
from typing import Union, List
from langchain_core.tools import tool

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
from embedding_clustering import cluster_and_label_dataset

logger = logging.getLogger(__name__)

@tool
async def cluster_and_find_weak_categories(dataset_path: str) -> dict:
    """Cluster the dataset, assign semantic categories using LLM, and identify scarce categories.

    Args:
        dataset_path: Local path to the dataset

    Returns:
        dict with categorized_dataset_path, category_stats, and scarce_categories
    """
    logger.info("Starting embedding and clustering pipeline...")

    # cluster_and_label_dataset now uses DashScope internally for labeling
    new_path, stats = await cluster_and_label_dataset(dataset_path)

    # Identify scarce categories (bottom 10% or fewer than threshold)
    total_samples = sum(stats.values())
    scarce_categories = [cat for cat, count in stats.items()
                         if count < (total_samples * 0.1) and cat != "未分类/杂项"]

    return {
        "status": "success",
        "categorized_dataset_path": new_path,
        "category_distribution": stats,
        "scarce_categories_needing_augmentation": scarce_categories
    }