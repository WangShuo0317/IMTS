"""
Evaluation Configuration

Configuration settings for the AutoGen evaluation system.
Includes environment variable loading for LLM configuration.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in the project root (imts-worker-python directory)
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    # Fallback: also try data_opt_agent's .env
    alt_env_path = os.path.join(os.path.dirname(__file__), "..", "data_opt_agent", ".env")
    if os.path.exists(alt_env_path):
        load_dotenv(alt_env_path)

EVALUATION_CONFIG = {
    # NLI batch processing
    "batch_size": 32,  # NLI model batch size

    # Parallel agent settings
    "parallel_agents": 2,  # Number of parallel evaluation agents

    # Timeout settings
    "timeout_per_sample": 30,  # seconds per sample

    # NLI model - using small model for speed
    "nli_model": "cross-encoder/nli-deberta-v3-xsmall",  # Lightweight NLI model

    # RAG settings
    "rag_top_k": 5,  # Number of documents to retrieve
    "rag_embedding_model": "all-MiniLM-L6-v2",  # Fast embedding model

    # Cache settings
    "cache_enabled": True,
    "cache_ttl": 3600,  # Cache TTL in seconds

    # Scoring weights for arbiter
    "scoring_weights": {
        "fact_accuracy": 0.35,  # 35% weight
        "logic_consistency": 0.25,  # 25% weight
        "completeness": 0.15,  # 15% weight
        "relevance": 0.15,  # 15% weight
        "conciseness": 0.10,  # 10% weight
    },

    # Pass threshold
    "pass_threshold": 75.0,  # Minimum score to pass

    # Radar chart dimensions
    "radar_dimensions": [
        "事实准确性",  # Fact Accuracy
        "逻辑一致性",  # Logic Consistency
        "完整性",      # Completeness
        "相关性",      # Relevance
        "一致性",      # Conciseness
    ],

    # External knowledge base settings
    "use_external_kb": True,
    "external_kb_type": "wikipedia",  # or "local" for local KB

    # Performance settings
    "max_workers": 4,  # Maximum concurrent workers for batch processing
    "streaming_enabled": True,  # Enable streaming for LLM calls
}

# LLM Configuration (from environment variables)
LLM_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "model_name": os.getenv("LLM_MODEL_NAME", "gpt-4"),
}

# Dimension descriptions for report generation
DIMENSION_DESCRIPTIONS = {
    "事实准确性": "模型回答中事实的正确程度，与标准答案的吻合度",
    "逻辑一致性": "模型回答中的逻辑连贯性和一致性程度",
    "完整性": "模型回答是否完整覆盖了问题要求的各个方面",
    "相关性": "模型回答与问题意图的相关程度",
    "一致性": "模型回答的简洁性和表达清晰度",
}
