"""
Hybrid RAG Knowledge Base

Provides retrieval-augmented generation using both training data and external knowledge bases.
- Source 1: Training dataset (CSV/JSON loaded and indexed)
- Source 2: External knowledge (Wikipedia API / local knowledge base)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result"""
    content: str
    source: str  # "training_data" or "external_kb"
    score: float
    metadata: Dict[str, Any]


class HybridRAGKnowledgeBase:
    """
    Hybrid RAG system combining training data and external knowledge base.
    Supports both local training data and Wikipedia-based external knowledge.
    """

    def __init__(
        self,
        dataset_path: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_external_kb: bool = True,
        external_kb_type: str = "wikipedia",
        top_k: int = 5
    ):
        self.dataset_path = dataset_path
        self.embedding_model_name = embedding_model
        self.use_external_kb = use_external_kb
        self.external_kb_type = external_kb_type
        self.top_k = top_k

        self.embedding_model = None
        self.train_index = None
        self.external_index = None
        self._initialized = False

    def initialize(self):
        """Initialize the RAG system"""
        if self._initialized:
            return

        logger.info("Initializing Hybrid RAG Knowledge Base...")

        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Build training data index
        if self.dataset_path and os.path.exists(self.dataset_path):
            self._build_training_index()
        else:
            logger.warning(f"Dataset path not found: {self.dataset_path}, skipping training index")

        # Build external KB index
        if self.use_external_kb:
            self._build_external_index()

        self._initialized = True
        logger.info("Hybrid RAG Knowledge Base initialized successfully")

    def _build_training_index(self):
        """Build FAISS index from training data"""
        logger.info(f"Building training data index from: {self.dataset_path}")

        try:
            # Load dataset based on file extension
            if self.dataset_path.endswith('.csv'):
                df = pd.read_csv(self.dataset_path)
            elif self.dataset_path.endswith('.json'):
                df = pd.read_json(self.dataset_path)
            else:
                logger.warning(f"Unsupported dataset format: {self.dataset_path}")
                return

            # Convert to documents
            documents = []
            for idx, row in df.iterrows():
                # Create document content from row
                content = self._row_to_content(row)
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": "training_data",
                            "row_id": idx,
                            "dataset": self.dataset_path
                        }
                    )
                    documents.append(doc)

            if documents:
                # Create embeddings and build FAISS index
                texts = [doc.page_content for doc in documents]
                embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

                self.train_index = FAISS.from_embeddings(
                    text_embeddings=list(zip(texts, embeddings)),
                    embedding=self.embedding_model,
                    metadatas=[doc.metadata for doc in documents]
                )
                logger.info(f"Training data index built with {len(documents)} documents")
            else:
                logger.warning("No documents generated from training data")

        except Exception as e:
            logger.error(f"Failed to build training index: {e}")
            self.train_index = None

    def _row_to_content(self, row: pd.Series) -> str:
        """Convert a dataframe row to document content"""
        content_parts = []

        # Try to extract question/answer pairs
        for col in ['question', 'input', 'prompt', 'instruction']:
            if col in row and pd.notna(row[col]):
                content_parts.append(f"Q: {row[col]}")

        for col in ['answer', 'output', 'response', 'target']:
            if col in row and pd.notna(row[col]):
                content_parts.append(f"A: {row[col]}")

        # If no standard columns, use all string columns
        if not content_parts:
            for col, val in row.items():
                if pd.notna(val) and isinstance(val, str):
                    content_parts.append(f"{col}: {val}")

        return " | ".join(content_parts[:4])  # Limit to first 4 parts

    def _build_external_index(self):
        """Build FAISS index from external knowledge base"""
        logger.info(f"Building external KB index (type: {self.external_kb_type})")

        if self.external_kb_type == "wikipedia":
            self._build_wikipedia_index()
        elif self.external_kb_type == "local":
            self._build_local_kb_index()
        else:
            logger.warning(f"Unknown external KB type: {self.external_kb_type}")

    def _build_wikipedia_index(self):
        """
        Build index from Wikipedia using wikipedia-api.
        Note: For production, consider using a pre-indexed Wikipedia dump.
        """
        try:
            # For now, create an empty index - will be populated lazily
            # In production, load from a pre-built Wikipedia FAISS index
            logger.info("Wikipedia index placeholder created")
            self.external_index = None
        except Exception as e:
            logger.error(f"Failed to build Wikipedia index: {e}")
            self.external_index = None

    def _build_local_kb_index(self):
        """Build index from local knowledge base files"""
        local_kb_path = os.path.join(os.path.dirname(self.dataset_path) if self.dataset_path else ".", "knowledge_base")

        if not os.path.exists(local_kb_path):
            logger.info(f"Local KB path not found: {local_kb_path}")
            return

        try:
            documents = []
            for filename in os.listdir(local_kb_path):
                if filename.endswith(('.txt', '.md', '.json')):
                    filepath = os.path.join(local_kb_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        doc = Document(
                            page_content=content,
                            metadata={"source": "local_kb", "file": filename}
                        )
                        documents.append(doc)

            if documents:
                texts = [doc.page_content for doc in documents]
                embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

                self.external_index = FAISS.from_embeddings(
                    text_embeddings=list(zip(texts, embeddings)),
                    embedding=self.embedding_model,
                    metadatas=[doc.metadata for doc in documents]
                )
                logger.info(f"Local KB index built with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to build local KB index: {e}")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_source: str = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The query string
            top_k: Number of results to return (default: self.top_k)
            filter_source: Filter by source ("training_data" or "external_kb")

        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        if not self._initialized:
            self.initialize()

        top_k = top_k or self.top_k
        results = []

        # Retrieve from training data
        if self.train_index and (filter_source is None or filter_source == "training_data"):
            try:
                train_results = self.train_index.similarity_search_with_score(query, k=top_k)
                for doc, score in train_results:
                    results.append(RetrievalResult(
                        content=doc.page_content,
                        source="training_data",
                        score=float(score),
                        metadata=doc.metadata
                    ))
            except Exception as e:
                logger.error(f"Training data retrieval failed: {e}")

        # Retrieve from external KB
        if self.external_index and (filter_source is None or filter_source == "external_kb"):
            try:
                external_results = self.external_index.similarity_search_with_score(query, k=top_k)
                for doc, score in external_results:
                    results.append(RetrievalResult(
                        content=doc.page_content,
                        source="external_kb",
                        score=float(score),
                        metadata=doc.metadata
                    ))
            except Exception as e:
                logger.error(f"External KB retrieval failed: {e}")

        # Sort by score and deduplicate
        results = sorted(results, key=lambda x: x.score)

        # Remove duplicates by content
        seen = set()
        unique_results = []
        for r in results:
            if r.content not in seen:
                seen.add(r.content)
                unique_results.append(r)

        return unique_results[:top_k]

    def add_document(self, content: str, source: str = "training_data", metadata: Dict = None):
        """Add a single document to the index"""
        if not self._initialized:
            self.initialize()

        doc = Document(
            page_content=content,
            metadata=metadata or {"source": source}
        )

        if source == "training_data" and self.train_index:
            self.train_index.add_documents([doc])
        elif source == "external_kb" and self.external_index:
            self.external_index.add_documents([doc])

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the knowledge base"""
        return {
            "training_docs": self.train_index.index.ntotal if self.train_index else 0,
            "external_docs": self.external_index.index.ntotal if self.external_index else 0,
            "embedding_model": self.embedding_model_name,
        }
