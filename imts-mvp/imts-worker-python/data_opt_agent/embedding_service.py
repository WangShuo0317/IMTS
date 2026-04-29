"""
Data Optimization Core Module

Provides core functionality for:
1. Distribution Analysis
2. Anomaly Detection
3. Metadata Extraction
4. Deduplication (Exact & Semantic)
5. Denoising & PII Masking
6. Data Augmentation (Rewriting, Reverse Translation, Synthetic Generation)
"""

import asyncio
import json
import re
import hashlib
import html
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """Represents a single data sample with metadata."""
    id: str
    question: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class EmbeddingService:
    """Service for generating embeddings using remote API or local model."""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self._client = None
        self._local_model = None

    async def initialize(self):
        """Initialize the OpenAI-compatible client."""
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key="not-needed",
            base_url=self.base_url,
            timeout=60
        )

    def _get_local_model(self):
        """Get local sentence-transformers model as fallback."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight Chinese-compatible model as fallback
            # Note: This outputs 384-dim, but remote model outputs 1024-dim
            self._local_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return self._local_model

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        if not self._client:
            await self.initialize()

        # Try remote API first
        try:
            response = await self._client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Remote embedding failed: {e}, trying local model")

        # Fallback to local sentence-transformers
        try:
            local_model = self._get_local_model()
            embedding = local_model.encode(text, convert_to_numpy=True)
            # Pad to match remote dimension (1024)
            embedding_padded = embedding.tolist() + [0.0] * (1024 - len(embedding))
            return embedding_padded
        except Exception as e2:
            logger.error(f"Local embedding also failed: {e2}")
            return [0.0] * 1024  # Remote model outputs 1024-dim

    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Get embeddings for multiple texts in batches."""
        if not self._client:
            await self.initialize()

        all_embeddings = []

        # Try remote API first
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = await self._client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                all_embeddings.extend([d.embedding for d in response.data])
            return all_embeddings
        except Exception as e:
            logger.warning(f"Remote batch embedding failed: {e}, using local model")

        # Fallback to local model
        try:
            local_model = self._get_local_model()
            all_embeddings = local_model.encode(texts, convert_to_numpy=True, batch_size=batch_size)
            # Pad to match remote dimension (1024)
            return [emb.tolist() + [0.0] * (1024 - len(emb)) for emb in all_embeddings]
        except Exception as e2:
            logger.error(f"Local batch embedding also failed: {e2}")
            return [[0.0] * 1024 for _ in texts]


class DistributionAnalyzer:
    """Analyzes data distribution: semantic, length, language."""

    @staticmethod
    def analyze_length_distribution(samples: List[DataSample]) -> Dict[str, Any]:
        """Analyze text length distribution."""
        lengths = []
        for s in samples:
            q_len = len(s.question)
            a_len = len(s.answer)
            lengths.append({"question_len": q_len, "answer_len": a_len, "total_len": q_len + a_len})

        if not lengths:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0}

        total_lengths = [l["total_len"] for l in lengths]
        total_lengths.sort()

        n = len(total_lengths)
        return {
            "mean": sum(total_lengths) / n,
            "std": (sum((x - sum(total_lengths)/n)**2 for x in total_lengths) / n) ** 0.5,
            "min": total_lengths[0],
            "max": total_lengths[-1],
            "p25": total_lengths[n // 4],
            "p50": total_lengths[n // 2],
            "p75": total_lengths[3 * n // 4],
            "p95": total_lengths[int(n * 0.95)]
        }

    @staticmethod
    def analyze_semantic_distribution(samples: List[DataSample]) -> Dict[str, Any]:
        """Analyze semantic/topical distribution using embeddings."""
        if not samples or not samples[0].embedding:
            return {"topics": {}, "diversity_score": 0}

        # Cluster embeddings to identify topics
        # Simple approach: use cosine similarity to group
        embeddings = [s.embedding for s in samples]
        num_samples = len(embeddings)

        # Calculate pairwise similarities (simplified)
        # For large datasets, use approximate methods
        if num_samples > 100:
            # Sample for efficiency
            sample_size = 100
            indices = list(range(0, num_samples, num_samples // sample_size))
            sample_embeddings = [embeddings[i] for i in indices[:sample_size]]
        else:
            sample_embeddings = embeddings

        # Calculate centroid diversity
        centroid = [sum(x) / len(sample_embeddings) for x in zip(*sample_embeddings)]

        # Calculate average distance to centroid
        import math
        distances = []
        for emb in sample_embeddings:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(emb, centroid)))
            distances.append(dist)

        diversity_score = sum(distances) / len(distances) if distances else 0

        return {
            "diversity_score": round(diversity_score, 4),
            "num_samples": num_samples,
            "embedding_dim": len(embeddings[0]) if embeddings else 0
        }

    @staticmethod
    def extract_topics(texts: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """Extract common topics/keywords from texts."""
        # Simple keyword extraction
        stopwords = {"的", "是", "在", "了", "和", "与", "或", "为", "等", "于", "上", "下", "中", "从", "到", "对", "这", "那", "有", "没有"}

        all_words = []
        for text in texts:
            # Simple Chinese word segmentation (character-based for simplicity)
            words = re.findall(r'[\w]+', text)
            all_words.extend([w for w in words if w not in stopwords and len(w) > 1])

        counter = Counter(all_words)
        return counter.most_common(top_n)


class AnomalyDetector:
    """Detects anomalies: outliers, format errors, corrupted data."""

    @staticmethod
    def detect_format_errors(samples: List[DataSample]) -> List[Dict[str, Any]]:
        """Detect format errors in samples."""
        errors = []

        for i, s in enumerate(samples):
            issues = []

            # Check for empty fields
            if not s.question.strip():
                issues.append("empty_question")
            if not s.answer.strip():
                issues.append("empty_answer")

            # Check for HTML tags
            if re.search(r'<[^>]+>', s.question) or re.search(r'<[^>]+>', s.answer):
                issues.append("contains_html")

            # Check for excessive length
            if len(s.question) > 5000:
                issues.append("question_too_long")
            if len(s.answer) > 10000:
                issues.append("answer_too_long")

            # Check for garbled text (high ratio of non-printable chars)
            non_printable_q = sum(1 for c in s.question if not c.isprintable())
            non_printable_a = sum(1 for c in s.answer if not c.isprintable())

            if len(s.question) > 0 and non_printable_q / len(s.question) > 0.1:
                issues.append("garbled_question")
            if len(s.answer) > 0 and non_printable_a / len(s.answer) > 0.1:
                issues.append("garbled_answer")

            # Check for question mark consistency
            if '?' in s.question and not s.answer.endswith(('?', '！', '。', '.')):
                issues.append("inconsistent_ending")

            if issues:
                errors.append({
                    "sample_id": s.id,
                    "index": i,
                    "issues": issues,
                    "severity": "high" if any(i in ["empty_question", "empty_answer", "garbled_question", "garbled_answer"] for i in issues) else "medium"
                })

        return errors

    @staticmethod
    def detect_outliers_by_length(samples: List[DataSample], threshold: float = 3.0) -> List[Dict[str, Any]]:
        """Detect outliers based on length statistics."""
        if not samples:
            return []

        lengths = [(i, len(s.question) + len(s.answer)) for i, s in enumerate(samples)]
        lengths.sort(key=lambda x: x[1])

        n = len(lengths)
        q1_idx = n // 4
        q3_idx = 3 * n // 4

        q1_len = lengths[q1_idx][1]
        q3_len = lengths[q3_idx][1]
        iqr = q3_len - q1_len

        lower_bound = q1_len - threshold * iqr
        upper_bound = q3_len + threshold * iqr

        outliers = []
        for idx, length in lengths:
            if length < lower_bound or length > upper_bound:
                outliers.append({
                    "sample_id": samples[idx].id,
                    "index": idx,
                    "length": length,
                    "bound": (lower_bound, upper_bound)
                })

        return outliers


class MetadataExtractor:
    """Extracts metadata: topics, difficulty, sentiment."""

    @staticmethod
    def extract_difficulty(question: str, answer: str) -> str:
        """Estimate difficulty level based on features."""
        # Simple heuristic-based difficulty detection
        score = 0

        # Length-based
        if len(question) > 100:
            score += 1
        if len(answer) > 300:
            score += 1

        # Complexity indicators
        complex_patterns = [
            r'为什么', r'如何', r'比较', r'分析', r'解释',
            r'difference', r'compare', r'analyze', r'explain',
            r'为什么|如何|比较'
        ]
        for pattern in complex_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                score += 1
                break

        # Technical terms
        tech_terms = len(re.findall(r'\b\d+\b', question + answer))
        if tech_terms > 5:
            score += 1

        if score <= 1:
            return "easy"
        elif score <= 3:
            return "medium"
        else:
            return "hard"

    @staticmethod
    def extract_sentiment(text: str) -> str:
        """Simple sentiment analysis."""
        positive_words = ["好", "优秀", "棒", "赞", "喜欢", "满意", "正确", "是"]
        negative_words = ["坏", "差", "错误", "不对", "不好", "不满意", "否"]

        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    @staticmethod
    def extract_domain_keywords(text: str) -> List[str]:
        """Extract domain-specific keywords."""
        domain_keywords = {
            "计算机": ["CPU", "内存", "进程", "线程", "操作系统", "算法", "数据结构", "网络", "数据库"],
            "数学": ["函数", "方程", "微分", "积分", "矩阵", "向量", "概率", "统计"],
            "物理": ["力", "能量", "速度", "加速度", "电磁", "量子", "热力学"],
            "化学": ["分子", "原子", "反应", "化学键", "元素", "化合物"],
            "历史": ["朝代", "战争", "革命", "文明", "帝国", "王朝"],
            "地理": ["国家", "城市", "河流", "山脉", "气候", "资源"]
        }

        found_domains = []
        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                found_domains.append(domain)

        return found_domains if found_domains else ["通用"]


class Deduplicator:
    """Handles exact and semantic deduplication."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service
        self.exact_hashes: Set[str] = set()

    def compute_hash(self, text: str) -> str:
        """Compute MD5 hash of normalized text."""
        normalized = re.sub(r'\s+', '', text.lower())
        return hashlib.md5(normalized.encode()).hexdigest()

    def exact_deduplicate(self, samples: List[DataSample]) -> Tuple[List[DataSample], List[Dict]]:
        """Remove exact duplicates."""
        unique_samples = []
        removed = []

        for s in samples:
            q_hash = self.compute_hash(s.question)
            a_hash = self.compute_hash(s.answer)
            combined_hash = q_hash + a_hash

            if combined_hash not in self.exact_hashes:
                self.exact_hashes.add(combined_hash)
                unique_samples.append(s)
            else:
                removed.append({"sample_id": s.id, "reason": "exact_duplicate"})

        return unique_samples, removed

    async def semantic_deduplicate(
        self,
        samples: List[DataSample],
        threshold: float = 0.95
    ) -> Tuple[List[DataSample], List[Dict]]:
        """Remove semantic duplicates using embeddings."""
        if not self.embedding_service or len(samples) < 2:
            return samples, []

        # Get embeddings for all samples
        texts = [f"{s.question} {s.answer}" for s in samples]
        embeddings = await self.embedding_service.get_embeddings_batch(texts)

        # Update samples with embeddings
        for s, emb in zip(samples, embeddings):
            s.embedding = emb

        # Compute similarity matrix (simplified for large datasets)
        unique_samples = []
        removed = []
        import math

        for i, s in enumerate(samples):
            is_duplicate = False

            for j, u in enumerate(unique_samples):
                if s.embedding and u.embedding:
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(s.embedding, u.embedding))
                    norm_s = math.sqrt(sum(a * a for a in s.embedding))
                    norm_u = math.sqrt(sum(a * a for a in u.embedding))

                    if norm_s > 0 and norm_u > 0:
                        similarity = dot / (norm_s * norm_u)
                        if similarity >= threshold:
                            is_duplicate = True
                            removed.append({
                                "sample_id": s.id,
                                "duplicate_of": u.id,
                                "similarity": round(similarity, 4),
                                "reason": "semantic_duplicate"
                            })
                            break

            if not is_duplicate:
                unique_samples.append(s)

        return unique_samples, removed


class Denoiser:
    """Handles denoising and PII masking."""

    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    MULTI_SPACE_PATTERN = re.compile(r'\s+')
    TRAILING_PUNCTUATION = re.compile(r'[,，\.。\s]+$')

    @staticmethod
    def remove_html(text: str) -> str:
        """Remove HTML tags."""
        return Denoiser.HTML_TAG_PATTERN.sub('', text)

    @staticmethod
    def fix_spacing(text: str) -> str:
        """Fix excessive spacing."""
        return Denoiser.MULTI_SPACE_PATTERN.sub(' ', text).strip()

    @staticmethod
    def fix_sentence_endings(text: str) -> str:
        """Fix missing or incorrect sentence endings."""
        if text and not text[-1] in '。！？.!?':
            text = text + '。'
        return text

    @staticmethod
    def remove_garbled(text: str) -> str:
        """Remove or replace garbled characters."""
        # Replace common garbled patterns
        replacements = {
            '\ufffd': '',  # Replacement character
            '\x00': '',     # Null character
            '\ufeff': '',   # BOM
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove non-printable characters
        text = ''.join(c if c.isprintable() or c in '\n\t' else ' ' for c in text)

        return Denoiser.fix_spacing(text)

    @staticmethod
    def mask_pii(text: str) -> Tuple[str, List[Dict]]:
        """Mask personally identifiable information."""
        masks = []

        # Phone numbers (Chinese mobile)
        phones = re.findall(r'1[3-9]\d{9}', text)
        for phone in phones:
            text = text.replace(phone, '[PHONE]')
            masks.append({"type": "phone", "original": phone})

        # Email
        emails = re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text)
        for email in emails:
            text = text.replace(email, '[EMAIL]')
            masks.append({"type": "email", "original": email})

        # ID numbers
        ids = re.findall(r'\b\d{15,18}\b', text)
        for id_num in ids:
            text = text.replace(id_num, '[ID]')
            masks.append({"type": "id_number", "original": id_num})

        # Bank cards
        banks = re.findall(r'\b\d{16,19}\b', text)
        for bank in banks:
            if bank not in [m["original"] for m in masks if m.get("type") == "phone"]:
                text = text.replace(bank, '[BANK_CARD]')
                masks.append({"type": "bank_card", "original": bank})

        return text, masks

    def denoise(self, text: str) -> str:
        """Apply full denoising pipeline."""
        text = self.remove_html(text)
        text = self.remove_garbled(text)
        text = self.fix_spacing(text)
        text = self.fix_sentence_endings(text)
        return text


class DataAugmenter:
    """Handles data augmentation: rewriting, reverse translation, synthetic generation."""

    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self._client = None

    async def initialize(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=120
        )

    async def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call LLM for augmentation."""
        if not self._client:
            await self.initialize()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.8,
            max_tokens=1000
        )
        return response.choices[0].message.content

    async def diversity_rewrite(self, question: str, answer: str) -> Tuple[str, str]:
        """Rewrite while preserving meaning but changing phrasing."""
        prompt = f"""请用不同的措辞、语气或结构重写以下问答对，同时保持完全相同的含义。

原始问答：
问：{question}
答：{answer}

要求：
1. 保持所有事实信息不变
2. 改变句子结构和词序
3. 改变语气（例如：从正式到口语，或反之）
4. 保持答案长度相同（±10%）

请以JSON格式回复：
{{
    "rewritten_question": "重写的问题",
    "rewritten_answer": "重写的答案"
}}"""

        response = await self._call_llm(prompt)
        try:
            result = json.loads(response)
            return result.get("rewritten_question", question), result.get("rewritten_answer", answer)
        except:
            return question, answer

    async def reverse_translation(self, question: str, answer: str) -> List[Dict]:
        """Generate more complex instruction pairs through logical deduction."""
        prompt = f"""请根据以下问答对，通过逻辑推导生成更复杂的变体：
1. 添加条件或约束
2. 改变范围（更一般或更具体）
3. 询问"为什么"的解释
4. 请求比较或对比

原始问答：
问：{question}
答：{answer}

请生成2-3个从原始问答逻辑推导出的变体，但要求更深入或不同视角。

请以JSON格式回复：
{{
    "variations": [
        {{
            "question": "变体问题",
            "answer": "对应答案",
            "type": "scope_general | scope_specific | explanation_why | comparison"
        }}
    ]
}}"""

        response = await self._call_llm(prompt)
        try:
            result = json.loads(response)
            return result.get("variations", [])
        except:
            return []

    async def generate_cot(self, question: str, answer: str) -> str:
        """Generate Chain-of-Thought reasoning for the answer."""
        prompt = f"""请为以下问答对生成详细的思维链，展示逐步思考的过程，最终得出答案。

问：{question}
答：{answer}

请生成思维链解释：
1. 将推理分解为清晰的步骤
2. 显示中间结论
3. 解释每一步为什么这样做
4. 逻辑地得出最终答案

请以JSON格式回复：
{{
    "chain_of_thought": "逐步推理过程..."
}}"""

        response = await self._call_llm(prompt)
        try:
            result = json.loads(response)
            return result.get("chain_of_thought", answer)
        except:
            return answer

    async def generate_synthetic_for_edge_cases(
        self,
        domain: str,
        concept: str,
        num_samples: int = 5
    ) -> List[Dict]:
        """Generate synthetic data for edge cases in a specific domain."""
        prompt = f"""请为 {domain} 领域的 {concept} 生成 {num_samples} 个边界情况问答对。

边界情况包括：
- 边界条件
- 非典型示例
- 极端场景
- 边界值
- 异常处理

每个问答对应该：
1. 测试对边界情况的理解
2. 有清晰的正确答案
3. 包含微妙的复杂性

请以JSON格式回复：
{{
    "edge_cases": [
        {{
            "question": "边界情况问题",
            "answer": "正确答案",
            "edge_case_type": "borderline | extreme | exception | boundary"
        }}
    ]
}}"""

        response = await self._call_llm(prompt)
        try:
            result = json.loads(response)
            return result.get("edge_cases", [])
        except:
            return []


class EvaluationLinker:
    """Links evaluation results back to data optimization."""

    @staticmethod
    def analyze_weak_areas(eval_results: Dict[str, Any], samples: List[DataSample]) -> Dict[str, Any]:
        """Analyze evaluation results to identify weak areas in the dataset."""
        if not eval_results.get("sample_results"):
            return {"weak_areas": [], "recommendations": []}

        # Find samples with low scores
        weak_samples = []
        for result in eval_results["sample_results"]:
            if result.get("arbiter_judgment", {}).get("overall_score", 100) < 75:
                weak_samples.append(result)

        # Group weaknesses by domain/topic
        weakness_by_topic = {}
        for ws in weak_samples:
            domain = ws.get("domain", "unknown")
            if domain not in weakness_by_topic:
                weakness_by_topic[domain] = []
            weakness_by_topic[domain].append(ws)

        # Generate recommendations
        recommendations = []
        for topic, issues in weakness_by_topic.items():
            avg_score = sum(w.get("arbiter_judgment", {}).get("overall_score", 0) for w in issues) / len(issues)
            recommendations.append({
                "topic": topic,
                "num_weak_samples": len(issues),
                "avg_score": round(avg_score, 2),
                "recommendation": f"需要增加{topic}领域的训练数据，特别是"
            })

        return {
            "weak_areas": weakness_by_topic,
            "recommendations": recommendations,
            "total_weak_samples": len(weak_samples)
        }

    @staticmethod
    def suggest_augmentation_for_weak_areas(
        weak_areas: Dict[str, Any],
        target_samples_per_area: int = 50
    ) -> List[Dict]:
        """Suggest augmentation targets for weak areas."""
        suggestions = []

        for topic, info in weak_areas.items():
            if isinstance(info, dict):
                suggestions.append({
                    "topic": topic,
                    "current_samples": info.get("num_samples", 0),
                    "target_samples": target_samples_per_area,
                    "augmentation_type": ["rewrite", "synthetic_generation"],
                    "priority": "high" if info.get("avg_score", 100) < 60 else "medium"
                })


# ---------------------------------------------------------------------------
# Shared helper: load dataset from state dict with title/content support
# ---------------------------------------------------------------------------
import os
import pandas as pd


def load_dataset_from_state(state: dict) -> Tuple[Optional[pd.DataFrame], str]:
    """Load a dataset from a state dict.

    Checks 'dataset_path', 'path', and 'output_path' keys for compatibility.
    Maps 'title' column to 'question' and 'content' column to 'answer'
    for datasets that use title/content instead of question/answer format.

    Returns:
        Tuple of (DataFrame with standardized columns, dataset_path string).
        Returns (None, "") if path not found or file doesn't exist.
    """
    dataset_path = state.get("dataset_path") or state.get("path") or state.get("output_path", "")
    if not dataset_path or not os.path.exists(dataset_path):
        return None, ""

    try:
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        else:
            return None, dataset_path
    except Exception:
        return None, dataset_path

    # Standardize column names: title→question, content→answer
    # This handles datasets like os_knowledge.json that use title/content
    if "title" in df.columns and "question" not in df.columns:
        df = df.rename(columns={"title": "question"})
    if "content" in df.columns and "answer" not in df.columns:
        df = df.rename(columns={"content": "answer"})
    # Also handle input/output (some formats use these)
    if "input" in df.columns and "question" not in df.columns:
        df = df.rename(columns={"input": "question"})
    if "output" in df.columns and "answer" not in df.columns:
        df = df.rename(columns={"output": "answer"})

    return df, dataset_path


def state_to_samples(state: dict) -> Tuple[List[DataSample], str]:
    """Convert state dict to list of DataSample objects.

    Returns:
        Tuple of (list of DataSample, dataset_path).
    """
    df, dataset_path = load_dataset_from_state(state)
    if df is None:
        return [], dataset_path

    samples = []
    for idx, row in df.iterrows():
        samples.append(DataSample(
            id=str(row.get("id", idx)),
            question=str(row.get("question", "")),
            answer=str(row.get("answer", ""))
        ))
    return samples, dataset_path
