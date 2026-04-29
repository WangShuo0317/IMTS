"""
Simplified Evaluation - Two-Model Architecture

This module provides evaluation with separate inference and evaluation models:
- Inference Model (vLLM on remote GPU): Generates model answers using LoRA-finetuned model
- Evaluation Model (DashScope qwen-max): Evaluates generated answers against ground truth

Architecture:
1. Load test data (Q&A pairs with ground truth)
2. For each question, call Inference Model (vLLM) to generate answer
3. Call Evaluation Model (DashScope) to evaluate:
   - Compare generated answer with ground truth
   - Score on multiple dimensions
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Callable
import pandas as pd

logger = logging.getLogger(__name__)

# Fix #5: 引入指数退避重试装饰器
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from retry_utils import async_retry


class ModelClient:
    """Unified LLM client for both inference and evaluation"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout: int = 60
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self._client = None

    async def initialize(self):
        """Initialize the OpenAI-compatible client"""
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        logger.info(f"ModelClient initialized: model={self.model_name}, base_url={self.base_url}")

    @async_retry(max_retries=3, base_delay=1.0, retryable_exc=(Exception,))
    async def generate(self, prompt: str, systemPrompt: str = None) -> str:
        """Generate text using the model"""
        if not self._client:
            await self.initialize()

        messages = []
        if systemPrompt:
            messages.append({"role": "system", "content": systemPrompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    @async_retry(max_retries=3, base_delay=1.0, retryable_exc=(Exception,))
    async def generate_streaming(self, prompt: str, systemPrompt: str = None, on_token: Callable = None) -> str:
        """Generate text using the model with streaming support"""
        if not self._client:
            await self.initialize()

        messages = []
        if systemPrompt:
            messages.append({"role": "system", "content": systemPrompt})
        messages.append({"role": "user", "content": prompt})

        full_content = ""
        try:
            stream = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stream=True
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                full_content += token
                if on_token:
                    await on_token(token)
            return full_content
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise


class InferenceModel:
    """Inference Model - Uses vLLM on remote GPU server to generate model answers.

    vLLM serves the LoRA-finetuned model with OpenAI-compatible API.
    When a LoRA module is loaded, the model name includes the LoRA suffix.
    Evaluation MUST use the trained model via vLLM — no fallback to untrained models.
    """

    def __init__(self, base_url: str, model_name: str, api_key: str = "not-needed"):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self._client = None

    async def initialize(self):
        """Initialize inference client (vLLM OpenAI-compatible API)."""
        self._client = ModelClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name,
            timeout=120
        )
        await self._client.initialize()
        logger.info(f"InferenceModel initialized: {self.model_name} @ {self.base_url}")

    async def generate_answer(self, question: str) -> str:
        """Generate answer for a question using inference model"""
        if not self._client:
            await self.initialize()

        prompt = f"""## 任务：回答问题

你是一位专业、耐心的AI助手。请根据你的知识准确回答用户的问题。

### 问题
{question}

### 要求
1. 首先理解问题的核心是什么
2. 如果问题有明确答案，直接给出
3. 如果问题需要解释，提供清晰、结构化的解释
4. 如果不确定答案，明确说明，不要编造
5. 回答要简洁明了，避免冗长

### 回答格式
直接给出答案，不需要额外格式。"""

        answer = await self._client.generate(
            prompt=prompt,
            systemPrompt="""你是一位知识渊博、严谨负责的AI助手。
- 对于事实性问题，确保答案准确无误
- 对于推理性问题，提供清晰的推理过程
- 对于开放性问题，给出全面但有重点的回答
- 不知道的问题明确表示不知道，不要虚构"""
        )
        return answer


class EvaluationModel:
    """Evaluation Model - Uses user-specified model to evaluate answers"""

    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self._client = None

    async def initialize(self):
        """Initialize evaluation client"""
        self._client = ModelClient(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name
        )
        await self._client.initialize()
        logger.info(f"EvaluationModel initialized: {self.model_name} @ {self.base_url}")

    async def generate_streaming(self, prompt: str, systemPrompt: str = None, on_token: Callable = None) -> str:
        """Generate text with streaming using the evaluation model"""
        if not self._client:
            await self.initialize()
        return await self._client.generate_streaming(prompt, systemPrompt, on_token)

    async def evaluate_sample(
        self,
        question: str,
        ground_truth: str,
        generated_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate the generated answer against ground truth.

        Returns scores for:
        - Fact Accuracy: Does the answer match the facts in ground truth?
        - Logic Consistency: Is the reasoning logical?
        - Completeness: Does it fully answer the question?
        - Relevance: Is the response relevant?
        """
        if not self._client:
            await self.initialize()

        prompt = f"""You are an expert evaluator. Evaluate the following answer quality.

Question: {question}
Ground Truth (Correct Answer): {ground_truth}
Model Generated Answer: {generated_answer}

Evaluate the Model Generated Answer on these dimensions:
1. Fact Accuracy (0-100): Does the answer match the facts in the ground truth? Consider if the key facts are correct.
2. Logic Consistency (0-100): Is the reasoning logical and well-structured?
3. Completeness (0-100): Does it fully answer the question?
4. Relevance (0-100): Is the response relevant to the question?

Respond in JSON format:
{{
    "fact_accuracy": 0-100,
    "logic_consistency": 0-100,
    "completeness": 0-100,
    "relevance": 0-100,
    "overall_score": 0-100,
    "issues": ["list of specific issues found"],
    "reasoning": "brief explanation of the evaluation"
}}"""

        try:
            response = await self._client.generate(
                prompt=prompt,
                systemPrompt="You are an expert evaluator. Always respond in valid JSON format."
            )

            # Parse JSON response
            result = json.loads(response)
            # Truncate large fields to reduce report size
            if "reasoning" in result and isinstance(result["reasoning"], str) and len(result["reasoning"]) > 500:
                result["reasoning"] = result["reasoning"][:500] + "..."
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation result: {e}, response: {response}")
            return {
                "fact_accuracy": 0,
                "logic_consistency": 0,
                "completeness": 0,
                "relevance": 0,
                "overall_score": 0,
                "issues": [f"JSON parse error: {str(e)}"],
                "reasoning": "Failed to parse evaluation result"
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "fact_accuracy": 0,
                "logic_consistency": 0,
                "completeness": 0,
                "relevance": 0,
                "overall_score": 0,
                "issues": [str(e)],
                "reasoning": "Evaluation failed"
            }


class TwoModelEvaluator:
    """
    Two-Model Evaluator:
    - Uses Inference Model (vLLM) to generate answers
    - Uses Evaluation Model (DashScope) to evaluate answers
    """

    def __init__(
        self,
        inference_base_url: str,
        inference_model: str,
        eval_api_key: str,
        eval_base_url: str,
        eval_model: str,
        inference_api_key: str = "not-needed"
    ):
        self.inference = InferenceModel(inference_base_url, inference_model, inference_api_key)
        self.evaluation = EvaluationModel(eval_api_key, eval_base_url, eval_model)

    async def initialize(self):
        """Initialize both models"""
        await self.inference.initialize()
        await self.evaluation.initialize()

    async def evaluate_sample(
        self,
        question: str,
        ground_truth: str,
        generated_answer: str,
        progress_callback: Callable = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample (answer already generated by batch inference).
        1. Emit question and generated answer
        2. Fact Evaluator checks factual accuracy - stream thought process
        3. Logic Checker evaluates logical consistency - stream thought process
        4. Arbiter produces final verdict - stream reasoning
        """
        async def emit(role, speaker, content, is_streaming=True, msg_type="CHAT_MESSAGE"):
            if progress_callback:
                cb = progress_callback(0, content, role=role, speaker=speaker, is_streaming=is_streaming, msg_type=msg_type)
                if asyncio.iscoroutine(cb):
                    await cb

        # Emit the question and generated answer
        await emit("MODEL", "Model Response", f"问题: {question[:100]}...", is_streaming=False, msg_type="CHAT_MESSAGE")
        await emit("MODEL", "Model Response", generated_answer, is_streaming=False, msg_type="CHAT_MESSAGE")

        # Step 2: Fact Evaluator - stream thought process
        await emit("FACT_EVALUATOR", "Fact Evaluator", "开始事实检查...", is_streaming=False, msg_type="CHAT_MESSAGE")

        fact_prompt = f"""## 任务：事实准确性评估

你是一名资深事实核查专家，负责验证模型回答中的事实是否准确。请严格按照以下标准进行评估。

### 评估对象
- **问题**: {question}
- **标准答案（Ground Truth）**: {ground_truth}
- **待评估回答**: {generated_answer}

### 评估步骤（必须按顺序执行）

**第一步：提取关键事实**
从模型回答中提取所有关键事实陈述，按点列出。

**第二步：逐项比对**
将每个事实与标准答案进行比对，标记为：
- ✓ 完全匹配
- △ 部分匹配（有条件成立）
- ✗ 不匹配（矛盾或错误）
- ○ 缺失（标准答案中有但回答未提及）

**第三步：识别幻觉**
检测是否存在以下问题：
- 与标准答案直接矛盾的内容
- 标准答案未提及的"补充"事实
- 看似合理但实际错误的信息

**第四步：计算评分**
- 完全匹配的事实：每个 +10分
- 部分匹配：每个 +5分
- 不匹配/幻觉：每个 -15分
- 缺失事实：每个 -8分
- 最终得分：0-100分（负分记为0）

### 输出要求
请以JSON格式输出：
{{
    "fact_accuracy": 0-100,
    "matched_facts": ["匹配的事实"],
    "missing_facts": ["缺失的事实"],
    "incorrect_facts": ["错误的事实"],
    "hallucinations": ["幻觉内容"],
    "reasoning": "详细的分析过程"
}}"""

        # Stream the fact evaluator response
        async def on_fact_token(token):
            if progress_callback:
                cb = progress_callback(0, token, role="FACT_EVALUATOR", speaker="Fact Evaluator", is_streaming=True, msg_type="CHAT_MESSAGE")
                if asyncio.iscoroutine(cb):
                    await cb

        fact_response = await self.evaluation.generate_streaming(
            prompt=fact_prompt,
            systemPrompt="""你是一位严谨的事实核查专家。你的职责是：
1. 仔细比对模型回答与标准答案中的每一个事实
2. 准确识别完全正确、部分正确、完全错误和缺失的内容
3. 特别注意"幻觉"——模型可能看似合理但实际错误的内容
4. 给出客观、可量化的评分

**评分原则**：
- 每发现一个与标准答案完全匹配的正确事实，加10分
- 每发现一个错误或矛盾的事实（幻觉），扣15分
- 每缺失一个重要事实，扣8分
- 分数范围：0-100分

请严格按照JSON格式输出你的评估结果。""",
            on_token=on_fact_token
        )

        # Parse fact accuracy from response
        try:
            fact_data = json.loads(fact_response)
            fact_accuracy = fact_data.get("fact_accuracy", 0)
        except json.JSONDecodeError:
            import re
            match = re.search(r'(\d+(?:\.\d+)?)\s*[分/]', fact_response)
            fact_accuracy = float(match.group(1)) if match else 50

        await emit("FACT_EVALUATOR", "Fact Evaluator", f"事实检查完成：事实准确性 = {fact_accuracy:.1f}/100", is_streaming=False, msg_type="CHAT_MESSAGE")

        # Step 3: Logic Checker - stream thought process
        await emit("LOGIC_CHECKER", "Logic Checker", "开始逻辑一致性检查...", is_streaming=False, msg_type="CHAT_MESSAGE")

        logic_prompt = f"""## 任务：逻辑一致性与完整性评估

你是一名逻辑分析专家，负责评估模型回答的推理质量和完整性。

### 评估对象
- **问题**: {question}
- **标准答案**: {ground_truth}
- **待评估回答**: {generated_answer}

### 评估维度

#### A. 逻辑一致性 (0-100分)
评估回答的推理过程是否合乎逻辑：

1. **推理链完整性**：回答是否有明确的推理步骤？步骤之间是否有逻辑联系？
2. **因果关系正确性**：原因→结果的推导是否正确？
3. **假设合理性**：回答中的假设是否合理？
4. **无逻辑漏洞**：是否存在自相矛盾或明显漏洞？

**扣分标准**：
- 推理链不完整：-15分
- 因果关系错误：-20分
- 假设不合理：-10分
- 存在自相矛盾：-25分

#### B. 完整性 (0-100分)
评估回答是否完整覆盖问题：

1. **问题覆盖**：是否回答了问题的所有方面？
2. **细节程度**：是否提供了足够的细节？
3. **边界情况**：是否考虑了边界情况？

**扣分标准**：
- 遗漏问题要点：-15分
- 细节不足：-10分
- 未考虑边界情况：-5分

#### C. 相关性 (0-100分)
评估回答是否紧扣问题：

1. **主题相关**：回答是否围绕问题核心？
2. **无冗余信息**：是否有与问题无关的内容？
3. **重点突出**：关键信息是否突出？

**扣分标准**：
- 严重跑题：-20分
- 冗余信息过多：-10分

### 输出要求
请以JSON格式输出：
{{
    "logic_consistency": 0-100,
    "completeness": 0-100,
    "relevance": 0-100,
    "logic_issues": ["逻辑问题列表"],
    "completeness_issues": ["完整性问题列表"],
    "relevance_issues": ["相关性问题列表"],
    "reasoning": "详细分析过程"
}}"""

        # Step 3: Logic Checker - stream thought process
        async def on_logic_token(token):
            if progress_callback:
                cb = progress_callback(0, token, role="LOGIC_CHECKER", speaker="Logic Checker", is_streaming=True, msg_type="CHAT_MESSAGE")
                if asyncio.iscoroutine(cb):
                    await cb

        logic_response = await self.evaluation.generate_streaming(
            prompt=logic_prompt,
            systemPrompt="""你是一位逻辑分析专家。你的职责是：
1. 严格检验模型回答的推理过程
2. 评估回答是否完整覆盖问题的各个方面
3. 判断回答是否紧扣问题核心

**评分权重**：
- 逻辑一致性：40%
- 完整性：35%
- 相关性：25%

请严格按照JSON格式输出评估结果。""",
            on_token=on_logic_token
        )

        try:
            logic_data = json.loads(logic_response)
            logic_consistency = logic_data.get("logic_consistency", 0)
            completeness = logic_data.get("completeness", 0)
            relevance = logic_data.get("relevance", 0)
        except json.JSONDecodeError:
            import re
            match = re.search(r'(\d+(?:\.\d+)?)\s*[分/]', logic_response)
            logic_consistency = float(match.group(1)) if match else 50
            completeness = 50
            relevance = 50

        await emit("LOGIC_CHECKER", "Logic Checker", f"逻辑审查完成：逻辑一致性 = {logic_consistency:.1f}/100, 完整性 = {completeness:.1f}/100, 相关性 = {relevance:.1f}/100", is_streaming=False, msg_type="CHAT_MESSAGE")

        # Step 4: Arbiter - produce final verdict
        await emit("ARBITER", "Arbiter", "综合评估结果...", is_streaming=False, msg_type="CHAT_MESSAGE")

        arbiter_prompt = f"""## 任务：综合裁决

你是首席评估官，需要综合事实检查和逻辑审查的结果，给出最终裁决。

### 评估对象
- **问题**: {question}
- **标准答案**: {ground_truth}
- **模型回答**: {generated_answer}

### 已有评估结果
- **事实准确性**: {fact_accuracy:.1f}/100
- **逻辑一致性**: {logic_consistency:.1f}/100
- **完整性**: {completeness:.1f}/100
- **相关性**: {relevance:.1f}/100

### 评分权重
根据问题类型动态调整权重：
- **事实类问题**（询问具体事实、数字、年份等）：事实准确性 40%，逻辑一致性 25%，完整性 20%，相关性 15%
- **推理类问题**（询问原因、结果、比较等）：逻辑一致性 40%，事实准确性 25%，完整性 20%，相关性 15%
- **综合类问题**：逻辑一致性 30%，完整性 30%，事实准确性 25%，相关性 15%

### 综合评分计算
overall_score = fact_accuracy × 权重1 + logic_consistency × 权重2 + completeness × 权重3 + relevance × 权重4

### 裁决标准
- **90-100分**：优秀 - 回答完全满足要求
- **75-89分**：良好 - 回答满足大部分要求
- **60-74分**：一般 - 回答满足基本要求
- **40-59分**：较差 - 存在明显问题
- **0-39分**：差 - 存在重大错误

### 输出要求
请以JSON格式输出最终裁决：
{{
    "fact_accuracy": 0-100,
    "logic_consistency": 0-100,
    "completeness": 0-100,
    "relevance": 0-100,
    "overall_score": 0-100,
    "grade": "优秀/良好/一般/较差/差",
    "issues": ["发现的具体问题列表"],
    "strengths": ["回答的优点"],
    "reasoning": "综合裁决理由"
}}"""

        async def on_arbiter_token(token):
            if progress_callback:
                cb = progress_callback(0, token, role="ARBITER", speaker="Arbiter", is_streaming=True, msg_type="CHAT_MESSAGE")
                if asyncio.iscoroutine(cb):
                    await cb

        arbiter_response = await self.evaluation.generate_streaming(
            prompt=arbiter_prompt,
            systemPrompt="""你是首席评估官，负责综合各项评估结果给出最终裁决。

**你的职责**：
1. 综合分析事实检查和逻辑审查的结果
2. 根据问题类型确定权重分配
3. 计算加权综合评分
4. 给出明确的等级评定
5. 指出回答的优点和不足

**裁决原则**：
- 实事求是，客观公正
- 分数要体现真实水平差距
- 特别严重的单项问题可一票否决

请严格按照JSON格式输出最终裁决结果。""",
            on_token=on_arbiter_token
        )

        try:
            final_result = json.loads(arbiter_response)
        except json.JSONDecodeError:
            final_result = {
                "fact_accuracy": fact_accuracy,
                "logic_consistency": logic_consistency,
                "completeness": completeness,
                "relevance": relevance,
                "overall_score": round(fact_accuracy * 0.35 + logic_consistency * 0.25 + completeness * 0.20 + relevance * 0.20, 1),
                "issues": ["评估结果解析失败"],
                "reasoning": "JSON解析失败，使用备用计算"
            }

        MAX_CONTENT_LEN = 500
        final_result["generated_answer"] = generated_answer[:MAX_CONTENT_LEN] + "..." if len(generated_answer) > MAX_CONTENT_LEN else generated_answer
        final_result["ground_truth"] = ground_truth[:MAX_CONTENT_LEN] + "..." if len(ground_truth) > MAX_CONTENT_LEN else ground_truth

        verdict = f"""裁决结果：
- 事实准确性: {final_result.get('fact_accuracy', 0):.1f}/100
- 逻辑一致性: {final_result.get('logic_consistency', 0):.1f}/100
- 完整性: {final_result.get('completeness', 0):.1f}/100
- 相关性: {final_result.get('relevance', 0):.1f}/100
- 综合评分: {final_result.get('overall_score', 0):.1f}/100"""
        await emit("ARBITER", "Arbiter", verdict, is_streaming=False, msg_type="CHAT_MESSAGE")

        return final_result

    async def batch_evaluate(
        self,
        samples: List[Dict[str, str]],
        progress_callback: Callable = None
    ) -> Dict[str, Any]:
        """Evaluate a batch of samples with concurrent inference then sequential evaluation.

        Phase 1: Concurrent inference — batch generate answers via vLLM (high GPU throughput).
        Phase 2: Sequential evaluation — DashScope evaluates each answer (rate-limited API).
        """
        total = len(samples)
        generated_answers: Dict[str, str] = {}

        # Phase 1: Concurrent inference via vLLM
        CONCURRENT_BATCH_SIZE = 8

        for batch_start in range(0, total, CONCURRENT_BATCH_SIZE):
            batch = samples[batch_start:batch_start + CONCURRENT_BATCH_SIZE]
            tasks = [self.inference.generate_answer(s.get("question", "")) for s in batch]
            answers = await asyncio.gather(*tasks, return_exceptions=True)

            for i, answer in enumerate(answers):
                sample = batch[i]
                sample_id = sample.get("id", f"sample_{batch_start + i}")
                if isinstance(answer, Exception):
                    logger.error(f"Inference failed for {sample_id}: {answer}")
                    generated_answers[sample_id] = f"推理失败: {answer}"
                else:
                    generated_answers[sample_id] = answer

            if progress_callback:
                pct = int((batch_start + len(batch)) / total * 50)
                cb = progress_callback(pct, f"推理进度 [{batch_start + len(batch)}/{total}]",
                                       role="MODEL", speaker="Model Response",
                                       is_streaming=False, msg_type="STAGE_PROGRESS")
                if asyncio.iscoroutine(cb):
                    await cb

        # Phase 2: Sequential evaluation via DashScope (rate-limited)
        results = []
        for i, sample in enumerate(samples):
            sample_id = sample.get("id", f"sample_{i}")
            question = sample.get("question", "")
            ground_truth = sample.get("ground_truth", "")
            generated_answer = generated_answers.get(sample_id, "推理未生成答案")

            if progress_callback:
                pct = 50 + int((i + 1) / total * 50)
                cb = progress_callback(pct, f"评估样本 [{i+1}/{total}]",
                                       role="ARBITER", speaker="Arbiter",
                                       is_streaming=False, msg_type="STAGE_PROGRESS")
                if asyncio.iscoroutine(cb):
                    await cb

            result = await self.evaluate_sample(question, ground_truth, generated_answer, progress_callback)
            result["sample_id"] = sample_id
            result["category"] = sample.get("category", sample.get("domain", "未分类"))
            results.append(result)

        return self._aggregate_results(results)

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate individual results into summary statistics"""
        if not results:
            return {
                "overall_score": 0,
                "passed": False,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "total_samples": 0,
                "detailed_metrics": {}
            }

        total = len(results)
        pass_threshold = 75.0

        # Calculate pass/fail for each sample
        passed_samples = sum(1 for r in results if r.get("overall_score", 0) >= pass_threshold)
        failed_samples = total - passed_samples

        # For binary classification (pass/fail based on threshold):
        # Each sample is either pass (>=75) or fail (<75)
        tp = passed_samples  # True positives: correctly passed
        fp = 0              # False positives: incorrectly passed (none, all should pass)
        fn = failed_samples # False negatives: incorrectly failed
        tn = 0              # True negatives: correctly failed (none, all should pass)

        # Accuracy: (TP + TN) / Total
        accuracy = (tp + tn) / total if total > 0 else 0

        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1 Score: 2 * Precision * Recall / (Precision + Recall)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate average dimension scores
        dimensions = ["fact_accuracy", "logic_consistency", "completeness", "relevance"]
        aggregated = {}

        for dim in dimensions:
            scores = [r.get(dim, 0) for r in results if dim in r]
            aggregated[dim] = sum(scores) / len(scores) if scores else 0

        # Overall score (weighted average)
        aggregated["overall_score"] = (
            aggregated["fact_accuracy"] * 0.35 +
            aggregated["logic_consistency"] * 0.25 +
            aggregated["completeness"] * 0.20 +
            aggregated["relevance"] * 0.20
        )

        # Radar chart data
        radar_data = {
            "dimensions": ["事实准确性", "逻辑一致性", "完整性", "相关性"],
            "scores": [
                aggregated["fact_accuracy"],
                aggregated["logic_consistency"],
                aggregated["completeness"],
                aggregated["relevance"]
            ],
            "max_score": 100
        }

        # Generate suggestions
        suggestions = []
        dim_cn = {
            "fact_accuracy": "事实准确性",
            "logic_consistency": "逻辑一致性",
            "completeness": "完整性",
            "relevance": "相关性"
        }

        for dim in dimensions:
            if aggregated[dim] < 75:
                suggestions.append({
                    "category": dim,
                    "description": f"{dim_cn[dim]}得分较低 ({aggregated[dim]:.1f})，建议加强训练",
                    "priority": "high" if aggregated[dim] < 60 else "medium",
                    "affected_samples": sum(1 for r in results if r.get(dim, 0) < 75)
                })

        # 限制 sample_results 数量，最多保留 10 个样本的详细结果

        # Category breakdown calculation
        category_stats = {}
        for r in results:
            # check domain or category in sample dict
            cat = r.get("category", "未分类")
            if cat not in category_stats:
                category_stats[cat] = {"count": 0, "total_score": 0}
            category_stats[cat]["count"] += 1
            category_stats[cat]["total_score"] += r.get("overall_score", 0)
            
        weak_categories = []
        for cat, stats in category_stats.items():
            stats["avg_score"] = round(stats["total_score"] / stats["count"], 2)
            if stats["avg_score"] < 75.0 and cat != "未分类":
                weak_categories.append(cat)
                
        if weak_categories:
            suggestions.append({
                "category": "category_performance",
                "description": f"以下类别得分较低：{', '.join(weak_categories)}，建议在下一轮进行针对性数据增强",
                "priority": "high",
                "weak_categories": weak_categories
            })
        max_sample_results = 10
        limited_results = results[:max_sample_results] if len(results) > max_sample_results else results

        return {
            "overall_score": round(aggregated["overall_score"], 2),
            "passed": aggregated["overall_score"] >= 75,
            "total_samples": len(results),
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1_score, 4),
            "detailed_metrics": aggregated,
            "radar_data": radar_data,
            "suggestions": suggestions,
            "sample_results": limited_results
        }


async def run_two_model_evaluation(
    state: Dict[str, Any],
    message_builder=None,
    progress_callback: Callable = None
) -> Dict[str, Any]:
    """
    Run evaluation with two-model architecture.

    Inference Model (vLLM on remote GPU): Generates model answers
    Evaluation Model (DashScope qwen-max): Evaluates answers
    """
    import os
    from dotenv import load_dotenv

    # Load env
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)

    job_id = state.get("job_id", "unknown")
    dataset_path = state.get("dataset_path", "")

    # Get inference config (vLLM on remote GPU server)
    inference_base_url = os.getenv("INFERENCE_BASE_URL", "http://10.242.33.21:8001/v1")
    inference_model = os.getenv("INFERENCE_MODEL_NAME", "Qwen3-8B")
    inference_api_key = os.getenv("INFERENCE_API_KEY", "not-needed")

    # If training produced a LoRA adapter, use the LoRA model name
    train_result = state.get("train_result", {})
    lora_model_name = train_result.get("lora_model_name")
    if lora_model_name:
        inference_model = lora_model_name
        logger.info(f"[Evaluation] Using LoRA model: {lora_model_name}")

    # Get evaluation config (DashScope) — used to evaluate answers generated by vLLM
    eval_api_key = os.getenv("EVAL_API_KEY", "")
    eval_base_url = os.getenv("EVAL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    eval_model = os.getenv("EVAL_MODEL_NAME", "qwen-max")

    logger.info(f"Starting two-model evaluation for job {job_id}")
    logger.info(f"  Inference (vLLM): {inference_model} @ {inference_base_url}")
    logger.info(f"  Evaluation (DashScope): {eval_model} @ {eval_base_url}")

    start_time = time.time()

    try:
        # Create two-model evaluator: vLLM generates, DashScope evaluates
        evaluator = TwoModelEvaluator(
            inference_base_url=inference_base_url,
            inference_model=inference_model,
            inference_api_key=inference_api_key,
            eval_api_key=eval_api_key,
            eval_base_url=eval_base_url,
            eval_model=eval_model
        )
        await evaluator.initialize()

        # Load test data
        test_data = await load_test_data(dataset_path, state)
        logger.info(f"DEBUG: load_test_data returned {len(test_data) if test_data else 0} samples for path: {dataset_path}")

        if not test_data:
            logger.warning(f"No test data loaded from: {dataset_path}, using mock data for evaluation")
            test_data = generate_mock_test_data(5)

        logger.info(f"Loaded {len(test_data)} test samples")

        # Run evaluation
        result = await evaluator.batch_evaluate(
            samples=test_data,
            progress_callback=progress_callback
        )

        # Add metadata
        end_time = time.time()
        result["evaluation_time_seconds"] = round(end_time - start_time, 2)
        result["job_id"] = job_id
        result["iteration"] = state.get("current_iteration", 1)
        result["inference_model"] = inference_model
        result["evaluation_model"] = eval_model

        logger.info(
            f"Evaluation complete for job {job_id}: "
            f"score={result.get('overall_score', 0):.2f}, "
            f"time={result.get('evaluation_time_seconds')}s"
        )

        return result

    except Exception as e:
        logger.error(f"Evaluation failed for job {job_id}: {e}", exc_info=True)
        return {
            "overall_score": 0,
            "passed": False,
            "error": str(e),
            "evaluation_time_seconds": time.time() - start_time,
            "job_id": job_id
        }


async def load_test_data(dataset_path: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load test data from dataset path - randomly sample 10% of the dataset for evaluation.
    This should be called AFTER data optimization, using the FINAL dataset that excludes test data.
    The actual test split should happen in the data optimization phase.
    For now, we sample 10% from the given dataset as the test set."""
    import os
    import pandas as pd
    import random

    # Normalize path
    dataset_path = dataset_path.replace("\\", "/")
    logger.info(f"[DEBUG load_test_data] dataset_path='{dataset_path}', exists={os.path.exists(dataset_path)}")

    all_data = []
    # Load directly from the given path
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.json'):
        df = pd.read_json(dataset_path)
    else:
        logger.warning(f"Unsupported dataset format: {dataset_path}")
        return []

    for idx, row in df.iterrows():
        # Support multiple field name formats
        question = str(row.get("question", row.get("title", row.get("input", ""))))
        answer = str(row.get("answer", row.get("content", row.get("output", ""))))

        # Skip if both are empty
        if not question.strip() and not answer.strip():
            continue

        all_data.append({
            "id": f"sample_{idx}",
            "question": question,
            "ground_truth": answer,
            "domain": str(row.get("domain", "")),
            "title": str(row.get("title", ""))
        })

    if not all_data:
        logger.warning(f"No valid data loaded from {dataset_path}")
        return []

    logger.info(f"Loaded {len(all_data)} test samples from {dataset_path}")
    return all_data  # Already pre-split, return all


def generate_mock_test_data(num_samples: int = 5) -> List[Dict[str, Any]]:
    """Generate mock test data for testing"""
    mock_data = [
        {
            "id": "mock_1",
            "question": "What is the capital of France?",
            "ground_truth": "Paris"
        },
        {
            "id": "mock_2",
            "question": "What is 2 + 2?",
            "ground_truth": "4"
        },
        {
            "id": "mock_3",
            "question": "Who wrote Hamlet?",
            "ground_truth": "William Shakespeare"
        },
        {
            "id": "mock_4",
            "question": "What is the largest planet in our solar system?",
            "ground_truth": "Jupiter"
        },
        {
            "id": "mock_5",
            "question": "What is H2O commonly known as?",
            "ground_truth": "Water"
        },
    ]

    result = []
    for i in range(num_samples):
        result.append(mock_data[i % len(mock_data)].copy())
        result[-1]["id"] = f"mock_{i+1}"

    return result


# Export main function
__all__ = ["run_two_model_evaluation", "TwoModelEvaluator"]
