"""
统一流式回调处理器 — deepagents integration

提供三种运行模式：
- "redis+console" : 同时输出到 Redis 和控制台（生产环境）
- "console"       : 仅输出到控制台（测试/调试）
- "silent"        : 静默模式，不输出任何内容

统一的 AsyncCallbackHandler 实现，替代原有的两条并行路径：
  1. DataOptCallbackHandler（ainvoke 模式，非流式）
  2. DataOptStreamingCallbackHandler（stream 模式，流式 + Redis）
"""

import asyncio
import io
import json
import sys
import os
from typing import Any, Dict, List, Optional

# 修复 Windows GBK 控制台编码问题
if sys.platform == "win32" and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ANSI color codes（Windows 10+ 原生支持，低于10 falls back）
_color_codes = {
    "header": "\033[95m", "blue": "\033[94m", "cyan": "\033[96m",
    "green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m",
    "bold": "\033[1m",   "dim": "\033[2m",   "reset": "\033[0m",
}


def _color(name: str, text: str) -> str:
    """Return colorized text. Falls back to plain text on unsupported terminals."""
    if sys.platform == "win32" and not os.getenv("WT_SESSION"):
        # Not in Windows Terminal — strip ANSI codes
        return text
    return f"{_color_codes.get(name, '')}{text}{_color_codes['reset']}"


class ConsoleStreamHandler:
    """
    统一的 ReAct 流式输出处理器。

    处理 deepagents agent.stream() 产生的 LangChain callback 事件，
    并按 mode 输出到对应目标。

    事件处理覆盖：
    - on_chat_model_start   : LLM 调用开始
    - on_llm_new_token       : 流式 token（打字机效果）
    - on_chat_model_end      : LLM 调用结束
    - on_tool_start          : 工具调用开始
    - on_tool_end            : 工具调用结束
    - on_tool_error          : 工具执行错误
    """

    def __init__(
        self,
        mode: str = "console",
        message_builder=None,
        job_id: str = "test",
        stop_checker=None,
    ):
        """
        Args:
            mode: "redis+console" | "console" | "silent"
            message_builder: MessageBuilder instance (Redis mode需要)
            job_id: 任务 ID（仅用于 console 模式的日志前缀）
            stop_checker: Async callable returning bool — 如果返回 True 则取消智能体执行
        """
        valid_modes = ("redis+console", "console", "silent")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        self.mode = mode
        self.builder = message_builder
        self.job_id = job_id
        self._stop_checker = stop_checker

        # LangChain callback manager 需要此属性来判断是否内联运行回调
        self.run_inline = False
        self.raise_error = False
        self.ignore_llm = False
        self.ignore_chain = False
        self.ignore_chat_model = False
        self.ignore_retriever = False
        self.ignore_agent = False
        self.ignore_retry = False
        self.ignore_custom_event = False

        # 状态
        self._tool_count = 0
        self._llm_count = 0
        self._current_thought = ""
        self._current_tool_name: Optional[str] = None
        self._current_tool_input: Optional[Any] = None
        self._pending_tasks: List[asyncio.Task] = []
        # 流式节流：每个 LLM 调用最多发送 20 次/秒
        self._last_stream_time = 0
        self._stream_interval = 0.05  # 50ms = 20次/秒
        self._last_sent_thought = ""
        self._step = 0

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _safe_emit(self, coro):
        """在不阻塞的情况下调度协程（仅适用于 Redis 模式）。"""
        if self.mode == "silent":
            return
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(coro)
            self._pending_tasks.append(task)
        except RuntimeError:
            pass

    def _print(self, text: str, color: str = None, file=sys.stdout):
        """仅在非 silent 模式打印到控制台。"""
        if self.mode == "silent":
            return
        text = _color(color, text) if color else text
        print(text, flush=True, file=file)

    # ------------------------------------------------------------------
    # LangChain AsyncCallbackHandler 接口
    # ------------------------------------------------------------------

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        流式 token 回调 — deepagents agent.stream() 会触发此方法。
        实现真正的打字机效果：节流后实时发送到 Redis。
        """
        if self.mode == "silent":
            return
        self._current_thought += token
        # 实时打印 token 到控制台
        self._print(token, color="cyan", file=sys.stdout)

        # 节流发送到 Redis（打字机效果）
        if self.builder and self.mode == "redis+console":
            import time
            now = time.time()
            # 每隔 _stream_interval 秒发送一次增量更新
            if now - self._last_stream_time >= self._stream_interval:
                self._last_stream_time = now
                # 只发送新增的部分，避免重复发送已发送的内容
                new_content = self._current_thought[len(self._last_sent_thought):]
                if new_content:
                    self._safe_emit(self.builder.agent_thought(
                        "DataOptAgent",
                        self._current_thought,  # 发送完整内容让前端直接替换
                        progress=30
                    ))
                    self._last_sent_thought = self._current_thought

    async def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        """LLM 开始调用（旧版接口）。"""
        pass

    async def on_llm_end(self, response, **kwargs) -> None:
        """
        LLM 结束调用 — 提取响应内容用于显示。

        ainvoke 模式下，on_llm_new_token 不触发（streaming=False），
        实际内容通过此回调的 LLMResult.generations 传递。
        """
        if self.mode == "silent":
            return
        print(f"[DEBUG] on_llm_end called, response type: {type(response)}", flush=True)
        print(f"[DEBUG] response dir: {[a for a in dir(response) if not a.startswith('_')]}", flush=True)
        try:
            # 从 LLMResult 中提取文本
            text = ""
            if hasattr(response, "generations") and response.generations:
                gen = response.generations[0]
                print(f"[DEBUG] gen[0] type: {type(gen[0]) if gen else None}", flush=True)
                if gen and hasattr(gen[0], "text"):
                    text = gen[0].text
                    print(f"[DEBUG] extracted text length: {len(text)}", flush=True)
                elif gen:
                    # ChatGeneration has text via .text property but also has .content
                    cg = gen[0]
                    print(f"[DEBUG] ChatGeneration type: {type(cg)}, fields: {[a for a in dir(cg) if not a.startswith('_')]}", flush=True)
                    if hasattr(cg, "text") and cg.text:
                        text = cg.text
                        print(f"[DEBUG] via .text: length={len(text)}", flush=True)
                    if hasattr(cg, "content") and cg.content:
                        text = cg.content
                        print(f"[DEBUG] via .content: length={len(text)}, content[:100]={str(text)[:100]}", flush=True)
                    if hasattr(cg, "message") and cg.message:
                        msg = cg.message
                        print(f"[DEBUG] message type: {type(msg)}, content={getattr(msg, 'content', None)}", flush=True)
                        if hasattr(msg, 'content') and msg.content:
                            text = msg.content
                            print(f"[DEBUG] message.content: {str(text)[:200]}", flush=True)
                elif gen and isinstance(gen[0], dict):
                    text = gen[0].get("text", "")

            if not text:
                # 备用：直接遍历 generations
                try:
                    for g in response.generations:
                        for item in g:
                            if hasattr(item, "text"):
                                text = item.text
                                break
                        if text:
                            break
                except Exception as e:
                    print(f"[DEBUG] fallback extraction error: {e}", flush=True)

            # 如果 text 为空但 _current_thought 有内容（流式模式），使用累积的文本
            if not text and self._current_thought:
                text = self._current_thought
                print(f"[DEBUG] Using accumulated _current_thought, length: {len(text)}", flush=True)

            if text:
                self._current_thought = text
                print(flush=True)
                self._print(f"  → 响应内容: ", color="dim")
                lines = text.strip().split("\n")
                for line in lines[:30]:
                    self._print(f"    {line}", color="dim")
                if len(lines) > 30:
                    self._print(f"    ... [共 {len(lines)} 行]", color="dim")

                # Redis
                if self.builder and self.mode == "redis+console":
                    self._safe_emit(self.builder.agent_thought(
                        "DataOptAgent",
                        f"[LLM #{self._llm_count}] 响应:\n{text}",
                        progress=50
                    ))
            else:
                # 调试：打印 generations 的完整结构
                print(f"[DEBUG] text is empty, printing generations structure:", flush=True)
                try:
                    for i, gen_list in enumerate(response.generations):
                        print(f"[DEBUG] generations[{i}]: {gen_list}", flush=True)
                        for j, gen_item in enumerate(gen_list):
                            print(f"[DEBUG]   gen[{i}][{j}] type={type(gen_item)}", flush=True)
                            print(f"[DEBUG]   gen[{i}][{j}] dir: {[a for a in dir(gen_item) if not a.startswith('_')]}", flush=True)
                            if hasattr(gen_item, 'text'):
                                print(f"[DEBUG]   gen[{i}][{j}].text = {repr(gen_item.text)}", flush=True)
                            if hasattr(gen_item, 'content'):
                                print(f"[DEBUG]   gen[{i}][{j}].content = {repr(gen_item.content)[:200]}", flush=True)
                            if hasattr(gen_item, 'message'):
                                msg = gen_item.message
                                print(f"[DEBUG]   gen[{i}][{j}].message type={type(msg)}", flush=True)
                                if hasattr(msg, 'content'):
                                    print(f"[DEBUG]   gen[{i}][{j}].message.content = {repr(msg.content)[:200]}", flush=True)
                except Exception as e:
                    print(f"[DEBUG] Error printing generations: {e}", flush=True)
        except Exception as e:
            print(f"[DEBUG] on_llm_end exception: {e}", flush=True)
            # 不应阻断流程，静默忽略
            pass

    async def _check_stop(self) -> None:
        """检查 stop_signal，如果任务已被取消则抛出 CancelledError。"""
        if self._stop_checker:
            try:
                if await self._stop_checker():
                    raise asyncio.CancelledError(f"Job {self.job_id} stopped by user")
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages,
        **kwargs
    ) -> None:
        """LLM 开始处理消息。"""
        await self._check_stop()
        if self.mode == "silent":
            return
        self._llm_count += 1
        self._current_thought = ""
        print(f"[DEBUG] on_chat_model_start #{self._llm_count}: _current_thought cleared", flush=True)
        model_name = serialized.get("name", "unknown")

        separator = _color("header", "=" * 60)
        self._print(f"\n{separator}")
        self._print(f"  [LLM 调用 #{self._llm_count}]  模型: {model_name}")
        self._print(separator)
        self._print(f"  思考: ", color="yellow", file=sys.stdout)

        if self.builder and self.mode == "redis+console":
            self._safe_emit(self.builder.agent_thought(
                "DataOptAgent",
                f"[LLM #{self._llm_count}] 开始 (模型: {model_name})",
                progress=30
            ))

    async def on_chat_model_end(
        self,
        output: Any,
        **kwargs
    ) -> None:
        """LLM 结束处理 — 打印完整思考内容。"""
        if self.mode == "silent":
            return

        # 优先使用 stream 过程中累积的 _current_thought
        thought = self._current_thought

        # 如果没有 stream token，从 output 中提取响应内容
        if not thought and output:
            try:
                # output 可能是 AIMessage 或包含 messages 的结构
                if hasattr(output, "content") and output.content:
                    thought = output.content
                elif isinstance(output, dict):
                    # 尝试从 dict 结构中提取 content
                    content = output.get("content", "")
                    if content:
                        thought = content
                    elif "messages" in output:
                        for msg in output.get("messages", []):
                            if hasattr(msg, "content") and msg.content:
                                thought = msg.content
                                break
                elif isinstance(output, (list, tuple)):
                    for item in output:
                        if hasattr(item, "content") and item.content:
                            thought = item.content
                            break
            except Exception:
                pass

        if thought:
            # 换行，结束当前思考行的实时打印
            print(flush=True)
            self._print(f"  → 思考完成 (#{self._llm_count}): ", color="dim")
            # 分行打印完整思考
            thought_lines = thought.strip().split("\n")
            for line in thought_lines:
                self._print(f"    {line}", color="dim")
        self._current_thought = ""

        if self.builder and self.mode == "redis+console":
            self._safe_emit(self.builder.agent_thought(
                "DataOptAgent",
                f"[LLM #{self._llm_count}] 响应:\n{thought or ''}",
                progress=50
            ))

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs
    ) -> None:
        """工具调用决策 — 打印工具名称和参数。"""
        await self._check_stop()
        if self.mode == "silent":
            return

        tool_name = serialized.get("name", "unknown")
        self._tool_count += 1
        self._step += 1
        self._current_tool_name = tool_name

        # 解析并缓存工具输入，供 on_tool_end 使用
        try:
            if isinstance(input_str, str) and input_str.startswith("{"):
                self._current_tool_input = json.loads(input_str)
            else:
                self._current_tool_input = {"raw": str(input_str)[:500]}
        except:
            self._current_tool_input = {"raw": str(input_str)[:500]}

        # 换行（如果当前在打印思考）
        if self._current_thought:
            print(flush=True)

        # 打印工具调用框
        box = _color("yellow", "╔" + "═" * 58 + "╗")
        self._print(box)
        mid = _color("yellow", f"║  步骤 #{self._step}  │  Tool: {tool_name:<28}  ║")
        self._print(mid)
        self._print(_color("yellow", "╚" + "═" * 58 + "╝"))

        # 解析并打印参数
        try:
            if isinstance(input_str, str) and input_str.startswith("{"):
                parsed = json.loads(input_str)
            else:
                parsed = {"raw": str(input_str)[:500]}
        except:
            parsed = {"raw": str(input_str)[:500]}

        args_str = json.dumps(parsed, ensure_ascii=False)
        if len(args_str) > 300:
            args_str = args_str[:300] + "..."
        self._print(f"  参数: {args_str}", color="dim")

        # Redis
        if self.builder and self.mode == "redis+console":
            self._safe_emit(self.builder.tool_call(
                tool_name, parsed, {"status": "running"}
            ))

    async def on_tool_end(self, output: Any, **kwargs) -> None:
        """工具执行完成 — 打印返回结果。"""
        if self.mode == "silent":
            return

        tool_name = self._current_tool_name or "unknown"
        self._current_tool_name = None

        # 解析输出
        if isinstance(output, str):
            output_str = output
        else:
            output_str = str(output)

        MAX_LINES = 20
        MAX_CHARS = 800
        output_str = output_str[:MAX_CHARS]
        lines = output_str.split("\n")
        display_lines = lines[:MAX_LINES]
        truncated = len(lines) > MAX_LINES

        # 打印返回头
        status_ok = not output_str.lower().startswith("error")
        status_icon = _color("green", "✓") if status_ok else _color("red", "✗")
        self._print(f"\n  {status_icon} {tool_name} 返回:")

        for line in display_lines:
            self._print(f"      {line}", color="dim")

        if truncated:
            self._print(f"      ... [共 {len(lines)} 行，已截断]", color="dim")

        # Redis
        if self.builder and self.mode == "redis+console":
            self._safe_emit(self.builder.tool_call(
                tool_name,
                self._current_tool_input or {},
                {"status": "completed", "output": output_str[:500]}
            ))

        self._current_tool_input = None

    async def on_tool_error(self, error: Exception, **kwargs) -> None:
        """工具执行出错。"""
        if self.mode == "silent":
            return
        tool_name = self._current_tool_name or "unknown"
        self._print(f"\n  [{_color('red', 'TOOL ERROR')}] {tool_name}: {error}", color="red")
        if self.builder and self.mode == "redis+console":
            self._safe_emit(self.builder.error(
                f"工具 {tool_name} 执行错误: {str(error)}",
                {"error_type": type(error).__name__}
            ))

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs
    ) -> None:
        """Agent 链开始。"""
        if self.mode == "silent":
            return
        chain_name = serialized.get("name", "unknown") if serialized else "unknown"
        if "agent" in chain_name.lower() or "data_opt" in chain_name.lower():
            self._print(f"\n{_color('bold', '🚀 Agent 启动')}: {chain_name}")

    async def on_chain_end(self, output: Any, **kwargs) -> None:
        """Agent 链结束 — 打印最终的 LLM 响应内容。"""
        if self.mode == "silent":
            return

        # 尝试从 output 中提取最终的 LLM 响应
        if output and self._current_thought:
            print(flush=True)
            self._print(f"  → 思考完成 (#{self._llm_count}): ", color="dim")
            thought_lines = self._current_thought.strip().split("\n")
            for line in thought_lines:
                self._print(f"    {line}", color="dim")
            self._current_thought = ""

        # 注意：on_llm_end 已经发送了完整的响应内容到 Redis，
        # 这里不需要再发送额外的消息，避免重复

    async def on_retriever_end(self, documents: Any, **kwargs) -> None:
        pass

    async def on_text_end(self, text: str, **kwargs) -> None:
        pass

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @staticmethod
    def for_redis(message_builder, stop_checker=None) -> "ConsoleStreamHandler":
        """创建 Redis+console 模式 handler（生产环境）。"""
        job_id = message_builder.job_id if hasattr(message_builder, 'job_id') else "test"
        return ConsoleStreamHandler(mode="redis+console", message_builder=message_builder, job_id=job_id, stop_checker=stop_checker)

    @staticmethod
    def for_console(job_id: str = "test") -> "ConsoleStreamHandler":
        """创建 console-only 模式 handler（测试/调试）。"""
        return ConsoleStreamHandler(mode="console", job_id=job_id)

    @staticmethod
    def silent() -> "ConsoleStreamHandler":
        """创建静默模式 handler（批量执行）。"""
        return ConsoleStreamHandler(mode="silent")


# ============================================================================
# 向后兼容别名（供现有代码使用）
# ============================================================================

# 非流式版（ainvoke 模式），功能同 DataOptCallbackHandler
DataOptCallbackHandler = ConsoleStreamHandler

# 流式版（stream 模式），功能同 DataOptStreamingCallbackHandler
DataOptStreamingCallbackHandler = ConsoleStreamHandler


def create_callback_handler(job_id: str, redis_client, sync_redis_client=None) -> ConsoleStreamHandler:
    """向后兼容工厂函数。"""
    from message_types import MessageBuilder
    builder = MessageBuilder(job_id, redis_client, sync_redis_client)
    return ConsoleStreamHandler.for_redis(builder)
