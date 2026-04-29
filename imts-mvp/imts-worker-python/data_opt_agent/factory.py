"""
Factory module for creating deep agents.
"""

import os
import re
from typing import Union
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.utils import validate_path as original_validate_path
from .skills_loader import load_all_tools, list_available_tools
from .tools import CUSTOM_TOOLS
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel


def _patch_validate_path():
    """Monkey patch validate_path to accept MinIO URIs and handle Windows paths."""
    import deepagents.backends.utils as utils_module

    def patched_validate_path(path: str, allowed_prefixes=None):
        """Patched version that accepts MinIO URIs and Windows paths."""
        # Accept MinIO URIs directly
        if path.startswith("minio://"):
            return path

        # Convert known virtual paths to Windows paths
        if path.startswith("/IMTS/"):
            path = "D:" + path

        # Check for Windows absolute path
        if re.match(r"^[a-zA-Z]:[/\\]", path):
            # It's a Windows absolute path, convert to forward slash format
            path = path.replace("\\", "/")
            return path

        # For other paths, use original validation
        return original_validate_path(path, allowed_prefixes)

    # Apply the patch
    utils_module.validate_path = patched_validate_path


# Apply the monkey patch at module import time
_patch_validate_path()

SKILLS_PATH = "data_opt_agent/skills/"

SYSTEM_PROMPT = """你是一个数据优化智能体。你只能按照下面的固定流程执行，不得跳步、不得回退、不得重复调用任何工具。

## 执行步骤（必须严格按顺序执行，每步完成后才能执行下一步）

第1步：调用 data_loader(path=数据集路径)
第2步：调用 exact_deduplicate(state=第1步返回结果)
第3步：调用 batch_clean(state=第2步返回结果)
第4步：调用 data_validator(state=第3步返回结果)
第5步：回复 {"output_path": 第4步返回结果中的path或output_path字段}

注意：
- 第5步不需要调用write_file，直接回复JSON即可。
- 每个步骤必须等待上一个工具返回结果后，将完整返回值作为下一个工具的state参数传入。
- 不要在第5步之后再调用任何工具。

## 绝对禁止

- 禁止在完成上述5步后继续调用任何工具
- 禁止重复调用已执行过的工具
- 禁止同时调用多个工具
- 禁止使用 Agent 或 write_todos 工具
- 禁止使用 #RETURNED_DATA_FROM_ 之类的占位符
- 禁止编造路径（如 /path/to/dataset.json）
- 禁止跳过某个步骤直接执行后面的步骤"""


def create_data_opt_agent(
    model: Union[str, BaseChatModel] = "anthropic:claude-sonnet-4-6",
    api_key: str = None,
    base_url: str = None,
    redis_client=None,
    backend=None,
    **kwargs
):
    """Create a deep agent for data optimization.

    Tools are automatically loaded from the skills directory following
    Agent Skills specification (SKILL.md + scripts/tool.py).

    Args:
        model: Model to use. Can be:
              - String in provider:model format (e.g., "anthropic:claude-sonnet-4-6")
              - Pre-configured LangChain chat model instance
              - For Qwen, use model="qwen-plus" with base_url to use OpenAI-compatible API
        api_key: API key for the LLM provider.
        base_url: Base URL for the LLM provider API.
        redis_client: Redis client for progress broadcasting (reserved for future use).
        backend: Optional backend for file storage. Defaults to FilesystemBackend with
                 project root as root_dir for Windows compatibility.
        **kwargs: Additional arguments passed to create_deep_agent.

    Returns:
        Compiled deep agent graph.
    """
    tools = load_all_tools()
    # Add custom tools to override deepagents built-ins (especially read_file, ls, write_file)
    tools.extend(CUSTOM_TOOLS)

    # Remove deepagents internal tools that confuse the LLM (Agent, write_todos, update_todo, etc.)
    blocked_tool_names = {"Agent", "write_todos", "update_todo", "spawn_agent"}
    tools = [t for t in tools if t.name not in blocked_tool_names]

    # Resolve model - handle string vs instance
    if isinstance(model, str):
        # Check if using Qwen or other OpenAI-compatible models
        if model.startswith("openai:") or "qwen" in model.lower():
            # Use ChatOpenAI directly to avoid Responses API issues
            model_name = model.replace("openai:", "") if model.startswith("openai:") else model
            chat_model = ChatOpenAI(
                model=model_name,
                api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("EVAL_API_KEY"),
                base_url=base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("EVAL_BASE_URL"),
                streaming=True,
                request_timeout=120
            )
            model = chat_model
        elif api_key or base_url:
            # Fix #10: 不再污染 os.environ，直接创建 ChatOpenAI 实例
            chat_model = ChatOpenAI(
                model=model,
                api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("EVAL_API_KEY"),
                base_url=base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("EVAL_BASE_URL"),
                streaming=True,
                request_timeout=120
            )
            model = chat_model

    # Use FilesystemBackend for Windows compatibility if no backend specified
    if backend is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backend = FilesystemBackend(root_dir=project_root)

    return create_deep_agent(
        model=model,
        tools=tools,
        skills=[SKILLS_PATH],
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        **kwargs
    )
