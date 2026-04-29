"""
DeepAgents Data Optimization Agent
基于 DeepAgents 架构的智能体系统

目录结构:
- factory.py: 工厂函数，创建 deep agent
- base.py: Agent Skills 基础类型 (SkillStatus, SkillResult, BaseSkill)
- skills_loader.py: 动态加载 Agent Skills
- callback.py: Redis Callback Handler
- embedding_service.py: 嵌入服务和数据分析核心模块
- skills/: Agent Skills 目录
  - data_loader/
  - data_analyzer/
  - data_cleaner/
  - data_deduplicator/
  - data_augmenter/
  - data_generator/
  - data_validator/
  - text_normalizer/
"""

from .factory import create_data_opt_agent
from .skills_loader import (
    load_all_tools,
    list_available_tools,
    get_tool_by_name,
    list_skills,
    reload_skills,
    get_registry,
)
from .callback import DataOptCallbackHandler, create_callback_handler
from .base import SkillStatus, SkillResult, SkillManifest, BaseSkill, SkillRegistry

__all__ = [
    # Factory
    "create_data_opt_agent",
    # Skill loading
    "load_all_tools",
    "list_available_tools",
    "get_tool_by_name",
    "list_skills",
    "reload_skills",
    "get_registry",
    # Callback
    "DataOptCallbackHandler",
    "create_callback_handler",
    # Base types
    "SkillStatus",
    "SkillResult",
    "SkillManifest",
    "BaseSkill",
    "SkillRegistry",
]
