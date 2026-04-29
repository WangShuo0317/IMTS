"""
Base classes and types for Agent Skills paradigm.

This module defines the core abstractions for skills:
- SkillStatus: Enum representing skill execution state
- SkillResult: Dataclass for skill execution results
- BaseSkill: Abstract base class for skills
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from langchain_core.tools import BaseTool


class SkillStatus(Enum):
    """Skill execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SkillResult:
    """Result of a skill execution."""
    skill_name: str
    status: SkillStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "skill_name": self.skill_name,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }


@dataclass
class SkillManifest:
    """Manifest metadata for a skill loaded from SKILL.md."""
    name: str
    description: str
    skill_path: str
    tool_functions: List[str] = field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    when_to_use: Optional[str] = None
    examples: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], skill_path: str) -> "SkillManifest":
        """Create manifest from dictionary (parsed SKILL.md)."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            skill_path=skill_path,
            tool_functions=data.get("tool_functions", []),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            when_to_use=data.get("when_to_use"),
            examples=data.get("examples", [])
        )


class BaseSkill(ABC):
    """Abstract base class for Agent Skills.

    Skills can be implemented in two ways:
    1. As a class extending BaseSkill with execute() method
    2. As a function decorated with @tool (backward compatible)

    The skills_loader discovers and loads both patterns.
    """

    def __init__(self, manifest: SkillManifest):
        self.manifest = manifest
        self._status = SkillStatus.PENDING

    @property
    def name(self) -> str:
        """Skill name from manifest."""
        return self.manifest.name

    @property
    def description(self) -> str:
        """Skill description from manifest."""
        return self.manifest.description

    @property
    def status(self) -> SkillStatus:
        """Current skill status."""
        return self._status

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        """Execute the skill with given context.

        Args:
            context: Execution context with input parameters

        Returns:
            SkillResult with execution output or error
        """
        pass

    def to_langchain_tool(self) -> BaseTool:
        """Convert skill to LangChain BaseTool for agent integration.

        Default implementation wraps execute() method.
        Subclasses can override for custom tool behavior.
        """
        from langchain_core.tools import tool

        @tool
        async def skill_tool(input_str: str) -> Dict[str, Any]:
            """Execute this skill.

            Args:
                input_str: JSON string with input parameters

            Returns:
                Skill execution result
            """
            import json
            context = json.loads(input_str) if isinstance(input_str, str) else input_str
            result = await self.execute(context)
            return result.to_dict()

        skill_tool.name = self.name
        skill_tool.description = self.description
        return skill_tool


class SkillRegistry:
    """Registry for managing available skills.

    Provides centralized skill management for the agent.
    """

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}
        self._tools: Dict[str, BaseTool] = {}
        self._manifests: Dict[str, SkillManifest] = {}

    def register_skill(self, skill: BaseSkill) -> None:
        """Register a skill."""
        self._skills[skill.name] = skill
        self._tools[skill.name] = skill.to_langchain_tool()
        self._manifests[skill.name] = skill.manifest

    def register_tool(self, name: str, tool: BaseTool, manifest: Optional[SkillManifest] = None) -> None:
        """Register a LangChain tool directly."""
        self._tools[name] = tool
        if manifest:
            self._manifests[manifest.name] = manifest

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """Get skill by name."""
        return self._skills.get(name)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return self._tools.get(name)

    def get_manifest(self, name: str) -> Optional[SkillManifest]:
        """Get skill manifest by name."""
        return self._manifests.get(name)

    def list_skills(self) -> List[SkillManifest]:
        """List all registered skill manifests."""
        return list(self._manifests.values())

    def list_tools(self) -> List[BaseTool]:
        """List all registered tools."""
        return list(self._tools.values())

    @property
    def tools_dict(self) -> Dict[str, BaseTool]:
        """Get tools as dictionary."""
        return self._tools.copy()
