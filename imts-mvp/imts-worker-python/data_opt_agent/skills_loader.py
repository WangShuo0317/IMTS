"""
Agent Skills Loader - Dynamic skill discovery and loading.

This module implements the Agent Skills specification:
- Discovers skills from skills/ directory
- Parses SKILL.md for skill metadata
- Dynamically imports tools from scripts/tool.py
- Registers tools for use by the deep agent

Directory structure expected:
    skills/
        {skill_name}/
            SKILL.md          # Skill specification
            scripts/
                tool.py        # Tool implementations (LangChain @tool functions)
"""

import importlib
import importlib.util
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool, tool

from .base import SkillManifest, SkillRegistry


# Global registry instance
_registry: Optional[SkillRegistry] = None


def _parse_skill_md(skill_path: Path) -> Dict[str, Any]:
    """Parse SKILL.md file to extract metadata.

    Handles YAML frontmatter format:
        ---
        name: skill-name
        description: "Skill description..."
        ---

    Args:
        skill_path: Path to the skill directory

    Returns:
        Dictionary with skill metadata
    """
    skill_md_path = skill_path / "SKILL.md"
    if not skill_md_path.exists():
        return {}

    content = skill_md_path.read_text(encoding="utf-8")

    # Parse YAML frontmatter
    frontmatter = {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if match:
        yaml_content = match.group(1)
        for line in yaml_content.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().strip('"').strip("'")
                value = value.strip().strip('"').strip("'")
                frontmatter[key] = value

    # Extract structured sections
    result = {
        "name": frontmatter.get("name", skill_path.name),
        "description": frontmatter.get("description", ""),
    }

    # Parse input/output parameter tables
    param_pattern = r"\|\s*Parameter\s*\|\s*Type\s*\|\s*Required\s*\|\s*Description\s*\|[\s\r\n]+\|[-\s|]+\|[\s\r\n]+((?:\|[^\n]+\|[\s\r\n]*)+)"
    output_pattern = r"\|\s*Field\s*\|\s*Type\s*\|\s*Description\s*\|[\s\r\n]+\|[-\s|]+\|[\s\r\n]+((?:\|[^\n]+\|[\s\r\n]*)+)"

    input_match = re.search(param_pattern, content, re.IGNORECASE)
    if input_match:
        result["input_schema"] = _parse_table(input_match.group(1))

    output_match = re.search(output_pattern, content, re.IGNORECASE)
    if output_match:
        result["output_schema"] = _parse_table(output_match.group(1))

    # Extract when to use section
    when_match = re.search(r"##\s*When to Use\s*\n+\s*((?:.+\n(?:\s*.+\n)*))", content, re.IGNORECASE)
    if when_match:
        result["when_to_use"] = when_match.group(1).strip()

    return result


def _parse_table(table_content: str) -> List[Dict[str, str]]:
    """Parse markdown table into list of dicts."""
    lines = [l.strip() for l in table_content.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return []

    headers = [h.strip() for h in lines[0].split("|") if h.strip()]
    result = []

    for line in lines[1:]:
        if line.startswith("|") or re.match(r"^\s*\|", line):
            cells = [c.strip() for c in re.split(r"\|\s*", line) if c.strip()]
            if len(cells) >= len(headers):
                row = {headers[i]: cells[i] for i in range(len(headers))}
                result.append(row)

    return result


def _discover_tool_functions(tool_py_path: Path) -> List[str]:
    """Discover @tool decorated functions in tool.py.

    Args:
        tool_py_path: Path to scripts/tool.py

    Returns:
        List of tool function names
    """
    if not tool_py_path.exists():
        return []

    content = tool_py_path.read_text(encoding="utf-8")

    # Find all function definitions decorated with @tool
    # Pattern: @tool (possibly with async) def function_name
    pattern = r"@(?:.*?\.)?tool\s*(?:async\s+)?def\s+(\w+)"
    matches = re.findall(pattern, content)
    return matches


def _load_tools_from_skill(skill_path: Path) -> Tuple[SkillManifest, List[BaseTool]]:
    """Load a single skill and its tools.

    Args:
        skill_path: Path to the skill directory

    Returns:
        Tuple of (SkillManifest, list of LangChain tools)
    """
    # Parse manifest
    manifest_data = _parse_skill_md(skill_path)
    manifest = SkillManifest.from_dict(manifest_data, str(skill_path))

    # Load tool.py module
    tool_py_path = skill_path / "scripts" / "tool.py"
    if not tool_py_path.exists():
        return manifest, []

    # Dynamically import the tool module
    module_name = f"data_opt_agent.skills.{skill_path.name}.scripts.tool"
    spec = importlib.util.spec_from_file_location(module_name, tool_py_path)
    if spec is None or spec.loader is None:
        return manifest, []

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Discover and collect @tool decorated functions
    tools = []
    tool_function_names = _discover_tool_functions(tool_py_path)

    for func_name in tool_function_names:
        if hasattr(module, func_name):
            func = getattr(module, func_name)
            if isinstance(func, BaseTool):
                tools.append(func)
            elif callable(func):
                # Wrap callable into a tool if not already wrapped
                from langchain_core.tools import tool
                wrapped = tool(func)
                wrapped.name = getattr(func, "name", func_name)
                wrapped.description = getattr(func, "description", "")
                tools.append(wrapped)

    manifest.tool_functions = tool_function_names
    return manifest, tools


def _get_skills_root() -> Path:
    """Get the skills directory path.

    Returns:
        Path to skills directory
    """
    # Assume this file is in data_opt_agent/
    current_file = Path(__file__).resolve()
    skills_root = current_file.parent / "skills"

    # Fallback: check relative to current working directory
    if not skills_root.exists():
        skills_root = Path("data_opt_agent/skills")

    return skills_root


def _discover_skill_directories(skills_root: Path) -> List[Path]:
    """Discover all skill directories.

    Args:
        skills_root: Path to skills directory

    Returns:
        List of skill directory paths
    """
    if not skills_root.exists():
        return []

    skill_dirs = []
    for entry in skills_root.iterdir():
        if entry.is_dir() and not entry.name.startswith("_"):
            # Check if this looks like a skill (has SKILL.md or scripts/tool.py)
            if (entry / "SKILL.md").exists() or (entry / "scripts" / "tool.py").exists():
                skill_dirs.append(entry)

    return sorted(skill_dirs, key=lambda p: p.name)


def _initialize_registry() -> SkillRegistry:
    """Initialize the global skill registry.

    Returns:
        Populated SkillRegistry
    """
    registry = SkillRegistry()
    skills_root = _get_skills_root()

    for skill_dir in _discover_skill_directories(skills_root):
        try:
            manifest, tools = _load_tools_from_skill(skill_dir)
            for t in tools:
                registry.register_tool(t.name, t, manifest)
        except Exception as e:
            import logging
            logging.warning(f"Failed to load skill from {skill_dir}: {e}")

    return registry


def get_registry() -> SkillRegistry:
    """Get the global skill registry (lazy initialization).

    Returns:
        SkillRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = _initialize_registry()
    return _registry


def load_all_tools() -> List[BaseTool]:
    """Load all available tools from skills directory.

    Returns:
        List of LangChain BaseTool instances
    """
    registry = get_registry()
    return registry.list_tools()


def list_available_tools() -> List[Dict[str, Any]]:
    """List all available tools with metadata.

    Returns:
        List of tool info dicts with name, description, source skill
    """
    registry = get_registry()
    tools = registry.list_tools()

    return [
        {
            "name": t.name,
            "description": t.description if hasattr(t, "description") else "",
            "source": "skills"
        }
        for t in tools
    ]


def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """Get a specific tool by name.

    Args:
        name: Tool name

    Returns:
        BaseTool instance or None if not found
    """
    registry = get_registry()
    return registry.get_tool(name)


def list_skills() -> List[SkillManifest]:
    """List all skill manifests.

    Returns:
        List of SkillManifest for each discovered skill
    """
    registry = get_registry()
    return registry.list_skills()


def reload_skills() -> SkillRegistry:
    """Force reload of all skills.

    Useful during development when skills change.

    Returns:
        Freshly populated SkillRegistry
    """
    global _registry
    _registry = _initialize_registry()
    return _registry
