from __future__ import annotations

from typing import Any

from local_llm_lab.agent_protocol import SYSTEM_PROMPT, TOOL_SPECS

COMPLEX_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

Planning rules for complex tasks:
- Before acting on the environment, call set_plan with a concise goal and ordered checklist.
- Treat the plan as a working list, not as evidence that any step succeeded.
- After executing and verifying the checklist, call update_plan with completed items, the next
  action, and concise evidence from tool results.
- A plausible plan never substitutes for tool execution or verification."""
)

PLANNING_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_plan",
            "description": "Record a concise ordered plan before a complex task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                },
                "required": ["goal", "steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": "Record completed checklist items, next action, and observed evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "completed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "next": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["completed", "next", "evidence"],
            },
        },
    },
]

COMPLEX_TOOL_SPECS = [*PLANNING_TOOL_SPECS, *TOOL_SPECS]
