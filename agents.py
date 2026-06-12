"""
agents.py — NIM-powered specialized agents via LangChain
Each agent uses the best NVIDIA NIM model for its role.

Agent → NIM Model Mapping:
  Planner    → meta/llama-3.3-70b-instruct
  Researcher → meta/llama-3.3-70b-instruct
  Coder      → qwen/qwen2.5-coder-32b-instruct
  Tester     → qwen/qwen2.5-coder-32b-instruct
  Optimizer  → nvidia/llama-3.1-nemotron-70b-instruct
"""
import asyncio
import json
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from config import model_cfg, swarm_cfg
from llm_client import LLMClient
from message_bus import MessageBus

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    IDLE    = "idle"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


@dataclass
class SubTask:
    task_id:     str
    title:       str
    description: str
    agent_type:  str
    depends_on:  List[str]    = field(default_factory=list)
    result:      Optional[str] = None
    error:       Optional[str] = None
    retries:     int   = 0
    started_at:  float = 0.0
    ended_at:    float = 0.0

    @classmethod
    def new(cls, title, description, agent_type, depends_on=None):
        return cls(
            task_id=uuid.uuid4().hex[:8],
            title=title,
            description=description,
            agent_type=agent_type,
            depends_on=depends_on or [],
        )


class BaseAgent(ABC):
    ROLE      = "base"
    NIM_MODEL = None   # None = use global config model

    def __init__(self, llm: LLMClient, bus: MessageBus, agent_id: str = ""):
        self.agent_id    = agent_id or f"{self.ROLE}_{uuid.uuid4().hex[:6]}"
        self.llm         = llm
        self.bus         = bus
        self.state       = AgentState.IDLE
        self.tasks_done   = 0
        self.tasks_failed = 0

    async def start(self):
        await self.bus.register(self.agent_id)
        await self.bus.subscribe("all", self.agent_id)
        logger.info("[%s] registered | NIM model: %s",
                    self.agent_id, self.NIM_MODEL or model_cfg.model)

    async def stop(self):
        await self.bus.unregister(self.agent_id)

    @abstractmethod
    async def process(self, subtask: SubTask) -> str: ...

    def stats(self):
        return {
            "agent_id":    self.agent_id,
            "role":        self.ROLE,
            "nim_model":   self.NIM_MODEL or model_cfg.model,
            "state":       self.state.value,
            "tasks_done":  self.tasks_done,
            "tasks_failed": self.tasks_failed,
        }


class PlannerAgent(BaseAgent):
    ROLE      = "planner"
    NIM_MODEL = "meta/llama-3.3-70b-instruct"

    SYSTEM = """You are an expert AI project planner for a multi-agent swarm system.
Your job: decompose complex goals into ordered, atomic subtasks.

RULES:
- Each subtask must be executable by ONE of: researcher | coder | tester | optimizer
- Use depends_on to express sequential ordering
- Keep subtasks focused (single responsibility)
- Output ONLY valid JSON — no markdown fences, no explanations

JSON SCHEMA (strict):
{
  "plan_title": "Short plan name",
  "subtasks": [
    {
      "id": "t1",
      "title": "Short title",
      "description": "Detailed description of what to do",
      "agent_type": "researcher|coder|tester|optimizer",
      "depends_on": []
    }
  ]
}"""

    async def process(self, subtask: SubTask) -> str:
        prompt = (
            f"Goal: {subtask.description}\n\n"
            f"Create a JSON execution plan with 4-8 subtasks. "
            f"Each must be handled by: researcher, coder, tester, or optimizer."
        )
        raw = await self.llm.generate(
            prompt, system=self.SYSTEM, agent_type=self.ROLE, max_tokens=3000
        )
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
        clean = match.group(1) if match else raw.strip()
        plan = json.loads(clean)
        return json.dumps(plan, indent=2)


class ResearcherAgent(BaseAgent):
    ROLE      = "researcher"
    NIM_MODEL = "meta/llama-3.3-70b-instruct"

    SYSTEM = """You are a senior AI research analyst with deep expertise in software engineering,
machine learning, and system design.

For the given research topic, provide a structured brief covering:
1. Key Concepts & Definitions
2. State-of-the-Art Approaches
3. Relevant Libraries, Tools & APIs
4. Trade-offs & Considerations
5. Recommended Implementation Strategy

Be concise but thorough. Use numbered sections and bullet points."""

    async def process(self, subtask: SubTask) -> str:
        prompt = (
            f"Research Topic: {subtask.title}\n\n"
            f"Details: {subtask.description}\n\n"
            f"Provide a structured research brief for the implementation team."
        )
        return await self.llm.generate(
            prompt, system=self.SYSTEM, agent_type=self.ROLE, max_tokens=2048
        )


class CoderAgent(BaseAgent):
    ROLE      = "coder"
    NIM_MODEL = "qwen/qwen2.5-coder-32b-instruct"

    SYSTEM = """You are an expert software engineer specializing in Python, APIs, and ML systems.

REQUIREMENTS:
- Write clean, production-grade, fully runnable code
- Include type hints, docstrings, and inline comments for complex logic
- Handle edge cases and errors gracefully
- Follow SOLID principles and PEP 8
- Use async/await where appropriate for I/O operations
- Output ONLY the code block with the correct language tag (```python, ```bash, etc.)"""

    async def process(self, subtask: SubTask) -> str:
        prompt = (
            f"Task: {subtask.title}\n\n"
            f"Requirements:\n{subtask.description}\n\n"
            f"Generate complete, production-ready, runnable code."
        )
        return await self.llm.generate(
            prompt, system=self.SYSTEM, agent_type=self.ROLE, max_tokens=3500
        )


class TesterAgent(BaseAgent):
    ROLE      = "tester"
    NIM_MODEL = "qwen/qwen2.5-coder-32b-instruct"

    SYSTEM = """You are a rigorous QA engineer and test automation expert.

For the given code or specification:
1. Write comprehensive pytest unit tests covering happy path + edge cases
2. Include fixtures, mocks, and parametrized tests where appropriate
3. Identify potential bugs, anti-patterns, and security issues
4. Provide coverage assessment

OUTPUT FORMAT:
## Test Suite
```python
<complete pytest test code>
```

## Bug Report
- BUG-001: <description + severity>

## Security Issues
- SEC-001: <description>

## Coverage Assessment
<estimated % and what is covered/missing>"""

    async def process(self, subtask: SubTask) -> str:
        prompt = (
            f"Task to test: {subtask.title}\n\n"
            f"Code / specification:\n{subtask.description}\n\n"
            f"Write a complete test suite and produce a full QA report."
        )
        return await self.llm.generate(
            prompt, system=self.SYSTEM, agent_type=self.ROLE, max_tokens=3000
        )


class OptimizerAgent(BaseAgent):
    ROLE      = "optimizer"
    NIM_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"

    SYSTEM = """You are a performance engineering expert targeting AMD MI300X (192GB HBM3, ROCm 6.x).

For the given code or algorithm:
1. Analyze current time/space complexity (Big-O)
2. Identify bottlenecks (CPU-bound, I/O-bound, memory-bound)
3. Suggest and implement optimizations:
   - Algorithmic improvements
   - Vectorization (NumPy, PyTorch on ROCm)
   - Async / parallel execution (asyncio, concurrent.futures)
   - AMD ROCm / HIP GPU acceleration where applicable
   - Memory access pattern improvements
4. Show before vs after with expected speedup

TARGET: AMD MI300X (192GB HBM3, 5.3TB/s memory bandwidth, 304 CUs, ROCm 6.x)"""

    async def process(self, subtask: SubTask) -> str:
        prompt = (
            f"Optimization target: {subtask.title}\n\n"
            f"Code / algorithm:\n{subtask.description}\n\n"
            f"Analyze and produce an optimized version tuned for AMD MI300X ROCm."
        )
        return await self.llm.generate(
            prompt, system=self.SYSTEM, agent_type=self.ROLE, max_tokens=3000
        )


# ── Agent Registry & Factory ──────────────────────────────────────────────────

AGENT_REGISTRY: Dict[str, type] = {
    "planner":    PlannerAgent,
    "researcher": ResearcherAgent,
    "coder":      CoderAgent,
    "tester":     TesterAgent,
    "optimizer":  OptimizerAgent,
}


def create_agent(role: str, llm: LLMClient, bus: MessageBus) -> BaseAgent:
    cls = AGENT_REGISTRY.get(role)
    if cls is None:
        raise ValueError(f"Unknown role: {role!r}. Choose from {list(AGENT_REGISTRY)}")
    return cls(llm=llm, bus=bus)
