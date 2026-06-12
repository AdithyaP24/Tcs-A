"""
config.py — NVIDIA NIM + AMD MI300X Swarm Configuration
TCS × AMD Hackathon
"""
import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModelConfig:
    # ── Backend ────────────────────────────────────────────────────────────────
    # 'nim_cloud'  → NVIDIA hosted API (api.nvidia.com)
    # 'nim_local'  → Self-hosted NIM on your infra
    # 'ollama'     → Local Ollama fallback
    backend: str = os.getenv("LLM_BACKEND", "nim_cloud")

    # ── NVIDIA NIM Settings ────────────────────────────────────────────────────
    # Get free API key at: https://build.nvidia.com
    nim_api_key: str = os.getenv("NVIDIA_API_KEY", "nvapi-REPLACE_WITH_YOUR_KEY")

    # NIM Cloud endpoint (default) or local NIM endpoint
    nim_base_url: str = os.getenv(
        "NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )

    # ── NIM Model Options ─────────────────────────────────────────────────────
    # "meta/llama-3.3-70b-instruct"            ← best general
    # "google/gemma-2-27b-it"                  ← matches original Gemma choice
    # "nvidia/llama-3.1-nemotron-70b-instruct" ← NVIDIA optimized
    # "qwen/qwen2.5-coder-32b-instruct"        ← best for code tasks
    # "deepseek-ai/deepseek-r1"                ← best reasoning
    # "mistralai/mistral-7b-instruct-v0.3"     ← fast + lightweight
    model: str = os.getenv("MODEL", "meta/llama-3.3-70b-instruct")

    # ── AMD MI300X Parallelism ─────────────────────────────────────────────────
    # MI300X handles orchestration; NIM provides LLM inference
    num_parallel: int = int(os.getenv("NUM_PARALLEL", "16"))
    ctx_size: int     = int(os.getenv("CTX_SIZE",     "8192"))

    # ── Per-Agent Temperatures ─────────────────────────────────────────────────
    temperatures: Dict[str, float] = field(default_factory=lambda: {
        "planner":    0.30,
        "researcher": 0.60,
        "coder":      0.15,
        "tester":     0.10,
        "optimizer":  0.40,
    })

    # ── Per-Agent Model Override ───────────────────────────────────────────────
    agent_models: Dict[str, str] = field(default_factory=lambda: {
        "planner":    "meta/llama-3.3-70b-instruct",
        "researcher": "meta/llama-3.3-70b-instruct",
        "coder":      "qwen/qwen2.5-coder-32b-instruct",
        "tester":     "qwen/qwen2.5-coder-32b-instruct",
        "optimizer":  "nvidia/llama-3.1-nemotron-70b-instruct",
    })
    use_agent_models: bool = os.getenv("USE_AGENT_MODELS", "false").lower() == "true"

    # ── Ollama Fallback ────────────────────────────────────────────────────────
    ollama_url: str   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:9b")


@dataclass
class SwarmConfig:
    max_agents:     int = 100
    max_concurrent: int = 16
    agent_timeout:  int = 120
    max_retries:    int = 3
    max_plan_steps: int = 10
    bus_capacity:   int = 1000


model_cfg = ModelConfig()
swarm_cfg = SwarmConfig()
