"""
llm_client.py — LangChain NVIDIA NIM Client
Supports: NIM Cloud, Local NIM, Ollama fallback
AMD MI300X handles orchestration; NIM provides LLM inference
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from config import ModelConfig, model_cfg

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified async LangChain client for NVIDIA NIM and Ollama.

    NVIDIA NIM models available:
        meta/llama-3.3-70b-instruct
        google/gemma-2-27b-it
        nvidia/llama-3.1-nemotron-70b-instruct
        qwen/qwen2.5-coder-32b-instruct
        deepseek-ai/deepseek-r1
        mistralai/mistral-7b-instruct-v0.3

    Parallelism: asyncio.Semaphore capped at NUM_PARALLEL
    AMD MI300X: manages agent pool + orchestration concurrently
    """

    def __init__(self, config: ModelConfig = model_cfg):
        self.config = config
        self._sem = asyncio.Semaphore(config.num_parallel)
        self._nim_cache: dict[str, ChatNVIDIA] = {}

    async def start(self) -> None:
        logger.info(
            "LLMClient ready | backend=%s model=%s parallel=%d",
            self.config.backend, self.config.model, self.config.num_parallel,
        )

    async def close(self) -> None:
        self._nim_cache.clear()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def generate(
        self,
        prompt: str,
        system: str = "",
        agent_type: str = "planner",
        max_tokens: int = 2048,
    ) -> str:
        async with self._sem:
            if self.config.backend in ("nim_cloud", "nim_local"):
                return await self._nim_generate(prompt, system, agent_type, max_tokens)
            return await self._ollama_generate(prompt, agent_type, max_tokens)

    def _get_nim_llm(self, agent_type: str) -> ChatNVIDIA:
        if agent_type not in self._nim_cache:
            if self.config.use_agent_models:
                model = self.config.agent_models.get(agent_type, self.config.model)
            else:
                model = self.config.model

            temp = self.config.temperatures.get(agent_type, 0.3)

            self._nim_cache[agent_type] = ChatNVIDIA(
                model=model,
                api_key=self.config.nim_api_key,
                base_url=self.config.nim_base_url,
                temperature=temp,
                max_tokens=2048,
            )
            logger.info("Created NIM LLM | agent=%s model=%s temp=%.2f",
                        agent_type, model, temp)
        return self._nim_cache[agent_type]

    async def _nim_generate(
        self, prompt: str, system: str, agent_type: str, max_tokens: int
    ) -> str:
        llm = self._get_nim_llm(agent_type)
        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        try:
            response = await llm.ainvoke(messages)
            return response.content
        except Exception as exc:
            logger.error("NIM error [%s]: %s", agent_type, exc)
            raise

    async def _ollama_generate(
        self, prompt: str, agent_type: str, max_tokens: int
    ) -> str:
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(
            model=self.config.ollama_model,
            base_url=self.config.ollama_url,
            temperature=self.config.temperatures.get(agent_type, 0.3),
            num_predict=max_tokens,
        )
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as exc:
            logger.error("Ollama error [%s]: %s", agent_type, exc)
            raise

    @staticmethod
    def list_nim_models() -> list[str]:
        try:
            models = ChatNVIDIA.get_available_models()
            return [m.id for m in models]
        except Exception as exc:
            logger.error("Could not fetch NIM models: %s", exc)
            return []


_shared: Optional[LLMClient] = None


async def get_client() -> LLMClient:
    global _shared
    if _shared is None:
        _shared = LLMClient()
        await _shared.start()
    return _shared
