"""
swarm.py — AMD MI300X Parallel Swarm Orchestrator
Uses NVIDIA NIM via LangChain for LLM inference

Architecture:
    user goal → PlannerAgent → Task DAG
                                  ↓ (parallel, GPU-bounded)
    [ResearcherAgent] [CoderAgent] [TesterAgent] [OptimizerAgent]
                                  ↓
                           Final JSON Report
"""
import asyncio
import json
import logging
import time
from typing import Callable, Dict, List, Optional

from config import swarm_cfg, model_cfg
from llm_client import LLMClient
from message_bus import MessageBus, bus as global_bus
from agents import (
    BaseAgent, SubTask, AgentState,
    PlannerAgent, create_agent, AGENT_REGISTRY,
)

logger = logging.getLogger(__name__)


class AgentPool:
    """
    Manages a pool of reusable agents per role.
    GPU concurrency is capped at NUM_PARALLEL via asyncio.Semaphore.
    """

    def __init__(self, llm: LLMClient, bus: MessageBus):
        self._llm   = llm
        self._bus   = bus
        self._pools = {role: asyncio.Queue() for role in AGENT_REGISTRY}
        self._all_agents: List[BaseAgent] = []
        self._sem   = asyncio.Semaphore(model_cfg.num_parallel)
        self._total = 0

    async def _ensure_agent(self, role: str) -> BaseAgent:
        try:
            return self._pools[role].get_nowait()
        except asyncio.QueueEmpty:
            if self._total < swarm_cfg.max_agents:
                agent = create_agent(role, self._llm, self._bus)
                await agent.start()
                self._all_agents.append(agent)
                self._total += 1
                logger.debug("Created agent %s (total=%d)", agent.agent_id, self._total)
                return agent
            return await self._pools[role].get()

    async def run_task(self, subtask: SubTask) -> SubTask:
        async with self._sem:
            agent = await self._ensure_agent(subtask.agent_type)
            agent.state    = AgentState.RUNNING
            subtask.started_at = time.time()
            try:
                result = await asyncio.wait_for(
                    agent.process(subtask),
                    timeout=swarm_cfg.agent_timeout,
                )
                subtask.result = result
                agent.tasks_done += 1
            except asyncio.TimeoutError:
                subtask.error = f"Timeout after {swarm_cfg.agent_timeout}s"
                agent.tasks_failed += 1
            except Exception as exc:  # noqa: BLE001
                subtask.error = str(exc)
                agent.tasks_failed += 1
                logger.error("[%s] task %s failed: %s",
                             agent.agent_id, subtask.task_id, exc)
            finally:
                subtask.ended_at = time.time()
                agent.state      = AgentState.IDLE
                await self._pools[subtask.agent_type].put(agent)
        return subtask

    async def shutdown(self) -> None:
        for agent in self._all_agents:
            await agent.stop()

    def stats(self) -> dict:
        return {
            "total_agents": self._total,
            "pool_depths":  {r: q.qsize() for r, q in self._pools.items()},
            "agents":       [a.stats() for a in self._all_agents],
        }


class SwarmOrchestrator:
    """
    High-level orchestrator:
    1. PlannerAgent decomposes the goal into a task DAG
    2. DAG is executed in parallel respecting dependencies
    3. Results are assembled into a final report
    """

    def __init__(
        self,
        llm: LLMClient,
        bus: MessageBus = global_bus,
        progress_cb: Optional[Callable[[str, dict], None]] = None,
    ):
        self._llm     = llm
        self._bus     = bus
        self._pool    = AgentPool(llm, bus)
        self._cb      = progress_cb or (lambda e, d: None)
        self._planner = PlannerAgent(llm=llm, bus=bus, agent_id="planner_master")

    async def run(self, goal: str) -> dict:
        start_ts = time.time()
        logger.info("Swarm started | goal: %s", goal[:80])

        self._emit("planning", {"goal": goal})
        await self._planner.start()

        plan_task      = SubTask.new("Master Plan", goal, "planner")
        plan_json_str  = await self._planner.process(plan_task)
        plan           = json.loads(plan_json_str)
        subtasks       = self._build_subtasks(plan)

        self._emit("plan_ready", {
            "plan_title": plan.get("plan_title", "Execution Plan"),
            "num_tasks":  len(subtasks),
            "tasks": [{"id": t.task_id, "title": t.title,
                       "agent_type": t.agent_type} for t in subtasks],
        })

        results  = await self._execute_dag(subtasks)
        elapsed  = time.time() - start_ts
        report   = self._assemble_report(goal, plan, results, elapsed)

        self._emit("done", {"elapsed_s": round(elapsed, 2)})
        logger.info("Swarm finished in %.1fs | %d tasks", elapsed, len(results))
        return report

    async def _execute_dag(self, subtasks: List[SubTask]) -> Dict[str, SubTask]:
        done:    Dict[str, SubTask] = {}
        pending: List[SubTask]      = list(subtasks)
        iteration = 0

        while pending:
            iteration += 1
            ready = [t for t in pending
                     if all(dep in done for dep in t.depends_on)]
            if not ready:
                logger.error("DAG deadlock at iteration %d", iteration)
                break

            pending = [t for t in pending if t not in ready]
            self._emit("dispatch", {
                "iteration":   iteration,
                "dispatching": [t.title for t in ready],
            })

            finished = await asyncio.gather(
                *[self._pool.run_task(t) for t in ready],
                return_exceptions=False,
            )

            for task in finished:
                done[task.task_id] = task
                self._emit("task_done", {
                    "status":     "OK" if task.result else "FAIL",
                    "task_id":    task.task_id,
                    "title":      task.title,
                    "agent_type": task.agent_type,
                    "elapsed":    round(task.ended_at - task.started_at, 2),
                    "error":      task.error,
                })

        return done

    @staticmethod
    def _build_subtasks(plan: dict) -> List[SubTask]:
        id_map:   Dict[str, str] = {}
        subtasks: List[SubTask]  = []
        for item in plan.get("subtasks", []):
            t = SubTask.new(item["title"], item["description"], item["agent_type"])
            id_map[item["id"]] = t.task_id
            subtasks.append(t)
        for item, task in zip(plan.get("subtasks", []), subtasks):
            task.depends_on = [id_map[d] for d in item.get("depends_on", [])
                               if d in id_map]
        return subtasks

    @staticmethod
    def _assemble_report(goal, plan, results, elapsed) -> dict:
        succeeded = [t for t in results.values() if t.result]
        failed    = [t for t in results.values() if t.error]
        return {
            "goal":        goal,
            "plan_title":  plan.get("plan_title", ""),
            "elapsed_s":   round(elapsed, 2),
            "total_tasks": len(results),
            "succeeded":   len(succeeded),
            "failed":      len(failed),
            "results": [
                {
                    "task_id":    t.task_id,
                    "title":      t.title,
                    "agent_type": t.agent_type,
                    "result":     t.result,
                    "error":      t.error,
                    "duration_s": round(t.ended_at - t.started_at, 2),
                }
                for t in sorted(results.values(), key=lambda x: x.started_at)
            ],
        }

    def _emit(self, event: str, data: dict) -> None:
        try:
            self._cb(event, data)
        except Exception:  # noqa: BLE001
            pass

    async def shutdown(self) -> None:
        await self._pool.shutdown()
        await self._planner.stop()

    def pool_stats(self) -> dict:
        return self._pool.stats()
