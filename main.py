from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import json
import os
import sys
import time

from config import model_cfg, swarm_cfg
from llm_client import LLMClient
from message_bus import bus
from swarm import SwarmOrchestrator

DEMO_GOAL = (
    "Design and implement a GPU-accelerated real-time anomaly detection system "
    "for network intrusion using Python. Include: data ingestion pipeline, "
    "Isolation Forest ML model, FastAPI REST endpoint, and full pytest test suite. "
    "Optimize all hot paths for AMD MI300X ROCm."
)


def progress_cb(event: str, data: dict) -> None:
    if event == "planning":
        print(f"\n[PLANNER] Goal: {data.get('goal', '')[:70]}...")

    elif event == "plan_ready":
        print(f"\n[PLAN] '{data['plan_title']}' → {data['num_tasks']} tasks:")
        for t in data.get("tasks", []):
            print(f"   [{t['agent_type']:10}] {t['title']}")

    elif event == "dispatch":
        tasks = ", ".join(data.get("dispatching", []))
        print(f"\n[DISPATCH iter-{data['iteration']}] {tasks}")

    elif event == "task_done":
        icon = "✓" if data.get("status") == "OK" else "✗"
        err  = f"  ERROR: {data['error']}" if data.get("error") else ""
        print(f"   {icon} [{data.get('agent_type', ''):10}] "
              f"{data.get('title', '')} ({data.get('elapsed', 0):.1f}s){err}")

    elif event == "done":
        print(f"\n[SWARM] Completed in {data.get('elapsed_s', 0):.1f}s\n")


async def run_swarm(goal: str) -> dict:
    async with LLMClient() as llm:
        orch   = SwarmOrchestrator(llm=llm, bus=bus, progress_cb=progress_cb)
        report = await orch.run(goal)
        await orch.shutdown()
    return report


async def benchmark(goal: str, n: int) -> None:
    print(f"\n[BENCHMARK] {n} concurrent swarms | AMD MI300X + NVIDIA NIM\n")
    start = time.time()
    async with LLMClient() as llm:
        results = await asyncio.gather(
            *[SwarmOrchestrator(llm=llm, bus=bus).run(goal) for _ in range(n)],
            return_exceptions=True,
        )
    elapsed = time.time() - start
    ok = sum(1 for r in results if isinstance(r, dict))
    print(f"\n[BENCHMARK] {ok}/{n} succeeded | "
          f"{elapsed:.1f}s total | {elapsed / n:.2f}s avg")


def list_models() -> None:
    print("\nFetching available NVIDIA NIM models...")
    models = LLMClient.list_nim_models()
    if models:
        print(f"\n{len(models)} models available:\n")
        for m in sorted(models):
            print(f"  {m}")
    else:
        print("Could not fetch models. Check your NVIDIA_API_KEY.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="AMD MI300X + NVIDIA NIM Multi-Agent Swarm | TCS Hackathon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo
  python main.py --goal "Build a FastAPI fraud detection service with ML"
  python main.py --list-models
  python main.py --benchmark 10 --demo
  python main.py --demo --output report.json
  python main.py --backend ollama --model gemma3:9b --goal "..."
        """,
    )
    p.add_argument("--goal",        type=str,  help="Goal for the swarm to accomplish")
    p.add_argument("--demo",        action="store_true", help="Run preset demo goal")
    p.add_argument("--benchmark",   type=int,  metavar="N",
                   help="Run N concurrent swarms (MI300X stress test)")
    p.add_argument("--list-models", action="store_true",
                   help="List all available NVIDIA NIM models")
    p.add_argument("--backend",     choices=["nim_cloud", "nim_local", "ollama"],
                   default=model_cfg.backend)
    p.add_argument("--model",       default=model_cfg.model,
                   help="NIM model e.g. meta/llama-3.3-70b-instruct")
    p.add_argument("--api-key",     default=None,
                   help="NVIDIA API key (or set NVIDIA_API_KEY env var)")
    p.add_argument("--nim-url",     default=None,
                   help="Local NIM endpoint e.g. http://localhost:8000/v1")
    p.add_argument("--parallel",    type=int,  default=model_cfg.num_parallel,
                   help="Max concurrent NIM API calls (default 16)")
    p.add_argument("--output",      type=str,  help="Save JSON report to file")
    p.add_argument("--verbose",     action="store_true")
    args = p.parse_args()

    # Apply CLI overrides to config
    model_cfg.backend      = args.backend
    model_cfg.model        = args.model
    model_cfg.num_parallel = args.parallel
    if args.api_key:
        model_cfg.nim_api_key  = args.api_key
    if args.nim_url:
        model_cfg.nim_base_url = args.nim_url

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║   AMD MI300X  ·  NVIDIA NIM  ·  Multi-Agent Swarm            ║
║   TCS × AMD Hackathon                                         ║
║   Backend  : {args.backend:<12}  Parallel : {model_cfg.num_parallel:<4}               ║
║   Model    : {args.model:<48}  ║
╚═══════════════════════════════════════════════════════════════╝
""")

    if args.list_models:
        list_models()
        return

    if args.benchmark:
        asyncio.run(benchmark(args.goal or DEMO_GOAL, args.benchmark))
        return

    goal = args.goal or (DEMO_GOAL if args.demo else None)
    if not goal:
        print("Error: provide --goal <text> or --demo\n")
        p.print_help()
        sys.exit(1)

    report = asyncio.run(run_swarm(goal))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved → {args.output}")
    else:
        print(json.dumps({
            "plan_title": report.get("plan_title"),
            "elapsed_s":  report["elapsed_s"],
            "succeeded":  report["succeeded"],
            "failed":     report["failed"],
        }, indent=2))


if __name__ == "__main__":
    main()
