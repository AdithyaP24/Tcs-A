"""
message_bus.py — Async inter-agent message passing for the swarm
Supports broadcast, direct, and topic-based messaging.
"""
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from config import swarm_cfg


class MsgType(str, Enum):
    TASK      = "task"
    RESULT    = "result"
    STATUS    = "status"
    ERROR     = "error"
    BROADCAST = "broadcast"
    DIRECT    = "direct"


@dataclass
class Message:
    type:      MsgType
    sender:    str
    payload:   dict
    recipient: Optional[str] = None
    msg_id:    str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    def __init__(self, capacity: int = swarm_cfg.bus_capacity):
        self._topics:  Dict[str, List[asyncio.Queue]] = {}
        self._inboxes: Dict[str, asyncio.Queue] = {}
        self._capacity = capacity
        self._lock = asyncio.Lock()
        self._history: List[Message] = []

    async def register(self, agent_id: str) -> asyncio.Queue:
        async with self._lock:
            q: asyncio.Queue = asyncio.Queue(maxsize=self._capacity)
            self._inboxes[agent_id] = q
            return q

    async def unregister(self, agent_id: str) -> None:
        async with self._lock:
            self._inboxes.pop(agent_id, None)

    async def subscribe(self, topic: str, agent_id: str) -> None:
        async with self._lock:
            self._topics.setdefault(topic, [])
            q = self._inboxes.get(agent_id)
            if q and q not in self._topics[topic]:
                self._topics[topic].append(q)

    async def publish(self, msg: Message) -> None:
        self._history.append(msg)
        if len(self._history) > 500:
            self._history = self._history[-500:]
        if msg.type == MsgType.BROADCAST or msg.recipient is None:
            async with self._lock:
                queues = list(self._inboxes.values())
            for q in queues:
                await self._put(q, msg)
        elif msg.recipient in self._inboxes:
            await self._put(self._inboxes[msg.recipient], msg)

    async def _put(self, q: asyncio.Queue, msg: Message) -> None:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            q.put_nowait(msg)

    async def send_task(self, sender, recipient, task):
        await self.publish(Message(type=MsgType.TASK, sender=sender,
                                   recipient=recipient, payload=task))

    async def send_result(self, sender, recipient, result):
        await self.publish(Message(type=MsgType.RESULT, sender=sender,
                                   recipient=recipient, payload=result))

    async def broadcast_status(self, sender, status, extra=None):
        await self.publish(Message(type=MsgType.STATUS, sender=sender,
                                   payload={"status": status, **(extra or {})}))

    def stats(self) -> dict:
        return {
            "registered_agents": len(self._inboxes),
            "topics":            list(self._topics.keys()),
            "history_size":      len(self._history),
            "inbox_depths":      {aid: q.qsize() for aid, q in self._inboxes.items()},
        }


bus = MessageBus()
