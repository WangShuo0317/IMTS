"""
IMTS 统一消息格式定义 (Async Version)

消息类型:
- STAGE_START: 阶段开始
- STAGE_END: 阶段结束
- AGENT_THOUGHT: 智能体思考 (打字机效果)
- TOOL_CALL: 工具调用
- TRAINING_LOSS: 训练损失更新 (折线图)
- CHAT_MESSAGE: 聊天消息 (聊天气泡)
- JOB_STATUS: 任务状态更新
- ERROR: 错误信息
"""

import json
import time
import asyncio
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Any, Optional, List

class MessageType(str, Enum):
    STAGE_START = "STAGE_START"
    STAGE_END = "STAGE_END"
    AGENT_THOUGHT = "AGENT_THOUGHT"
    TOOL_CALL = "TOOL_CALL"
    TRAINING_LOSS = "TRAINING_LOSS"
    CHAT_MESSAGE = "CHAT_MESSAGE"
    JOB_STATUS = "JOB_STATUS"
    ERROR = "ERROR"

class Stage(str, Enum):
    INIT = "INIT"
    DATA_OPTIMIZATION = "DATA_OPTIMIZATION"
    TRAINING = "TRAINING"
    EVALUATION = "EVALUATION"
    COMPLETED = "COMPLETED"

class ChatRole(str, Enum):
    FACT_EVALUATOR = "FACT_EVALUATOR"
    LOGIC_EVALUATOR = "LOGIC_EVALUATOR"
    ARBITER = "ARBITER"

@dataclass
class IMTSMessage:
    """统一消息格式"""
    msg_type: str
    job_id: str
    stage: str
    timestamp: int
    progress: int
    data: dict
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
    
    @staticmethod
    def from_json(json_str: str) -> 'IMTSMessage':
        d = json.loads(json_str)
        return IMTSMessage(
            msg_type=d['msg_type'],
            job_id=d['job_id'],
            stage=d['stage'],
            timestamp=d['timestamp'],
            progress=d['progress'],
            data=d['data']
        )

class MessageBuilder:
    """异步消息构建器"""

    def __init__(self, job_id: str, redis_client=None, sync_redis_client=None, channel: str = "job_events"):
        self.job_id = job_id
        self.redis_client = redis_client  # 异步客户端，用于 rpush
        self.sync_redis_client = sync_redis_client  # 同步客户端，用于 publish (Spring ReactiveRedisTemplate 兼容)
        self.channel = channel
        self.current_stage = Stage.INIT
        self.progress = 0

    def set_redis(self, redis_client):
        self.redis_client = redis_client

    def set_sync_redis(self, sync_redis_client):
        self.sync_redis_client = sync_redis_client
    
    async def _emit(self, msg_type: str, data: dict, progress: int = None):
        msg = IMTSMessage(
            msg_type=msg_type,
            job_id=self.job_id,
            stage=self.current_stage.value,
            timestamp=int(time.time() * 1000),
            progress=progress if progress is not None else self.progress,
            data=data
        )

        if self.redis_client or self.sync_redis_client:
            channel = f"{self.channel}:{self.job_id}"
            msg_json = msg.to_json()
            # 使用同步客户端发布到 Pub/Sub (解决与 Spring ReactiveRedisTemplate 的兼容性问题)
            if self.sync_redis_client:
                self.sync_redis_client.publish(channel, msg_json)
                print(f"[DEBUG] Published to {channel} via sync_redis (msg_type={msg_type})")
            else:
                print(f"[DEBUG] sync_redis_client is None! channel={channel}")
            # 使用异步客户端写入 Redis List (历史消息)
            if self.redis_client:
                await self.redis_client.rpush(f"imts_messages:{self.job_id}", msg_json)
                await self.redis_client.expire(f"imts_messages:{self.job_id}", 3600)
        else:
            print(f"[DEBUG] No Redis client! redis_client={self.redis_client}, sync_redis_client={self.sync_redis_client}")

        return msg
    
    async def stage_start(self, stage: str):
        """阶段开始"""
        self.current_stage = Stage(stage)
        return await self._emit(MessageType.STAGE_START.value, {
            "stage": stage,
            "message": f"Starting {stage.replace('_', ' ')}..."
        }, progress=0)
    
    async def stage_end(self, stage: str, summary: dict = None):
        """阶段结束"""
        self.progress = 100
        return await self._emit(MessageType.STAGE_END.value, {
            "stage": stage,
            "summary": summary or {}
        }, progress=100)
    
    async def agent_thought(self, agent: str, thought: str, progress: int = None):
        """智能体思考 - 用于打字机效果"""
        if progress is not None:
            self.progress = progress
        return await self._emit(MessageType.AGENT_THOUGHT.value, {
            "agent": agent,
            "thought": thought,
            "is_complete": thought.endswith('.') or thought.endswith('!') or thought.endswith('?')
        })
    
    async def tool_call(self, tool_name: str, args: dict, result: dict = None):
        """工具调用"""
        return await self._emit(MessageType.TOOL_CALL.value, {
            "tool_name": tool_name,
            "args": args,
            "result": result
        })
    
    async def training_loss(self, epoch: int, step: int, loss: float, loss_history: List[float]):
        """训练损失更新 - 用于折线图"""
        return await self._emit(MessageType.TRAINING_LOSS.value, {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "loss_history": loss_history
        })
    
    async def chat_message(self, role: str, speaker: str, message: str, is_streaming: bool = False):
        """聊天消息 - 用于聊天气泡"""
        return await self._emit(MessageType.CHAT_MESSAGE.value, {
            "role": role,
            "speaker": speaker,
            "message": message,
            "is_streaming": is_streaming
        })
    
    async def job_status(self, status: str, message: str = ""):
        """任务状态更新"""
        return await self._emit(MessageType.JOB_STATUS.value, {
            "status": status,
            "message": message
        })
    
    async def error(self, error_message: str, details: dict = None):
        """错误信息"""
        return await self._emit(MessageType.ERROR.value, {
            "error_message": error_message,
            "details": details or {}
        })