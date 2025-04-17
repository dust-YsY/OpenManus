from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import Optional  # 导入Optional类型提示

from pydantic import Field  # 导入Pydantic的Field用于模型属性定义

from app.agent.base import BaseAgent  # 导入BaseAgent基类
from app.llm import LLM  # 导入语言模型类
from app.schema import AgentState, Memory  # 导入Agent状态和内存类

# Reason & Acting
class ReActAgent(BaseAgent, ABC):  # 定义ReActAgent类，继承自BaseAgent和ABC
    name: str  # Agent名称
    description: Optional[str] = None  # 可选的Agent描述

    system_prompt: Optional[str] = None  # 可选的系统提示
    next_step_prompt: Optional[str] = None  # 可选的下一步提示

    llm: Optional[LLM] = Field(default_factory=LLM)  # 语言模型实例，默认创建新实例
    memory: Memory = Field(default_factory=Memory)  # 内存实例，默认创建新实例
    state: AgentState = AgentState.IDLE  # Agent状态，默认为空闲状态

    max_steps: int = 10  # 最大步骤数，默认为10
    current_step: int = 0  # 当前步骤数，默认为0

    @abstractmethod
    async def think(self) -> bool:
        """处理当前状态并决定下一步行动"""

    @abstractmethod
    async def act(self) -> str:
        """执行已决定的行动"""

    async def step(self) -> str:
        """执行单个步骤：思考和行动。"""
        should_act = await self.think()  # 调用think方法，决定是否需要行动
        if not should_act:  # 如果不需要行动
            return "Thinking complete - no action needed"  # 返回思考完成的信息
        return await self.act()  # 否则执行行动并返回结果
