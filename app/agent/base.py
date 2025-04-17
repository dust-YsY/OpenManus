from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message


class BaseAgent(BaseModel, ABC):
    """Agent基类，用于管理agent状态和执行。

    提供状态转换、内存管理和基于步骤的执行循环的基础功能。
    子类必须实现`step`方法。
    """

    # 核心属性
    name: str = Field(..., description="Unique name of the agent")  # Agent的唯一名称
    description: Optional[str] = Field(None, description="Optional agent description")  # 可选的Agent描述

    # 提示词
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"  # 系统级指令提示
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"  # 用于确定下一步行动的提示
    )

    # 依赖项
    llm: LLM = Field(default_factory=LLM, description="Language model instance")  # 语言模型实例
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")  # Agent的内存存储
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"  # 当前Agent状态
    )

    # 执行控制
    max_steps: int = Field(default=10, description="Maximum steps before termination")  # 终止前的最大步骤数
    current_step: int = Field(default=0, description="Current step in execution")  # 执行中的当前步骤

    duplicate_threshold: int = 2  # 重复内容检测阈值

    class Config:
        arbitrary_types_allowed = True  # 允许任意类型
        extra = "allow"  # 允许额外字段，增加子类灵活性

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """如果未提供默认设置，则使用默认设置初始化agent。"""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """用于安全Agent状态转换的上下文管理器。

        Args:
            new_state: 在上下文期间要转换到的状态。

        Yields:
            None: 允许在新状态下执行。

        Raises:
            ValueError: 如果new_state无效。
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # 失败时转换到ERROR状态
            raise e
        finally:
            self.state = previous_state  # 恢复到之前的状态

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """向Agent的内存添加消息。

        Args:
            role: 消息发送者的角色（用户、系统、助手、工具）。
            content: 消息内容。
            base64_image: 可选的base64编码图像。
            **kwargs: 额外参数（例如，工具消息的tool_call_id）。

        Raises:
            ValueError: 如果角色不受支持。
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # 根据角色创建具有适当参数的消息
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """异步执行Agent的主循环。

        Args:
            request: 可选的初始用户请求进行处理。

        Returns:
            总结执行结果的字符串。

        Raises:
            RuntimeError: 如果Agent在开始时不处于IDLE状态。
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step()

                # 检查是否陷入循环状态
                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
        await SANDBOX_CLIENT.cleanup()
        return "\n".join(results) if results else "No steps executed"

    @abstractmethod
    async def step(self) -> str:
        """执行Agent工作流中的单个步骤。

        必须由子类实现以定义特定行为。
        """

    def handle_stuck_state(self):
        """通过添加提示来改变策略以处理卡住状态"""
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """通过检测重复内容来检查Agent是否陷入循环"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # 计算相同内容出现的次数
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """从Agent的内存中检索消息列表。"""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """设置Agent内存中的消息列表。"""
        self.memory.messages = value
