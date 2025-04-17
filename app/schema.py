from enum import Enum
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field

"""这段代码定义了一个基于 Pydantic 的聊天对话系统模型，主要用于处理多角色（用户、助手、系统、工具）的对话消息、工具调用和记忆管理。"""
class Role(str, Enum):
    """消息角色选项"""

    SYSTEM = "system"  # 系统消息
    USER = "user"  # 用户消息
    ASSISTANT = "assistant"  # 助手消息
    TOOL = "tool"  # 工具消息


ROLE_VALUES = tuple(role.value for role in Role)  # 所有角色值的元组
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore  # 角色类型定义


class ToolChoice(str, Enum):
    """工具选择选项"""

    NONE = "none"  # 不使用工具
    AUTO = "auto"  # 自动决定是否使用工具
    REQUIRED = "required"  # 必须使用工具


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)  # 所有工具选择值的元组
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore  # 工具选择类型定义


class AgentState(str, Enum):
    """Agent执行状态"""

    IDLE = "IDLE"  # 空闲状态
    RUNNING = "RUNNING"  # 运行中状态
    FINISHED = "FINISHED"  # 已完成状态
    ERROR = "ERROR"  # 错误状态


class Function(BaseModel):
    """函数模型，表示一个工具函数调用"""
    name: str  # 函数名称
    arguments: str  # 函数参数（JSON字符串）


class ToolCall(BaseModel):
    """表示消息中的工具/函数调用"""

    id: str  # 工具调用的唯一ID
    type: str = "function"  # 工具类型，默认为函数
    function: Function  # 函数信息


class Message(BaseModel):
    """表示对话中的一条消息"""

    role: ROLE_TYPE = Field(...)  # type: ignore  # 消息角色（必填）
    content: Optional[str] = Field(default=None)  # 消息内容（可选）
    tool_calls: Optional[List[ToolCall]] = Field(default=None)  # 工具调用列表（可选）
    name: Optional[str] = Field(default=None)  # 名称，用于工具消息（可选）
    tool_call_id: Optional[str] = Field(default=None)  # 工具调用ID，用于工具消息（可选）
    base64_image: Optional[str] = Field(default=None)  # Base64编码的图像（可选）

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """将消息转换为字典格式"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        return message

    @classmethod
    def user_message(
        cls, content: str, base64_image: Optional[str] = None
    ) -> "Message":
        """创建一个用户消息"""
        return cls(role=Role.USER, content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """创建一个系统消息"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls, content: Optional[str] = None, base64_image: Optional[str] = None
    ) -> "Message":
        """创建一个助手消息"""
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image)

    @classmethod
    def tool_message(
        cls, content: str, name, tool_call_id: str, base64_image: Optional[str] = None
    ) -> "Message":
        """创建一个工具消息"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        base64_image: Optional[str] = None,
        **kwargs,
    ):
        """从原始工具调用创建ToolCallsMessage

        Args:
            tool_calls: 来自LLM的原始工具调用
            content: 可选的消息内容
            base64_image: 可选的base64编码图像
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )


class Memory(BaseModel):
    """内存类，用于存储和管理消息历史"""
    messages: List[Message] = Field(default_factory=list)  # 消息列表
    max_messages: int = Field(default=100)  # 最大消息数量

    def add_message(self, message: Message) -> None:
        """添加一条消息到内存"""
        self.messages.append(message)
        # 可选：实现消息数量限制
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """添加多条消息到内存"""
        self.messages.extend(messages)
        # 可选：实现消息数量限制
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """清空所有消息"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """获取最近的n条消息"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """将消息转换为字典列表"""
        return [msg.to_dict() for msg in self.messages]
