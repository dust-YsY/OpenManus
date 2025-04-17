from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import Any, Dict, Optional  # 导入类型注解

from pydantic import BaseModel, Field  # 导入Pydantic模型和字段定义


class BaseTool(ABC, BaseModel):  # 基础工具类，继承自抽象基类和Pydantic模型
    """Base class for all tools in the system.

    This class defines the common interface and behavior for all tools.
    It combines ABC for interface definition and Pydantic for data validation.
    """

    name: str  # 工具名称
    description: str  # 工具描述
    parameters: Optional[dict] = None  # 工具参数，可选

    class Config:  # Pydantic配置类
        arbitrary_types_allowed = True  # 允许任意类型

    async def __call__(self, **kwargs) -> Any:  # 使实例可调用
        """Execute the tool with given parameters.

        This method allows tools to be called like functions.
        """
        return await self.execute(**kwargs)  # 调用execute方法执行工具

    @abstractmethod  # 抽象方法装饰器
    async def execute(self, **kwargs) -> Any:  # 抽象执行方法
        """Execute the tool with given parameters.

        This method must be implemented by all concrete tool classes.
        """

    def to_param(self) -> Dict:  # 转换为函数调用格式
        """Convert tool to function call format.

        Returns a dictionary representing the tool in function call format.
        """
        return {
            "type": "function",  # 类型为函数
            "function": {
                "name": self.name,  # 函数名称
                "description": self.description,  # 函数描述
                "parameters": self.parameters,  # 函数参数
            },
        }


class ToolResult(BaseModel):  # 工具执行结果类
    """Represents the result of a tool execution.

    This class encapsulates the output, error, and additional metadata
    from a tool execution.
    """

    output: Any = Field(default=None)  # 工具输出结果
    error: Optional[str] = Field(default=None)  # 错误信息
    base64_image: Optional[str] = Field(default=None)  # Base64编码的图片
    system: Optional[str] = Field(default=None)  # 系统信息

    class Config:  # Pydantic配置类
        arbitrary_types_allowed = True  # 允许任意类型

    def __bool__(self):  # 布尔值转换
        """Convert to boolean based on whether any field has a value."""
        return any(getattr(self, field) for field in self.__fields__)  # 检查是否有任何字段有值

    def __add__(self, other: "ToolResult"):  # 结果合并
        """Combine two tool results.

        Args:
            other: Another ToolResult to combine with

        Returns:
            A new ToolResult with combined fields
        """
        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):  # 字段合并辅助函数
            """Combine two fields with optional concatenation."""
            if field and other_field:  # 如果两个字段都有值
                if concatenate:  # 如果需要连接
                    return field + other_field  # 连接字段
                raise ValueError("Cannot combine tool results")  # 否则抛出错误
            return field or other_field  # 返回非空字段

        return ToolResult(  # 返回合并后的结果
            output=combine_fields(self.output, other.output),  # 合并输出
            error=combine_fields(self.error, other.error),  # 合并错误
            base64_image=combine_fields(self.base64_image, other.base64_image, False),  # 合并图片
            system=combine_fields(self.system, other.system),  # 合并系统信息
        )

    def __str__(self):  # 字符串表示
        """Convert to string representation."""
        return f"Error: {self.error}" if self.error else self.output  # 如果有错误显示错误，否则显示输出

    def replace(self, **kwargs):  # 字段替换
        """Returns a new ToolResult with the given fields replaced.

        Args:
            **kwargs: Fields to replace

        Returns:
            A new ToolResult with updated fields
        """
        return type(self)(**{**self.dict(), **kwargs})  # 创建新实例并更新字段


class CLIResult(ToolResult):  # CLI结果类
    """A ToolResult that can be rendered as a CLI output.

    This class is specifically designed for command-line interface output.
    """


class ToolFailure(ToolResult):  # 工具失败类
    """A ToolResult that represents a failure.

    This class is used to represent failed tool executions.
    """
