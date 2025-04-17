from typing import Optional  # 导入Optional类型用于可选参数

from pydantic import Field, model_validator  # 导入Pydantic的字段定义和模型验证器

from app.agent.browser import BrowserContextHelper  # 导入浏览器上下文助手
from app.agent.toolcall import ToolCallAgent  # 导入工具调用代理基类
from app.config import config  # 导入配置模块
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT  # 导入Manus代理的提示词模板
from app.tool import Terminate, ToolCollection  # 导入工具集合和终止工具
from app.tool.browser_use_tool import BrowserUseTool  # 导入浏览器使用工具
from app.tool.knowledge_base import KnowledgeBaseTool  # 导入知识库工具
from app.tool.python_execute import PythonExecute  # 导入Python执行工具
from app.tool.str_replace_editor import StrReplaceEditor  # 导入字符串替换编辑器工具


class Manus(ToolCallAgent):
    """A versatile general-purpose agent."""

    name: str = "Manus"  # 代理的名称
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"  # 代理的描述
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)  # 系统提示词，使用工作空间根目录格式化
    next_step_prompt: str = NEXT_STEP_PROMPT  # 下一步行动提示词

    max_observe: int = 10000  # 最大观察次数限制
    max_steps: int = 20  # 最大执行步骤限制

    # 添加通用工具到工具集合
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(  # 使用默认工厂函数创建工具集合
            PythonExecute(),  # Python执行工具
            BrowserUseTool(),  # 浏览器使用工具
            StrReplaceEditor(),  # 字符串替换编辑器工具
            KnowledgeBaseTool(),  # 知识库工具
            Terminate()  # 终止工具
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])  # 特殊工具名称列表，默认为终止工具

    browser_context_helper: Optional[BrowserContextHelper] = None  # 可选的浏览器上下文助手

    @model_validator(mode="after")  # 使用Pydantic的模型验证器，在模型验证后执行
    def initialize_helper(self) -> "Manus":
        """初始化浏览器上下文助手"""
        self.browser_context_helper = BrowserContextHelper(self)  # 创建浏览器上下文助手实例
        return self  # 返回自身以支持链式调用

    async def think(self) -> bool:
        """处理当前状态并决定下一步行动"""
        original_prompt = self.next_step_prompt  # 保存原始提示词
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []  # 获取最近3条消息
        # 检查最近的消息中是否使用了浏览器工具
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use:  # 如果正在使用浏览器
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()  # 使用浏览器上下文格式化提示词
            )

        result = await super().think()  # 调用父类的think方法

        # 恢复原始提示词
        self.next_step_prompt = original_prompt

        return result  # 返回思考结果

    async def cleanup(self):
        """清理Manus代理的资源"""
        if self.browser_context_helper:  # 如果存在浏览器上下文助手
            await self.browser_context_helper.cleanup_browser()  # 清理浏览器资源
