import asyncio  # 导入异步IO库
import json  # 导入JSON处理库
from typing import Any, List, Optional, Union  # 导入类型提示

from pydantic import Field  # 导入Pydantic的Field用于模型属性定义

from app.agent.react import ReActAgent  # 导入ReActAgent基类
from app.exceptions import TokenLimitExceeded  # 导入Token限制异常
from app.logger import logger  # 导入日志记录器
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT  # 导入提示词常量
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice  # 导入相关模型和常量
from app.tool import CreateChatCompletion, Terminate, ToolCollection  # 导入工具类


TOOL_CALL_REQUIRED = "Tool calls required but none provided"  # 定义工具调用必需的错误消息


class ToolCallAgent(ReActAgent):
    """处理工具/函数调用的基础Agent类，具有增强的抽象功能"""

    name: str = "toolcall"  # Agent名称
    description: str = "an agent that can execute tool calls."  # Agent描述

    system_prompt: str = SYSTEM_PROMPT  # 系统提示
    next_step_prompt: str = NEXT_STEP_PROMPT  # 下一步提示

    available_tools: ToolCollection = ToolCollection(  # 可用工具集合
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore  # 工具选择模式，默认为自动
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])  # 特殊工具名称列表

    tool_calls: List[ToolCall] = Field(default_factory=list)  # 工具调用列表
    _current_base64_image: Optional[str] = None  # 当前base64编码图像

    max_steps: int = 30  # 最大步骤数
    max_observe: Optional[Union[int, bool]] = None  # 最大观察长度限制

    async def think(self) -> bool:
        """处理当前状态并使用工具决定下一步操作"""
        if self.next_step_prompt:  # 如果有下一步提示
            user_msg = Message.user_message(self.next_step_prompt)  # 创建用户消息
            self.messages += [user_msg]  # 添加到消息列表

        try:
            # 获取带工具选项的响应
            response = await self.llm.ask_tool(
                messages=self.messages,  # 当前消息列表
                system_msgs=(  # 系统消息
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),  # 工具参数
                tool_choice=self.tool_choices,  # 工具选择模式
            )
        except ValueError:  # 捕获值错误
            raise
        except Exception as e:  # 捕获其他异常
            # 检查是否是包含TokenLimitExceeded的RetryError
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED  # 将状态设置为已完成
                return False
            raise

        self.tool_calls = tool_calls = (  # 设置工具调用
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""  # 获取响应内容

        # 记录响应信息
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:  # 如果有工具调用
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

        try:
            if response is None:  # 如果没有响应
                raise RuntimeError("No response received from the LLM")

            # 处理不同的工具选择模式
            if self.tool_choices == ToolChoice.NONE:  # 如果是NONE模式
                if tool_calls:  # 如果尝试使用工具但工具不可用
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:  # 如果有内容
                    self.memory.add_message(Message.assistant_message(content))  # 添加助手消息
                    return True
                return False

            # 创建并添加助手消息
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls  # 如果有工具调用
                else Message.assistant_message(content)  # 否则创建普通助手消息
            )
            self.memory.add_message(assistant_msg)  # 添加到内存

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:  # 如果需要工具调用但没有提供
                return True  # 将在act()中处理

            # 对于'auto'模式，如果没有命令但有内容则继续
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)  # 根据内容是否存在决定是否继续

            return bool(self.tool_calls)  # 返回是否有工具调用
        except Exception as e:  # 捕获异常
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """执行工具调用并处理结果"""
        if not self.tool_calls:  # 如果没有工具调用
            if self.tool_choices == ToolChoice.REQUIRED:  # 如果需要工具调用
                raise ValueError(TOOL_CALL_REQUIRED)  # 抛出错误

            # 如果没有工具调用，返回最后一条消息内容
            return self.messages[-1].content or "No content or commands to execute"

        results = []  # 结果列表
        for command in self.tool_calls:  # 遍历每个工具调用
            # 为每个工具调用重置base64_image
            self._current_base64_image = None

            result = await self.execute_tool(command)  # 执行工具

            if self.max_observe:  # 如果设置了最大观察长度
                result = result[: self.max_observe]  # 截断结果

            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            # 将工具响应添加到内存
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)  # 添加到内存
            results.append(result)  # 添加到结果列表

        return "\n\n".join(results)  # 合并并返回所有结果

    async def execute_tool(self, command: ToolCall) -> str:
        """执行单个工具调用，具有健壮的错误处理功能"""
        if not command or not command.function or not command.function.name:  # 检查命令格式是否有效
            return "Error: Invalid command format"

        name = command.function.name  # 获取函数名称
        if name not in self.available_tools.tool_map:  # 如果工具不存在
            return f"Error: Unknown tool '{name}'"

        try:
            # 解析参数
            args = json.loads(command.function.arguments or "{}")

            # 执行工具
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)  # 执行工具

            # 处理特殊工具
            await self._handle_special_tool(name=name, result=result)

            # 检查结果是否包含base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                # 存储base64_image以供后续在tool_message中使用
                self._current_base64_image = result.base64_image

                # 格式化显示结果
                observation = (
                    f"Observed output of cmd `{name}` executed:\n{str(result)}"
                    if result
                    else f"Cmd `{name}` completed with no output"
                )
                return observation

            # 格式化显示结果（标准情况）
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation  # 返回观察结果
        except json.JSONDecodeError:  # 捕获JSON解析错误
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:  # 捕获其他异常
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """处理特殊工具执行和状态变化"""
        if not self._is_special_tool(name):  # 如果不是特殊工具
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):  # 如果应该结束执行
            # 将agent状态设置为已完成
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """确定工具执行是否应该结束agent"""
        return True  # 默认应该结束执行

    def _is_special_tool(self, name: str) -> bool:
        """检查工具名称是否在特殊工具列表中"""
        return name.lower() in [n.lower() for n in self.special_tool_names]  # 不区分大小写比较

    async def cleanup(self):
        """清理agent的工具使用的资源"""
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():  # 遍历所有工具
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(  # 如果工具有cleanup方法
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()  # 清理工具资源
                except Exception as e:  # 捕获清理过程中的异常
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None) -> str:
        """运行agent并在完成后进行清理"""
        try:
            return await super().run(request)  # 调用父类的run方法
        finally:
            await self.cleanup()  # 确保在完成后进行清理
