from typing import Any, Dict, List, Optional, Tuple  # 导入类型注解

from pydantic import Field  # 导入Pydantic字段定义

from app.agent.toolcall import ToolCallAgent  # 导入工具调用代理基类
from app.logger import logger  # 导入日志模块
from app.prompt.mcp import MULTIMEDIA_RESPONSE_PROMPT, NEXT_STEP_PROMPT, SYSTEM_PROMPT  # 导入MCP提示词模板
from app.schema import AgentState, Message  # 导入代理状态和消息类型
from app.tool.base import ToolResult  # 导入工具结果基类
from app.tool.mcp import MCPClients  # 导入MCP客户端


class MCPAgent(ToolCallAgent):
    """Agent for interacting with MCP (Model Context Protocol) servers.

    This agent connects to an MCP server using either SSE or stdio transport
    and makes the server's tools available through the agent's tool interface.
    """

    name: str = "mcp_agent"  # 代理名称
    description: str = "An agent that connects to an MCP server and uses its tools."  # 代理描述

    system_prompt: str = SYSTEM_PROMPT  # 系统提示词
    next_step_prompt: str = NEXT_STEP_PROMPT  # 下一步行动提示词

    # 初始化MCP工具集合
    mcp_clients: MCPClients = Field(default_factory=MCPClients)  # MCP客户端实例
    available_tools: MCPClients = None  # 可用工具集合，将在initialize()中设置

    max_steps: int = 20  # 最大执行步骤
    connection_type: str = "stdio"  # 连接类型："stdio" 或 "sse"

    # 跟踪工具模式以检测变化
    tool_schemas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # 工具模式字典
    _refresh_tools_interval: int = 5  # 每N步刷新一次工具列表

    # 应该触发终止的特殊工具名称
    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate"])  # 特殊工具名称列表

    async def initialize(
        self,
        connection_type: Optional[str] = None,  # 连接类型
        server_url: Optional[str] = None,  # 服务器URL
        command: Optional[str] = None,  # 命令
        args: Optional[List[str]] = None,  # 命令参数
    ) -> None:
        """Initialize the MCP connection.

        Args:
            connection_type: Type of connection to use ("stdio" or "sse")
            server_url: URL of the MCP server (for SSE connection)
            command: Command to run (for stdio connection)
            args: Arguments for the command (for stdio connection)
        """
        if connection_type:  # 如果提供了连接类型
            self.connection_type = connection_type  # 更新连接类型

        # 根据连接类型连接到MCP服务器
        if self.connection_type == "sse":  # SSE连接
            if not server_url:  # 检查服务器URL
                raise ValueError("Server URL is required for SSE connection")
            await self.mcp_clients.connect_sse(server_url=server_url)  # 建立SSE连接
        elif self.connection_type == "stdio":  # 标准输入输出连接
            if not command:  # 检查命令
                raise ValueError("Command is required for stdio connection")
            await self.mcp_clients.connect_stdio(command=command, args=args or [])  # 建立stdio连接
        else:
            raise ValueError(f"Unsupported connection type: {self.connection_type}")

        # 设置可用工具为MCP实例
        self.available_tools = self.mcp_clients

        # 存储初始工具模式
        await self._refresh_tools()  # 刷新工具列表

        # 添加关于可用工具的系统消息
        tool_names = list(self.mcp_clients.tool_map.keys())  # 获取工具名称列表
        tools_info = ", ".join(tool_names)  # 将工具名称连接成字符串

        # 添加系统提示词和可用工具信息
        self.memory.add_message(
            Message.system_message(
                f"{self.system_prompt}\n\nAvailable MCP tools: {tools_info}"  # 组合系统提示词和工具信息
            )
        )

    async def _refresh_tools(self) -> Tuple[List[str], List[str]]:
        """Refresh the list of available tools from the MCP server.

        Returns:
            A tuple of (added_tools, removed_tools)
        """
        if not self.mcp_clients.session:  # 检查会话是否存在
            return [], []

        # 直接从服务器获取当前工具模式
        response = await self.mcp_clients.session.list_tools()  # 获取工具列表
        current_tools = {tool.name: tool.inputSchema for tool in response.tools}  # 创建工具模式字典

        # 确定添加、删除和更改的工具
        current_names = set(current_tools.keys())  # 当前工具名称集合
        previous_names = set(self.tool_schemas.keys())  # 之前工具名称集合

        added_tools = list(current_names - previous_names)  # 新增的工具
        removed_tools = list(previous_names - current_names)  # 移除的工具

        # 检查现有工具的架构变化
        changed_tools = []
        for name in current_names.intersection(previous_names):  # 遍历共同工具
            if current_tools[name] != self.tool_schemas.get(name):  # 检查模式是否变化
                changed_tools.append(name)  # 记录变化的工具

        # 更新存储的模式
        self.tool_schemas = current_tools

        # 记录和通知变化
        if added_tools:  # 如果有新增工具
            logger.info(f"Added MCP tools: {added_tools}")  # 记录日志
            self.memory.add_message(
                Message.system_message(f"New tools available: {', '.join(added_tools)}")  # 添加系统消息
            )
        if removed_tools:  # 如果有移除工具
            logger.info(f"Removed MCP tools: {removed_tools}")  # 记录日志
            self.memory.add_message(
                Message.system_message(
                    f"Tools no longer available: {', '.join(removed_tools)}"  # 添加系统消息
                )
            )
        if changed_tools:  # 如果有变化工具
            logger.info(f"Changed MCP tools: {changed_tools}")  # 记录日志

        return added_tools, removed_tools  # 返回变化信息

    async def think(self) -> bool:
        """Process current state and decide next action."""
        # 检查MCP会话和工具可用性
        if not self.mcp_clients.session or not self.mcp_clients.tool_map:  # 检查会话和工具映射
            logger.info("MCP service is no longer available, ending interaction")  # 记录日志
            self.state = AgentState.FINISHED  # 设置状态为完成
            return False  # 返回False表示结束

        # 定期刷新工具
        if self.current_step % self._refresh_tools_interval == 0:  # 检查是否需要刷新
            await self._refresh_tools()  # 刷新工具列表
            # 所有工具被移除表示关闭
            if not self.mcp_clients.tool_map:  # 检查工具映射是否为空
                logger.info("MCP service has shut down, ending interaction")  # 记录日志
                self.state = AgentState.FINISHED  # 设置状态为完成
                return False  # 返回False表示结束

        # 使用父类的think方法
        return await super().think()  # 调用父类的think方法

    async def _handle_special_tool(self, name: str, result: Any, **kwargs) -> None:
        """Handle special tool execution and state changes"""
        # 首先使用父类处理器处理
        await super()._handle_special_tool(name, result, **kwargs)  # 调用父类方法

        # 处理多媒体响应
        if isinstance(result, ToolResult) and result.base64_image:  # 检查是否有图片
            self.memory.add_message(
                Message.system_message(
                    MULTIMEDIA_RESPONSE_PROMPT.format(tool_name=name)  # 添加多媒体响应提示
                )
            )

    def _should_finish_execution(self, name: str, **kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        # 如果工具名称是'terminate'则终止
        return name.lower() == "terminate"  # 检查是否是终止工具

    async def cleanup(self) -> None:
        """Clean up MCP connection when done."""
        if self.mcp_clients.session:  # 检查会话是否存在
            await self.mcp_clients.disconnect()  # 断开连接
            logger.info("MCP connection closed")  # 记录日志

    async def run(self, request: Optional[str] = None) -> str:
        """Run the agent with cleanup when done."""
        try:
            result = await super().run(request)  # 调用父类的run方法
            return result  # 返回结果
        finally:
            # 确保即使发生错误也会执行清理
            await self.cleanup()  # 执行清理操作
