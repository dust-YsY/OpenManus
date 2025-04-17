import asyncio  # 导入异步IO库，用于异步操作和锁机制
import base64  # 导入base64库，用于图像编码
import json  # 导入JSON处理库
from typing import Generic, Optional, TypeVar  # 导入类型相关工具，支持泛型编程

# 导入浏览器自动化库的核心组件
from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator  # 导入Pydantic字段和验证器
from pydantic_core.core_schema import ValidationInfo

from app.config import config  # 导入应用配置
from app.llm import LLM  # 导入LLM客户端
from app.tool.base import BaseTool, ToolResult  # 导入工具基类和结果类型
from app.tool.web_search import WebSearch  # 导入网页搜索工具


# 定义浏览器工具的详细描述，展示工具的核心功能和使用方式
_BROWSER_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""

# 定义泛型类型变量，允许工具支持不同类型的上下文
Context = TypeVar("Context")


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"  # 工具名称
    description: str = _BROWSER_DESCRIPTION  # 工具描述
    parameters: dict = {  # 参数定义，详细指定每个操作的参数要求
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [  # 支持的浏览器操作列表
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "query": {
                "type": "string",
                "description": "Search query for 'web_search' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' action",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send for 'send_keys' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
            },
        },
        "required": ["action"],  # action是唯一必需的参数
        "dependencies": {  # 定义参数依赖关系，根据不同操作类型需要不同的附加参数
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)  # 异步锁，用于保护浏览器操作的线程安全
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)  # 浏览器实例
    context: Optional[BrowserContext] = Field(default=None, exclude=True)  # 浏览器上下文
    dom_service: Optional[DomService] = Field(default=None, exclude=True)  # DOM服务，用于操作网页元素
    web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)  # 网页搜索工具实例

    # 泛型上下文，用于存储特定类型的上下文信息
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)  # LLM客户端，用于内容提取和分析

    @field_validator("parameters", mode="before")  # 参数验证器，确保参数非空
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """
        确保浏览器和上下文已初始化。

        如果浏览器或上下文未初始化，创建新的实例并配置它们。
        返回浏览器上下文实例。
        """
        if self.browser is None:  # 如果浏览器未初始化
            browser_config_kwargs = {"headless": False, "disable_security": True}  # 默认配置

            if config.browser_config:  # 如果有自定义浏览器配置
                from browser_use.browser.browser import ProxySettings

                # 处理代理设置
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                # 从配置中获取其他浏览器属性
                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            # 创建浏览器实例
            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:  # 如果上下文未初始化
            context_config = BrowserContextConfig()  # 默认上下文配置

            # 如果有自定义上下文配置，使用它
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            # 创建新的浏览器上下文
            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def execute(
        self,
        action: str,  # 要执行的浏览器操作
        url: Optional[str] = None,  # 用于导航的URL
        index: Optional[int] = None,  # 元素索引，用于点击或输入
        text: Optional[str] = None,  # 要输入的文本或搜索查询
        scroll_amount: Optional[int] = None,  # 滚动像素量
        tab_id: Optional[int] = None,  # 标签页ID，用于标签页切换
        query: Optional[str] = None,  # 搜索查询
        goal: Optional[str] = None,  # 内容提取目标
        keys: Optional[str] = None,  # 要发送的按键
        seconds: Optional[int] = None,  # 等待秒数
        **kwargs,  # 额外参数
    ) -> ToolResult:
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action or search query
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            query: Search query for Google search
            goal: Extraction goal for content extraction
            keys: Keys to send for keyboard actions
            seconds: Seconds to wait
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
        async with self.lock:  # 使用异步锁确保操作的线程安全
            try:
                # 确保浏览器已初始化
                context = await self._ensure_browser_initialized()

                # 从配置获取最大内容长度
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 2000
                )

                # 导航类操作
                if action == "go_to_url":
                    if not url:
                        return ToolResult(
                            error="URL is required for 'go_to_url' action"
                        )
                    page = await context.get_current_page()
                    await page.goto(url)  # 导航到指定URL
                    await page.wait_for_load_state()  # 等待页面加载完成
                    return ToolResult(output=f"Navigated to {url}")

                elif action == "go_back":
                    await context.go_back()  # 返回上一页
                    return ToolResult(output="Navigated back")

                elif action == "refresh":
                    await context.refresh_page()  # 刷新当前页面
                    return ToolResult(output="Refreshed current page")

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'web_search' action"
                        )
                    # 执行网络搜索并返回结果，不直接导航到搜索页面
                    search_response = await self.web_search_tool.execute(
                        query=query, fetch_content=True, num_results=1
                    )
                    # 导航到第一个搜索结果
                    first_search_result = search_response.results[0]
                    url_to_navigate = first_search_result.url

                    page = await context.get_current_page()
                    await page.goto(url_to_navigate)  # 导航到搜索结果URL
                    await page.wait_for_load_state()  # 等待页面加载

                    return search_response

                # 元素交互类操作
                elif action == "click_element":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'click_element' action"
                        )
                    element = await context.get_dom_element_by_index(index)  # 获取指定索引的DOM元素
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    download_path = await context._click_element_node(element)  # 点击元素
                    output = f"Clicked element at index {index}"
                    if download_path:  # 如果点击触发了下载
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(output=output)

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)  # 获取指定索引的DOM元素
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    await context._input_text_element_node(element, text)  # 在元素中输入文本
                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}"
                    )

                elif action == "scroll_down" or action == "scroll_up":
                    direction = 1 if action == "scroll_down" else -1  # 确定滚动方向
                    amount = (
                        scroll_amount
                        if scroll_amount is not None
                        else context.config.browser_window_size["height"]  # 默认滚动一屏高度
                    )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {direction * amount});"  # 执行JavaScript滚动
                    )
                    return ToolResult(
                        output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels"
                    )

                elif action == "scroll_to_text":
                    if not text:
                        return ToolResult(
                            error="Text is required for 'scroll_to_text' action"
                        )
                    page = await context.get_current_page()
                    try:
                        locator = page.get_by_text(text, exact=False)  # 通过文本定位元素
                        await locator.scroll_into_view_if_needed()  # 滚动到元素可见
                        return ToolResult(output=f"Scrolled to text: '{text}'")
                    except Exception as e:
                        return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                elif action == "send_keys":
                    if not keys:
                        return ToolResult(
                            error="Keys are required for 'send_keys' action"
                        )
                    page = await context.get_current_page()
                    await page.keyboard.press(keys)  # 发送键盘按键
                    return ToolResult(output=f"Sent keys: {keys}")

                elif action == "get_dropdown_options":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'get_dropdown_options' action"
                        )
                    element = await context.get_dom_element_by_index(index)  # 获取指定索引的DOM元素
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    # 执行JavaScript获取下拉选项
                    options = await page.evaluate(
                        """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            return Array.from(select.options).map(opt => ({
                                text: opt.text,
                                value: opt.value,
                                index: opt.index
                            }));
                        }
                    """,
                        element.xpath,
                    )
                    return ToolResult(output=f"Dropdown options: {options}")

                elif action == "select_dropdown_option":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'select_dropdown_option' action"
                        )
                    element = await context.get_dom_element_by_index(index)  # 获取指定索引的DOM元素
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    await page.select_option(element.xpath, label=text)  # 在下拉列表中选择选项
                    return ToolResult(
                        output=f"Selected option '{text}' from dropdown at index {index}"
                    )

                # 内容提取类操作
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_content' action"
                        )

                    page = await context.get_current_page()
                    import markdownify

                    # 将HTML内容转换为Markdown格式
                    content = markdownify.markdownify(await page.content())

                    # 构建提示，指导LLM提取内容
                    prompt = f"""\
Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format.
Extraction goal: {goal}

Page content:
{content[:max_content_length]}
"""
                    messages = [{"role": "system", "content": prompt}]

                    # 定义提取函数的模式
                    extraction_function = {
                        "type": "function",
                        "function": {
                            "name": "extract_content",
                            "description": "Extract specific information from a webpage based on a goal",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "extracted_content": {
                                        "type": "object",
                                        "description": "The content extracted from the page according to the goal",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "Text content extracted from the page",
                                            },
                                            "metadata": {
                                                "type": "object",
                                                "description": "Additional metadata about the extracted content",
                                                "properties": {
                                                    "source": {
                                                        "type": "string",
                                                        "description": "Source of the extracted content",
                                                    }
                                                },
                                            },
                                        },
                                    }
                                },
                                "required": ["extracted_content"],
                            },
                        },
                    }

                    # 使用LLM提取内容，要求使用函数调用
                    response = await self.llm.ask_tool(
                        messages,
                        tools=[extraction_function],
                        tool_choice="required",
                    )

                    if response and response.tool_calls:
                        args = json.loads(response.tool_calls[0].function.arguments)
                        extracted_content = args.get("extracted_content", {})
                        return ToolResult(
                            output=f"Extracted from page:\n{extracted_content}\n"
                        )

                    return ToolResult(output="No content was extracted from the page.")

                # 标签页管理类操作
                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)  # 切换到指定标签页
                    page = await context.get_current_page()
                    await page.wait_for_load_state()  # 等待页面加载
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'open_tab' action")
                    await context.create_new_tab(url)  # 创建新标签页
                    return ToolResult(output=f"Opened new tab with {url}")

                elif action == "close_tab":
                    await context.close_current_tab()  # 关闭当前标签页
                    return ToolResult(output="Closed current tab")

                # 实用工具类操作
                elif action == "wait":
                    seconds_to_wait = seconds if seconds is not None else 3  # 默认等待3秒
                    await asyncio.sleep(seconds_to_wait)  # 异步等待指定秒数
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """
        获取当前浏览器状态并返回ToolResult。
        如果未提供context，使用self.context。

        这个方法返回浏览器的当前状态，包括URL、标题、标签页信息、
        可交互元素和滚动信息，以及页面截图。
        """
        try:
            # 使用提供的上下文或回退到self.context
            ctx = context or self.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")

            state = await ctx.get_state()  # 获取浏览器状态

            # 创建视口信息字典（如果不存在）
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            # 为状态获取截图
            page = await ctx.get_current_page()

            await page.bring_to_front()  # 确保页面在前台
            await page.wait_for_load_state()  # 等待页面加载

            # 捕获全页面截图
            screenshot = await page.screenshot(
                full_page=True, animations="disabled", type="jpeg", quality=100
            )

            screenshot = base64.b64encode(screenshot).decode("utf-8")  # Base64编码截图

            # 构建包含所有必需字段的状态信息
            state_info = {
                "url": state.url,  # 当前URL
                "title": state.title,  # 页面标题
                "tabs": [tab.model_dump() for tab in state.tabs],  # 标签页信息
                "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
                "interactive_elements": (  # 可交互元素列表
                    state.element_tree.clickable_elements_to_string()
                    if state.element_tree
                    else ""
                ),
                "scroll_info": {  # 滚动信息
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,  # 视口高度
            }

            return ToolResult(
                output=json.dumps(state_info, indent=4, ensure_ascii=False),
                base64_image=screenshot,  # 包含页面截图
            )
        except Exception as e:
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """
        清理浏览器资源。

        安全地关闭浏览器上下文和浏览器实例，
        释放所有相关资源。
        """
        async with self.lock:  # 使用锁确保安全清理
            if self.context is not None:
                await self.context.close()  # 关闭上下文
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()  # 关闭浏览器
                self.browser = None

    def __del__(self):
        """
        对象被销毁时确保清理。

        这是一个保底机制，确保即使没有显式调用cleanup，
        资源也会在对象被垃圾回收时释放。
        """
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())  # 尝试在主事件循环中清理
            except RuntimeError:
                # 如果主事件循环已关闭，创建新循环
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """
        工厂方法，创建带有特定上下文的BrowserUseTool。

        这允许使用特定上下文创建工具实例，
        便于在不同环境中使用该工具。
        """
        tool = cls()
        tool.tool_context = context
        return tool
