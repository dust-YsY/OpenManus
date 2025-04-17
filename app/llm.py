import math
from typing import Dict, List, Optional, Union

import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.bedrock import BedrockClient
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)


REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class TokenCounter:
    # Token常量
    BASE_MESSAGE_TOKENS = 4  # 每条消息的基础token数
    FORMAT_TOKENS = 2  # 消息格式的token数
    LOW_DETAIL_IMAGE_TOKENS = 85  # 低详细度图像的token数
    HIGH_DETAIL_TILE_TOKENS = 170  # 高详细度图像块的token数

    # 图像处理常量
    MAX_SIZE = 2048  # 图像的最大尺寸
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768  # 高详细度模式下短边的目标尺寸
    TILE_SIZE = 512  # 图像分块的大小

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer  # 初始化tokenizer实例

    def count_text(self, text: str) -> int:
        """计算文本字符串的token数量"""
        return 0 if not text else len(self.tokenizer.encode(text))  # 如果文本为空返回0，否则计算token数

    def count_image(self, image_item: dict) -> int:
        """
        基于详细度级别和尺寸计算图像的token数量

        对于"low"详细度：固定85 tokens
        对于"high"详细度：
        1. 缩放以适应2048x2048方块
        2. 缩放最短边到768px
        3. 计算512px块（每块170 tokens）
        4. 添加85 tokens
        """
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        # Default values when dimensions aren't available or detail level is unknown
        if detail == "high":
            # Default to a 1024x1024 image calculation for high detail
            return self._calculate_high_detail_tokens(1024, 1024)  # 765 tokens
        elif detail == "medium":
            # Default to a medium-sized image for medium detail
            return 1024  # This matches the original default
        else:
            # For unknown detail levels, use medium as default
            return 1024

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            # Add token counting related attributes
            self.total_input_tokens = 0
            self.total_completion_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            # Initialize tokenizer
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # If the model is not in tiktoken's presets, use cl100k_base as default
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            elif self.api_type == "aws":
                self.client = BedrockClient()
            else:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

            self.token_counter = TokenCounter(self.tokenizer)

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # If max_input_tokens is not set, always return True
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """生成token限制超出的错误消息"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"  # 默认错误消息

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        将消息转换为LLM的OpenAI消息格式。

        Args:
            messages: 消息列表，可以是dict或Message对象
            supports_images: 标志，指示目标模型是否支持图像输入

        Returns:
            List[dict]: 格式化为OpenAI格式的消息列表

        Raises:
            ValueError: 如果消息无效或缺少必需字段
            TypeError: 如果提供了不支持的消息类型

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # 将Message对象转换为字典
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # 如果消息是一个dict，确保它有必需的字段
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # 如果存在base64图像且模型支持图像，则处理图像
                if supports_images and message.get("base64_image"):
                    # 初始化或将content转换为适当的格式
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # 将字符串项转换为适当的文本对象
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # 将图像添加到content中
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # 移除base64_image字段
                    del message["base64_image"]
                # 如果模型不支持图像但消息有base64_image，优雅处理
                elif not supports_images and message.get("base64_image"):
                    # 只移除base64_image字段并保留文本内容
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # 否则：不包含该消息
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # 验证所有消息都有必需的字段
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @retry(  # 重试装饰器配置
        wait=wait_random_exponential(min=1, max=60),  # 随机指数退避策略
        stop=stop_after_attempt(6),  # 6次尝试后停止
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # 只对特定异常类型重试
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],  # 对话消息列表
        system_msgs: Optional[List[Union[dict, Message]]] = None,  # 可选的系统消息
        stream: bool = True,  # 是否流式传输响应
        temperature: Optional[float] = None,  # 采样温度
    ) -> str:
        """
        向LLM发送提示并获取响应。

        Args:
            messages: 对话消息列表
            system_msgs: 可选的系统消息前缀
            stream (bool): 是否流式传输响应
            temperature (float): 响应的采样温度

        Returns:
            str: 生成的响应

        Raises:
            TokenLimitExceeded: 如果超出token限制
            ValueError: 如果消息无效或响应为空
            OpenAIError: 如果API调用在重试后失败
            Exception: 对于意外错误
        """
        try:
            # 检查模型是否支持图像
            supports_images = self.model in MULTIMODAL_MODELS

            # 格式化系统和用户消息，检查图像支持
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # 计算输入token数量
            input_tokens = self.count_message_tokens(messages)

            # 检查是否超出token限制
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # 抛出一个特殊异常，不会被重试
                raise TokenLimitExceeded(error_message)

            # 构建API请求参数
            params = {
                "model": self.model,
                "messages": messages,
            }

            # 为不同模型设置不同的参数
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens  # 推理模型使用max_completion_tokens
            else:
                params["max_tokens"] = self.max_tokens  # 标准模型使用max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature  # 使用提供的温度或默认温度
                )

            if not stream:
                # 非流式请求处理
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                # 更新token计数
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                return response.choices[0].message.content

            # 流式请求处理，在发起请求前更新估计的token计数
            self.update_token_count(input_tokens)

            # 创建流式响应
            response = await self.client.chat.completions.create(**params, stream=True)

            # 收集流式响应片段
            collected_messages = []
            completion_text = ""
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message
                print(chunk_message, end="", flush=True)  # 实时打印流式响应

            print()  # 流式响应结束后换行
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            # 估计流式响应的完成token数
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"Estimated completion tokens for streaming response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # 不记录日志，直接重新抛出token限制错误
            raise
        except ValueError:
            logger.exception(f"Validation error")  # 记录验证错误
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")  # 记录OpenAI API错误
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask")  # 记录未预期的错误
            raise

    @retry(  # 重试装饰器配置
        wait=wait_random_exponential(min=1, max=60),  # 随机指数退避策略
        stop=stop_after_attempt(6),  # 6次尝试后停止
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # 只对特定异常类型重试
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],  # 对话消息列表
        images: List[Union[str, dict]],  # 图像URL或图像数据字典列表
        system_msgs: Optional[List[Union[dict, Message]]] = None,  # 可选的系统消息
        stream: bool = False,  # 是否流式传输响应
        temperature: Optional[float] = None,  # 采样温度
    ) -> str:
        """
        向LLM发送带有图像的提示并获取响应。

        Args:
            messages: 对话消息列表
            images: 图像URL或图像数据字典列表
            system_msgs: 可选的系统消息前缀
            stream (bool): 是否流式传输响应
            temperature (float): 响应的采样温度

        Returns:
            str: 生成的响应

        Raises:
            TokenLimitExceeded: 如果超出token限制
            ValueError: 如果消息无效或响应为空
            OpenAIError: 如果API调用在重试后失败
            Exception: 对于意外错误
        """
        try:
            # 对于ask_with_images，我们总是将supports_images设置为True
            # 因为此方法应该只与支持图像的模型一起使用
            if self.model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS}"
                )

            # 启用图像支持格式化消息
            formatted_messages = self.format_messages(messages, supports_images=True)

            # 确保最后一条消息来自用户，以便附加图像
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # 处理最后一条用户消息以包含图像
            last_message = formatted_messages[-1]

            # 将content转换为多模态格式（如果需要）
            content = last_message["content"]
            # 根据content类型进行适当的转换
            multimodal_content = (
                [{"type": "text", "text": content}]  # 字符串转为文本对象
                if isinstance(content, str)
                else content  # 已经是列表则保持原样
                if isinstance(content, list)
                else []  # 其他情况初始化为空列表
            )

            # 将图像添加到content中
            for image in images:
                if isinstance(image, str):  # 如果是URL字符串
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:  # 如果是带url的字典
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:  # 如果已经是格式化的图像对象
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # 使用多模态内容更新消息
            last_message["content"] = multimodal_content

            # 如果提供了系统消息，添加系统消息
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # 计算token并检查限制
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # 设置API参数
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # 添加特定于模型的参数
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # 处理非流式请求
            if not stream:
                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content

            # 处理流式请求
            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            # 收集流式响应片段
            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)  # 实时打印响应

            print()  # 流式响应后换行
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            return full_response

        except TokenLimitExceeded:
            raise  # 直接重新抛出token限制错误
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
            raise

    @retry(  # 重试装饰器配置
        wait=wait_random_exponential(min=1, max=60),  # 随机指数退避策略
        stop=stop_after_attempt(6),  # 6次尝试后停止
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],  # 对话消息列表
        system_msgs: Optional[List[Union[dict, Message]]] = None,  # 可选的系统消息
        timeout: int = 300,  # 请求超时时间（秒）
        tools: Optional[List[dict]] = None,  # 要使用的工具列表
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # 工具选择策略
        temperature: Optional[float] = None,  # 采样温度
        **kwargs,  # 额外的完成参数
    ) -> ChatCompletionMessage | None:
        """
        使用工具/函数询问LLM并返回响应。

        Args:
            messages: 对话消息列表
            system_msgs: 可选的系统消息前缀
            timeout: 请求超时时间（秒）
            tools: 要使用的工具列表
            tool_choice: 工具选择策略
            temperature: 响应的采样温度
            **kwargs: 额外的完成参数

        Returns:
            ChatCompletionMessage: 模型的响应

        Raises:
            TokenLimitExceeded: 如果超出token限制
            ValueError: 如果工具、工具选择或消息无效
            OpenAIError: 如果API调用在重试后失败
            Exception: 对于意外错误
        """
        try:
            # 验证工具选择
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # 检查模型是否支持图像
            supports_images = self.model in MULTIMODAL_MODELS

            # 格式化消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # 计算输入token数量
            input_tokens = self.count_message_tokens(messages)

            # 如果有工具，计算工具描述的token数量
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # 检查是否超出token限制
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # 抛出一个特殊异常，不会被重试
                raise TokenLimitExceeded(error_message)

            # 如果提供了工具，验证工具
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # 设置完成请求
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            # 为不同模型设置不同的参数
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            params["stream"] = False  # 工具请求始终使用非流式
            response: ChatCompletion = await self.client.chat.completions.create(
                **params
            )

            # 检查响应是否有效
            if not response.choices or not response.choices[0].message:
                print(response)
                # raise ValueError("Invalid or empty response from LLM")
                return None

            # 更新token计数
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

            return response.choices[0].message

        except TokenLimitExceeded:
            # 不记录日志，直接重新抛出token限制错误
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise
