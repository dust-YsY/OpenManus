import threading
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Get the project root directory"""
    # 获取项目根目录
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()  # 项目根目录
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"  # 工作空间根目录


class LLMSettings(BaseModel):
    # LLM模型设置类，定义语言模型的配置参数
    model: str = Field(..., description="Model name")  # 模型名称
    base_url: str = Field(..., description="API base URL")  # API基础URL
    api_key: str = Field(..., description="API key")  # API密钥
    max_tokens: int = Field(4096, description="Maximum number of tokens per request")  # 每次请求的最大token数
    max_input_tokens: Optional[int] = Field(
        None,
        description="Maximum input tokens to use across all requests (None for unlimited)",
    )  # 所有请求的最大输入token数（None表示无限制）
    temperature: float = Field(1.0, description="Sampling temperature")  # 采样温度
    api_type: str = Field(..., description="Azure, Openai, or Ollama")  # API类型
    api_version: str = Field(..., description="Azure Openai version if AzureOpenai")  # 如果是Azure OpenAI，则指定版本


class ProxySettings(BaseModel):
    # 代理设置类，定义代理服务器配置
    server: str = Field(None, description="Proxy server address")  # 代理服务器地址
    username: Optional[str] = Field(None, description="Proxy username")  # 代理用户名
    password: Optional[str] = Field(None, description="Proxy password")  # 代理密码


class SearchSettings(BaseModel):
    # 搜索设置类，定义搜索引擎配置
    engine: str = Field(default="Google", description="Search engine the llm to use")  # 使用的搜索引擎
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["DuckDuckGo", "Baidu", "Bing"],
        description="Fallback search engines to try if the primary engine fails",
    )  # 主搜索引擎失败时的备用搜索引擎
    retry_delay: int = Field(
        default=60,
        description="Seconds to wait before retrying all engines again after they all fail",
    )  # 所有引擎失败后重试的等待时间（秒）
    max_retries: int = Field(
        default=3,
        description="Maximum number of times to retry all engines when all fail",
    )  # 所有引擎失败时的最大重试次数
    lang: str = Field(
        default="en",
        description="Language code for search results (e.g., en, zh, fr)",
    )  # 搜索结果的语言代码
    country: str = Field(
        default="us",
        description="Country code for search results (e.g., us, cn, uk)",
    )  # 搜索结果的国家代码


class BrowserSettings(BaseModel):
    # 浏览器设置类，定义浏览器配置
    headless: bool = Field(False, description="Whether to run browser in headless mode")  # 是否在无头模式下运行浏览器
    disable_security: bool = Field(
        True, description="Disable browser security features"
    )  # 是否禁用浏览器安全功能
    extra_chromium_args: List[str] = Field(
        default_factory=list, description="Extra arguments to pass to the browser"
    )  # 传递给浏览器的额外参数
    chrome_instance_path: Optional[str] = Field(
        None, description="Path to a Chrome instance to use"
    )  # 要使用的Chrome实例路径
    wss_url: Optional[str] = Field(
        None, description="Connect to a browser instance via WebSocket"
    )  # 通过WebSocket连接到浏览器实例
    cdp_url: Optional[str] = Field(
        None, description="Connect to a browser instance via CDP"
    )  # 通过CDP连接到浏览器实例
    proxy: Optional[ProxySettings] = Field(
        None, description="Proxy settings for the browser"
    )  # 浏览器的代理设置
    max_content_length: int = Field(
        2000, description="Maximum length for content retrieval operations"
    )  # 内容检索操作的最大长度


class SandboxSettings(BaseModel):
    """Configuration for the execution sandbox"""
    # 沙箱执行环境配置类

    use_sandbox: bool = Field(False, description="Whether to use the sandbox")  # 是否使用沙箱
    image: str = Field("python:3.12-slim", description="Base image")  # 基础镜像
    work_dir: str = Field("/workspace", description="Container working directory")  # 容器工作目录
    memory_limit: str = Field("512m", description="Memory limit")  # 内存限制
    cpu_limit: float = Field(1.0, description="CPU limit")  # CPU限制
    timeout: int = Field(300, description="Default command timeout (seconds)")  # 默认命令超时时间（秒）
    network_enabled: bool = Field(
        False, description="Whether network access is allowed"
    )  # 是否允许网络访问


class MCPSettings(BaseModel):
    """Configuration for MCP (Model Context Protocol)"""
    # MCP（模型上下文协议）配置类

    server_reference: str = Field(
        "app.mcp.server", description="Module reference for the MCP server"
    )  # MCP服务器的模块引用


class AppConfig(BaseModel):
    # 应用程序配置类
    llm: Dict[str, LLMSettings]  # LLM设置字典
    sandbox: Optional[SandboxSettings] = Field(
        None, description="Sandbox configuration"
    )  # 沙箱配置
    browser_config: Optional[BrowserSettings] = Field(
        None, description="Browser configuration"
    )  # 浏览器配置
    search_config: Optional[SearchSettings] = Field(
        None, description="Search configuration"
    )  # 搜索配置
    mcp_config: Optional[MCPSettings] = Field(None, description="MCP configuration")  # MCP配置

    class Config:
        arbitrary_types_allowed = True  # 允许任意类型


class Config:
    # 配置单例类
    _instance = None  # 单例实例
    _lock = threading.Lock()  # 线程锁
    _initialized = False  # 初始化标志

    def __new__(cls):
        # 实现单例模式
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 初始化配置，只执行一次
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        # 获取配置文件路径
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found in config directory")  # 未找到配置文件

    def _load_config(self) -> dict:
        # 加载配置文件
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        # 加载初始配置
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }  # 获取LLM覆盖配置

        # 默认LLM设置
        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "max_input_tokens": base_llm.get("max_input_tokens"),
            "temperature": base_llm.get("temperature", 1.0),
            "api_type": base_llm.get("api_type", ""),
            "api_version": base_llm.get("api_version", ""),
        }

        # 处理浏览器配置
        browser_config = raw_config.get("browser", {})
        browser_settings = None

        if browser_config:
            # 处理代理设置
            proxy_config = browser_config.get("proxy", {})
            proxy_settings = None

            if proxy_config and proxy_config.get("server"):
                proxy_settings = ProxySettings(
                    **{
                        k: v
                        for k, v in proxy_config.items()
                        if k in ["server", "username", "password"] and v
                    }
                )

            # 过滤有效的浏览器配置参数
            valid_browser_params = {
                k: v
                for k, v in browser_config.items()
                if k in BrowserSettings.__annotations__ and v is not None
            }

            # 如果有代理设置，将其添加到参数中
            if proxy_settings:
                valid_browser_params["proxy"] = proxy_settings

            # 只有在有有效参数时才创建BrowserSettings
            if valid_browser_params:
                browser_settings = BrowserSettings(**valid_browser_params)

        # 处理搜索配置
        search_config = raw_config.get("search", {})
        search_settings = None
        if search_config:
            search_settings = SearchSettings(**search_config)

        # 处理沙箱配置
        sandbox_config = raw_config.get("sandbox", {})
        if sandbox_config:
            sandbox_settings = SandboxSettings(**sandbox_config)
        else:
            sandbox_settings = SandboxSettings()  # 使用默认配置

        # 处理MCP配置
        mcp_config = raw_config.get("mcp", {})
        mcp_settings = None
        if mcp_config:
            mcp_settings = MCPSettings(**mcp_config)
        else:
            mcp_settings = MCPSettings()  # 使用默认配置

        # 创建配置字典
        config_dict = {
            "llm": {
                "default": default_settings,
                **{
                    name: {**default_settings, **override_config}
                    for name, override_config in llm_overrides.items()
                },
            },
            "sandbox": sandbox_settings,
            "browser_config": browser_settings,
            "search_config": search_settings,
            "mcp_config": mcp_settings,
        }

        self._config = AppConfig(**config_dict)  # 创建应用配置实例

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        # 获取LLM设置
        return self._config.llm

    @property
    def sandbox(self) -> SandboxSettings:
        # 获取沙箱设置
        return self._config.sandbox

    @property
    def browser_config(self) -> Optional[BrowserSettings]:
        # 获取浏览器配置
        return self._config.browser_config

    @property
    def search_config(self) -> Optional[SearchSettings]:
        # 获取搜索配置
        return self._config.search_config

    @property
    def mcp_config(self) -> MCPSettings:
        """Get the MCP configuration"""
        # 获取MCP配置
        return self._config.mcp_config

    @property
    def workspace_root(self) -> Path:
        """Get the workspace root directory"""
        # 获取工作空间根目录
        return WORKSPACE_ROOT

    @property
    def root_path(self) -> Path:
        """Get the root path of the application"""
        # 获取应用程序根路径
        return PROJECT_ROOT


config = Config()  # 创建全局配置实例
