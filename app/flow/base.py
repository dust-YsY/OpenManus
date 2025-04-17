from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import Dict, List, Optional, Union  # 导入类型提示

from pydantic import BaseModel  # 导入Pydantic的BaseModel用于数据验证

from app.agent.base import BaseAgent  # 导入基础代理类


class BaseFlow(BaseModel, ABC):
    """Base class for execution flows supporting multiple agents"""

    agents: Dict[str, BaseAgent]  # 代理字典，键为代理名称，值为代理实例
    tools: Optional[List] = None  # 可选的工具列表
    primary_agent_key: Optional[str] = None  # 主代理的键名，可选

    class Config:
        arbitrary_types_allowed = True  # 允许模型属性使用任意Python类型

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # 处理不同方式提供的代理
        if isinstance(agents, BaseAgent):
            agents_dict = {"default": agents}  # 单个代理转换为字典
        elif isinstance(agents, list):
            agents_dict = {f"agent_{i}": agent for i, agent in enumerate(agents)}  # 代理列表转换为字典
        else:
            agents_dict = agents  # 已经是字典格式

        # 如果未指定主代理，使用第一个代理
        primary_key = data.get("primary_agent_key")
        if not primary_key and agents_dict:
            primary_key = next(iter(agents_dict))  # 获取字典中的第一个键
            data["primary_agent_key"] = primary_key

        # 设置代理字典
        data["agents"] = agents_dict

        # 使用BaseModel的初始化方法
        super().__init__(**data)

    @property
    def primary_agent(self) -> Optional[BaseAgent]:
        """Get the primary agent for the flow"""
        return self.agents.get(self.primary_agent_key)  # 返回主代理实例

    def get_agent(self, key: str) -> Optional[BaseAgent]:
        """Get a specific agent by key"""
        return self.agents.get(key)  # 根据键获取特定代理

    def add_agent(self, key: str, agent: BaseAgent) -> None:
        """Add a new agent to the flow"""
        self.agents[key] = agent  # 添加新代理到字典

    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """Execute the flow with given input"""
        # 子类必须实现此方法，定义流程的执行逻辑
