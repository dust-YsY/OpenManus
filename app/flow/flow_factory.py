from enum import Enum  # 导入Enum用于创建枚举类型
from typing import Dict, List, Union  # 导入类型提示相关的工具

from app.agent.base import BaseAgent  # 导入基础代理类
from app.flow.base import BaseFlow  # 导入基础流程类
from app.flow.planning import PlanningFlow  # 导入计划流程实现类


class FlowType(str, Enum):  # 定义流程类型枚举，同时继承str便于序列化
    PLANNING = "planning"  # 目前仅支持"planning"类型流程


class FlowFactory:
    """Factory for creating different types of flows with support for multiple agents"""

    @staticmethod  # 静态方法，不需要实例化即可调用
    def create_flow(
        flow_type: FlowType,  # 流程类型参数，类型为FlowType枚举
        agents: Union[
            BaseAgent, List[BaseAgent], Dict[str, BaseAgent]
        ],  # 代理参数，支持单个代理、代理列表或代理字典
        **kwargs,  # 额外的关键字参数，将传递给具体流程类的构造函数
    ) -> BaseFlow:  # 返回类型为BaseFlow
        flows = {  # 流程类型到流程类的映射字典
            FlowType.PLANNING: PlanningFlow,  # 将PLANNING类型映射到PlanningFlow类
        }

        flow_class = flows.get(flow_type)  # 根据流程类型获取对应的流程类
        if not flow_class:  # 如果找不到对应的流程类
            raise ValueError(f"Unknown flow type: {flow_type}")  # 抛出ValueError异常

        return flow_class(
            agents, **kwargs
        )  # 使用提供的代理和其他参数创建并返回流程实例
