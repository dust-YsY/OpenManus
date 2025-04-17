import multiprocessing  # 导入多进程模块，用于隔离执行代码
import sys  # 导入系统模块，用于重定向标准输出
from io import StringIO  # 导入StringIO，用于捕获标准输出
from typing import Dict  # 导入类型注解

from app.tool.base import BaseTool  # 导入基础工具类


class PythonExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "python_execute"  # 工具名称
    description: str = "Executes Python code string. Note: Only print outputs are visible, function return values are not captured. Use print statements to see results."  # 工具描述
    parameters: dict = {  # 参数定义
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],  # 指定code为必需参数
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        """
        在隔离环境中执行Python代码并捕获输出和错误。

        Args:
            code: 要执行的Python代码字符串
            result_dict: 用于存储执行结果的共享字典
            safe_globals: 执行代码时使用的全局命名空间
        """
        original_stdout = sys.stdout  # 保存原始的标准输出
        try:
            output_buffer = StringIO()  # 创建内存缓冲区用于捕获输出
            sys.stdout = output_buffer  # 重定向标准输出到缓冲区
            exec(code, safe_globals, safe_globals)  # 在提供的全局命名空间中执行代码
            result_dict["observation"] = output_buffer.getvalue()  # 获取输出内容
            result_dict["success"] = True  # 标记执行成功
        except Exception as e:  # 捕获执行过程中的任何异常
            result_dict["observation"] = str(e)  # 将异常转换为字符串作为输出
            result_dict["success"] = False  # 标记执行失败
        finally:
            sys.stdout = original_stdout  # 恢复原始标准输出

    async def execute(
        self,
        code: str,  # 要执行的Python代码
        timeout: int = 5,  # 超时时间，默认5秒
    ) -> Dict:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.

        Returns:
            Dict: Contains 'output' with execution output or error message and 'success' status.
        """

        with multiprocessing.Manager() as manager:  # 创建进程管理器
            result = manager.dict({"observation": "", "success": False})  # 创建共享字典用于存储结果

            # 准备安全的全局命名空间，只包含内置函数
            if isinstance(__builtins__, dict):  # 处理不同Python环境中__builtins__的两种可能形式
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}

            # 创建新进程来执行代码
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            proc.start()  # 启动进程
            proc.join(timeout)  # 等待进程完成，最多等待timeout秒

            # 处理超时情况
            if proc.is_alive():  # 如果进程仍在运行，说明发生了超时
                proc.terminate()  # 终止进程
                proc.join(1)  # 等待进程结束，最多等待1秒
                return {
                    "observation": f"Execution timeout after {timeout} seconds",  # 返回超时信息
                    "success": False,  # 标记执行失败
                }
            return dict(result)  # 将共享字典转换为普通字典并返回结果
