#!/usr/bin/env python3
"""
知识库集成功能的示例脚本。
这个示例演示了如何使用OpenManus的本地知识库功能进行文档索引和查询。
"""

import asyncio
import os
from pathlib import Path

from app.agent.manus import Manus
from app.logger import logger


async def knowledge_base_demo():
    """演示知识库功能的主函数。"""

    # 初始化Manus代理
    agent = Manus()

    try:
        # 1. 创建示例文档路径
        samples_dir = Path("workspace/knowledge_base/documents/samples")
        if not samples_dir.exists():
            logger.warning(f"示例文档目录不存在: {samples_dir}")
            return

        logger.info("==== 开始知识库功能演示 ====")

        # 2. 列出现有索引
        logger.info("查询现有索引...")
        result = await agent.available_tools.execute(
            name="knowledge_base",
            tool_input={
                "command": "list_indexes"
            }
        )
        logger.info(f"现有索引: {result.output}")

        # 3. 创建新索引
        logger.info(f"从 {samples_dir} 创建新索引...")
        result = await agent.available_tools.execute(
            name="knowledge_base",
            tool_input={
                "command": "create_index",
                "index_name": "semantic_search_demo"
            }
        )
        logger.info(f"索引创建结果: {result.output}")

        index_id = "semantic_search_demo"

        # 4. 执行查询
        queries = [
            "什么是本地知识库功能?",
            "语义搜索和传统搜索有什么区别?",
            "FAISS是什么技术?",
            "语义搜索的应用场景有哪些?"
        ]

        logger.info("执行搜索查询...")
        for query in queries:
            logger.info(f"\n查询: {query}")
            result = await agent.available_tools.execute(
                name="knowledge_base",
                tool_input={
                    "command": "query",
                    "index_id": index_id,
                    "query_text": query,
                    "top_k": 2  # 每个查询返回前2个结果
                }
            )
            logger.info(f"查询结果:\n{result.output}")

        # 5. 基于本地知识库回答问题
        logger.info("\n基于知识库回答复杂问题...")
        complex_query = "比较语义搜索和传统关键词搜索的优缺点，并给出适合的应用场景。"

        # 首先查询相关内容
        kb_result = await agent.available_tools.execute(
            name="knowledge_base",
            tool_input={
                "command": "query",
                "index_id": index_id,
                "query_text": complex_query,
                "top_k": 3
            }
        )

        # 然后让代理基于查询结果回答问题
        prompt = f"""
        请基于以下从本地知识库检索到的信息，回答问题:

        问题: {complex_query}

        知识库检索结果:
        {kb_result.output}

        请提供详细、准确的回答，并引用知识库中的相关信息。
        """

        logger.info(f"向代理发送问题...")
        response = await agent.run(prompt)
        logger.info(f"代理回答完成")

        # 6. 清理（可选 - 注释掉以保留索引）
        logger.info(f"\n删除演示索引...")
        result = await agent.available_tools.execute(
            name="knowledge_base",
            tool_input={
                "command": "delete_index",
                "index_id": index_id
            }
        )
        logger.info(f"删除索引结果: {result.output}")

        logger.info("==== 知识库功能演示完成 ====")

    except Exception as e:
        logger.error(f"演示过程中出错: {e}")
    finally:
        # 清理资源
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(knowledge_base_demo())
