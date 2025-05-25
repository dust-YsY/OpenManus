import asyncio
import os
import time

from app.agent.knowledge_base import KnowledgeBaseAgent
from app.agent.manus import Manus
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger

os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"


async def run_flow():
    agents = {
        # "manus": Manus(),
        "knowledge_base": KnowledgeBaseAgent(),
    }

    try:
        prompt = input("Enter your prompt: ")

        if prompt.strip().isspace() or not prompt:
            logger.warning("Empty prompt provided.")
            return

        # 创建Flow对象
        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        logger.warning("Processing your request...")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 60 minute timeout for the entire execution
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Request processed in {elapsed_time:.2f} seconds")
            logger.info(result)
        except asyncio.TimeoutError:
            logger.error("Request processing timed out after 1 hour")
            logger.info(
                "Operation terminated due to timeout. Please try a simpler request."
            )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_flow())
