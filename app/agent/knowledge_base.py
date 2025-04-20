from typing import TYPE_CHECKING

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.knowledge_base import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import ToolChoice
from app.tool import KnowledgeBaseTool, Terminate, ToolCollection

if TYPE_CHECKING:
    from app.agent.base import BaseAgent


class KnowledgeBaseAgent(ToolCallAgent):
    """
    A knowledge base agent that specializes in managing and querying document collections.

    This agent can create indexes from documents, search through indexed content,
    and manage knowledge base resources efficiently.
    """

    name: str = "knowledge_base"
    description: str = (
        "A knowledge base agent that can manage and query document collections"
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_steps: int = 10

    # Configure the available tools
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(KnowledgeBaseTool(), Terminate())
    )

    # Use Auto for tool choice to allow both tool usage and free-form responses
    tool_choices: ToolChoice = ToolChoice.AUTO
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
