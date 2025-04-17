"""Local Knowledge Base tool for document indexing and semantic search."""  # 本地知识库工具，用于文档索引和语义搜索

import os  # 导入操作系统接口模块
import uuid  # 导入UUID生成模块
from pathlib import Path  # 导入路径操作模块
from typing import Any, Dict, List, Optional, Union  # 导入类型提示工具

import numpy as np  # 导入数值计算库
from langchain_community.document_loaders import CSVLoader  # CSV文件加载器
from langchain_community.document_loaders import PyPDFLoader  # 基于PyPDF的PDF加载器
from langchain_community.document_loaders import TextLoader  # 文本文件加载器
from langchain_community.document_loaders import (  # 目录加载器，用于批量处理文件; 基于PDFMiner的PDF加载器; 导入各种文档加载器; Markdown文件加载器
    DirectoryLoader,
    PDFMinerLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import Chroma  # 导入Chroma向量存储
from langchain_core.documents import Document  # 导入文档数据结构
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 导入文本分割器
from pydantic import Field, validator  # 导入Pydantic验证工具
from sentence_transformers import (
    SentenceTransformer,
)  # 导入句子转换器，用于生成文本嵌入

from app.config import config  # 导入应用配置
from app.logger import logger  # 导入日志记录器
from app.tool.base import BaseTool, ToolResult  # 导入基础工具类和工具结果类


class KnowledgeBaseManager:
    """Manager for local knowledge base operations."""  # 本地知识库操作管理器

    def __init__(self, base_path: Optional[str] = None):
        """Initialize the knowledge base manager.

        Args:
            base_path: Base directory for the knowledge base. Defaults to workspace/knowledge_base.
        """
        self.base_path = Path(
            base_path or os.path.join(config.workspace_root, "knowledge_base")
        )  # 设置知识库根目录，默认为工作空间下的knowledge_base目录
        self.base_path.mkdir(
            parents=True, exist_ok=True
        )  # 确保根目录存在，如不存在则创建

        self.indexes_dir = self.base_path / "indexes"  # 设置索引存储目录
        self.indexes_dir.mkdir(exist_ok=True)  # 确保索引目录存在

        self.documents_dir = self.base_path / "documents"  # 设置文档存储目录
        self.documents_dir.mkdir(exist_ok=True)  # 确保文档目录存在

        # Use all-MiniLM-L6-v2 model for embeddings
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # 初始化文本嵌入模型，使用预训练的all-MiniLM-L6-v2模型

        # Map of index_id to Chroma collection
        self.loaded_indexes = {}  # 初始化索引缓存字典，用于存储已加载的索引

        # Document chunking settings
        self.text_splitter = RecursiveCharacterTextSplitter(  # 初始化文本分割器，用于将长文档分割成更小的块
            chunk_size=1000,  # 每个块的大小，以字符为单位
            chunk_overlap=200,  # 块之间的重叠区域大小，确保上下文连贯性
        )

        # File type to loader mapping
        self.loader_map = {  # 文件类型到加载器的映射
            ".txt": TextLoader,  # 文本文件使用TextLoader
            ".md": UnstructuredMarkdownLoader,  # Markdown文件使用UnstructuredMarkdownLoader
            ".pdf": PyPDFLoader,  # PDF文件使用PyPDFLoader
            ".csv": CSVLoader,  # CSV文件使用CSVLoader
        }

    def _get_file_loader(self, file_path: Path):
        """Get the appropriate document loader based on file extension."""  # 根据文件扩展名获取适当的文档加载器
        extension = file_path.suffix.lower()  # 获取小写的文件扩展名
        if extension in self.loader_map:  # 如果扩展名在映射中
            return self.loader_map[extension](str(file_path))  # 返回对应的加载器实例
        # Default to text loader
        return TextLoader(str(file_path))  # 默认使用文本加载器

    def _load_documents(self, path: Path) -> List[Document]:
        """Load documents from file or directory."""  # 从文件或目录加载文档
        documents = []  # 初始化文档列表

        if path.is_file():  # 如果路径是文件
            loader = self._get_file_loader(path)  # 获取对应的加载器
            documents.extend(loader.load())  # 加载文档并添加到列表
        elif path.is_dir():  # 如果路径是目录
            for ext in self.loader_map:  # 遍历所有支持的文件类型
                loader = DirectoryLoader(  # 使用目录加载器
                    str(path),  # 目录路径
                    glob=f"**/*{ext}",  # 匹配所有子目录中的指定扩展名文件
                    loader_cls=self.loader_map[ext],  # 使用对应的加载器类
                )
                documents.extend(loader.load())  # 加载文档并添加到列表

        # Split documents into chunks
        if documents:  # 如果加载了文档
            return self.text_splitter.split_documents(documents)  # 将文档分割成块并返回
        return []  # 如果没有文档，返回空列表

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""  # 为文本列表生成嵌入向量
        return self.model.encode(
            texts, normalize_embeddings=True
        )  # 使用模型编码文本并规范化嵌入

    def create_index(self, source_path: str, index_name: Optional[str] = None) -> str:
        """Create a new vector index from documents.

        Args:
            source_path: Path to file or directory containing documents
            index_name: Optional name for the index. If not provided, a UUID will be generated.

        Returns:
            The ID of the created index
        """
        source_path = Path(source_path)  # 转换源路径为Path对象
        if not source_path.exists():  # 检查源路径是否存在
            raise FileNotFoundError(
                f"Source path {source_path} does not exist"
            )  # 如果不存在，抛出文件未找到错误

        # Generate an index ID or use provided name
        index_id = index_name or str(
            uuid.uuid4()
        )  # 使用提供的索引名称或生成UUID作为索引ID
        index_dir = self.indexes_dir / index_id  # 确定索引存储目录

        # Load and process documents
        documents = self._load_documents(source_path)  # 加载并处理文档
        if not documents:  # 如果没有找到文档
            raise ValueError(f"No documents found at {source_path}")  # 抛出值错误

        # Create embedding function wrapper
        embedding_function = lambda texts: self.model.encode(  # 创建嵌入函数包装器
            texts, normalize_embeddings=True  # 规范化嵌入向量
        )

        # Create Chroma collection
        chroma_db = Chroma.from_documents(  # 从文档创建Chroma集合
            documents=documents,  # 文档列表
            embedding=embedding_function,  # 嵌入函数
            persist_directory=str(index_dir),  # 持久化目录
            collection_name=index_id,  # 集合名称
        )

        # Add to loaded indexes
        self.loaded_indexes[index_id] = (
            chroma_db  # 将新创建的索引添加到已加载索引字典中
        )

        logger.info(
            f"Created knowledge base index '{index_id}' with {len(documents)} chunks"
        )  # 记录创建索引的信息
        return index_id  # 返回索引ID

    def list_indexes(self) -> List[str]:
        """List all available indexes."""  # 列出所有可用的索引
        return [
            d.name for d in self.indexes_dir.iterdir() if d.is_dir()
        ]  # 返回索引目录中所有子目录的名称

    def load_index(self, index_id: str) -> bool:
        """Load a specific index into memory.

        Args:
            index_id: The ID of the index to load

        Returns:
            True if loaded successfully, False otherwise
        """
        index_dir = self.indexes_dir / index_id  # 确定索引目录
        if not index_dir.exists():  # 检查索引目录是否存在
            return False  # 如果不存在，返回False

        if index_id in self.loaded_indexes:  # 检查索引是否已经加载
            return True  # 如果已加载，返回True

        try:
            # Create embedding function wrapper
            embedding_function = lambda texts: self.model.encode(  # 创建嵌入函数包装器
                texts, normalize_embeddings=True  # 规范化嵌入向量
            )

            # Load Chroma collection
            chroma_db = Chroma(  # 加载Chroma集合
                persist_directory=str(index_dir),  # 持久化目录
                embedding_function=embedding_function,  # 嵌入函数
                collection_name=index_id,  # 集合名称
            )

            self.loaded_indexes[index_id] = (
                chroma_db  # 将加载的索引添加到已加载索引字典中
            )
            return True  # 返回加载成功
        except Exception as e:  # 捕获任何异常
            logger.error(f"Failed to load index {index_id}: {e}")  # 记录加载失败的错误
            return False  # 返回加载失败

    def query(self, index_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query an index with a text query.

        Args:
            index_id: The ID of the index to query
            query: The query text
            top_k: Number of top results to return

        Returns:
            List of results with document content and metadata
        """
        # Ensure index is loaded
        if index_id not in self.loaded_indexes:  # 检查索引是否已加载
            if not self.load_index(index_id):  # 尝试加载索引
                raise ValueError(
                    f"Index {index_id} not found"
                )  # 如果加载失败，抛出值错误

        chroma_db = self.loaded_indexes[index_id]  # 获取已加载的索引

        # Query the index
        results = chroma_db.similarity_search_with_relevance_scores(
            query, k=top_k
        )  # 执行相似性搜索，返回相关度分数

        # Format results
        formatted_results = []  # 初始化格式化结果列表
        for doc, score in results:  # 遍历搜索结果
            formatted_results.append(
                {  # 添加格式化结果
                    "content": doc.page_content,  # 文档内容
                    "metadata": doc.metadata,  # 文档元数据
                    "score": float(score),  # 将numpy浮点数转换为Python浮点数
                }
            )

        return formatted_results  # 返回格式化结果列表

    def delete_index(self, index_id: str) -> bool:
        """Delete an index.

        Args:
            index_id: The ID of the index to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        index_dir = self.indexes_dir / index_id  # 确定索引目录
        if not index_dir.exists():  # 检查索引目录是否存在
            return False  # 如果不存在，返回False

        # Remove from loaded indexes
        if index_id in self.loaded_indexes:  # 检查索引是否已加载
            del self.loaded_indexes[index_id]  # 从已加载索引字典中删除

        # Delete directory
        try:
            import shutil  # 导入文件操作模块

            shutil.rmtree(index_dir)  # 删除整个目录及其内容
            return True  # 返回删除成功
        except Exception as e:  # 捕获任何异常
            logger.error(
                f"Failed to delete index {index_id}: {e}"
            )  # 记录删除失败的错误
            return False  # 返回删除失败


class KnowledgeBaseTool(BaseTool):
    """Tool for managing and querying local knowledge bases."""  # 用于管理和查询本地知识库的工具

    name: str = "knowledge_base"  # 工具名称
    description: str = """Manage and query local knowledge bases.
Available commands:
- create_index: Create a new knowledge base index from documents
- list_indexes: List all available knowledge base indexes
- query: Search a knowledge base with a text query
- delete_index: Delete a knowledge base index
"""  # 工具描述，包含可用命令

    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {  # 命令参数
                "type": "string",
                "enum": [
                    "create_index",
                    "list_indexes",
                    "query",
                    "delete_index",
                ],  # 可用命令枚举
                "description": "The operation to perform on the knowledge base",
            },
            "source_path": {  # 源路径参数
                "type": "string",
                "description": "Path to document or directory to index (for create_index)",
            },
            "index_name": {  # 索引名称参数
                "type": "string",
                "description": "Name for the index (optional for create_index)",
            },
            "index_id": {  # 索引ID参数
                "type": "string",
                "description": "ID of the index to use (for query and delete_index)",
            },
            "query_text": {  # 查询文本参数
                "type": "string",
                "description": "The search query (for query)",
            },
            "top_k": {  # 返回结果数量参数
                "type": "integer",
                "description": "Number of top results to return (for query, default: 5)",
            },
        },
        "required": ["command"],  # 只有命令参数是必需的
        "additionalProperties": False,  # 不允许额外属性
    }

    kb_manager: KnowledgeBaseManager = Field(
        default_factory=KnowledgeBaseManager
    )  # 知识库管理器实例，默认创建新实例

    @validator(
        "kb_manager", pre=True, always=True
    )  # 验证器装饰器，在验证前执行，始终执行
    def set_kb_manager(cls, v):  # 设置知识库管理器的验证方法
        if isinstance(v, KnowledgeBaseManager):  # 如果已经是KnowledgeBaseManager实例
            return v  # 直接返回
        return KnowledgeBaseManager()  # 否则创建新实例

    async def execute(
        self,
        command: str,
        source_path: Optional[str] = None,
        index_name: Optional[str] = None,
        index_id: Optional[str] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        **kwargs,
    ) -> ToolResult:
        """Execute the knowledge base tool with specified parameters."""  # 使用指定参数执行知识库工具
        try:
            if command == "create_index":  # 如果命令是创建索引
                if not source_path:  # 检查源路径是否提供
                    return ToolResult(
                        error="source_path is required for create_index command"
                    )  # 返回错误结果

                index_id = self.kb_manager.create_index(
                    source_path, index_name
                )  # 创建索引
                return ToolResult(
                    output=f"Created knowledge base index: {index_id}"
                )  # 返回成功结果

            elif command == "list_indexes":  # 如果命令是列出索引
                indexes = self.kb_manager.list_indexes()  # 获取索引列表
                if indexes:  # 如果有索引
                    return ToolResult(
                        output=f"Available knowledge base indexes:\n{', '.join(indexes)}"
                    )  # 返回索引列表
                return ToolResult(
                    output="No knowledge base indexes found"
                )  # 返回未找到索引的消息

            elif command == "query":  # 如果命令是查询
                if not index_id:  # 检查索引ID是否提供
                    return ToolResult(
                        error="index_id is required for query command"
                    )  # 返回错误结果
                if not query_text:  # 检查查询文本是否提供
                    return ToolResult(
                        error="query_text is required for query command"
                    )  # 返回错误结果

                results = self.kb_manager.query(index_id, query_text, top_k)  # 执行查询
                if not results:  # 如果没有结果
                    return ToolResult(
                        output=f"No results found for query '{query_text}' in index {index_id}"
                    )  # 返回未找到结果的消息

                output = f"Search results for '{query_text}' in index {index_id}:\n\n"  # 构建输出文本的开头
                for i, result in enumerate(results, 1):  # 遍历结果，从1开始编号
                    output += f"Result {i} (Score: {result['score']:.4f}):\n"  # 添加结果编号和分数
                    output += (
                        f"Content: {result['content'][:300]}...\n"  # 添加截断的内容
                    )
                    if result["metadata"]:  # 如果有元数据
                        output += f"Source: {result['metadata'].get('source', 'Unknown')}\n"  # 添加源信息
                    output += "\n"  # 添加空行分隔

                return ToolResult(output=output)  # 返回格式化的结果

            elif command == "delete_index":  # 如果命令是删除索引
                if not index_id:  # 检查索引ID是否提供
                    return ToolResult(
                        error="index_id is required for delete_index command"
                    )  # 返回错误结果

                success = self.kb_manager.delete_index(index_id)  # 删除索引
                if success:  # 如果成功
                    return ToolResult(
                        output=f"Successfully deleted knowledge base index: {index_id}"
                    )  # 返回成功消息
                return ToolResult(
                    error=f"Failed to delete knowledge base index: {index_id}"
                )  # 返回失败消息

            else:  # 如果是未知命令
                return ToolResult(error=f"Unknown command: {command}")  # 返回错误结果

        except Exception as e:  # 捕获任何异常
            logger.error(f"Error in knowledge base tool: {str(e)}")  # 记录错误
            return ToolResult(error=f"Error: {str(e)}")  # 返回错误结果
