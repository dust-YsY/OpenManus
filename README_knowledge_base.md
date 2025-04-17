# OpenManus 本地知识库集成

![OpenManus Logo](assets/logo.jpg)

## 功能概述

OpenManus 本地知识库集成是一项创新性扩展，为 OpenManus 代理提供了基于用户自有文档和知识的学习和回答能力。通过本地知识库集成，您可以：

- 索引并搜索本地文档（如 Markdown, PDF, TXT 等）
- 利用语义搜索技术，找到概念相关的内容
- 让代理基于您的个人或专业知识回答问题
- 构建企业内部知识库和智能问答系统

## 核心特性

- **多格式文档支持**：支持 Markdown、TXT、PDF 和 CSV 等多种格式
- **高效语义搜索**：基于 FAISS 和 SentenceTransformers 的高性能语义检索
- **易用的 API**：简单直观的接口，便于集成和使用
- **可定制性**：支持自定义嵌入模型和文本分块策略
- **与现有系统无缝集成**：作为标准工具加入 OpenManus 工具链

## 快速开始

### 安装依赖

确保已安装相关依赖：

```bash
pip install -r requirements.txt
```

### 基础用法

```python
# 创建索引
await agent.available_tools.execute(
    name="knowledge_base",
    tool_input={
        "command": "create_index",
        "source_path": "path/to/documents",
        "index_name": "my_knowledge"
    }
)

# 查询知识库
result = await agent.available_tools.execute(
    name="knowledge_base",
    tool_input={
        "command": "query",
        "index_id": "my_knowledge",
        "query_text": "你的问题",
        "top_k": 3
    }
)
```

### 运行示例

我们提供了一个完整的示例脚本，演示如何使用知识库功能：

```bash
python examples/knowledge_base_demo.py
```

## 技术架构

本地知识库集成基于以下技术构建：

- **文本嵌入**：使用 SentenceTransformers 将文本转换为向量表示
- **向量索引**：使用 FAISS 进行高效的向量相似度搜索
- **文档处理**：使用 LangChain 进行文档加载和分块
- **工具接口**：与 OpenManus 工具系统无缝集成

## 在项目中使用

知识库功能适用于多种场景：

- **个人知识管理**：整合个人笔记、文档和资料
- **专业领域问答**：构建特定领域的专业知识库
- **企业知识库**：集成公司文档、流程和政策
- **研究辅助**：整合研究论文和文献资料

## 文档

详细文档请参阅：

- [知识库集成技术文档](docs/knowledge_base_integration.md)
- [API 参考](docs/knowledge_base_api.md)
- [使用示例](examples/knowledge_base_demo.py)

## 贡献

我们欢迎任何形式的贡献！如果您有改进建议或发现问题，请创建 issue 或提交 pull request。

## 许可

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件。
