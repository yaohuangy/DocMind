"""
LLMClient 解析逻辑单元测试。

重点测试 extract_concepts 的逗号分隔解析、前缀剥离、
generate_query_variants 的 JSON 解析回退等纯逻辑部分。
API 调用通过 mock 注入。
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.config import LLMConfig
from src.core.llm_client import LLMClient


@pytest.fixture
def llm_client():
    """创建带有 mock API 的 LLMClient。"""
    with patch("src.core.llm_client.OpenAI") as mock_openai:
        api_mock = MagicMock()
        mock_openai.return_value = api_mock
        config = LLMConfig(
            api_key="test-key",
            base_url="https://api.test.com/v1",
            model="test-model",
        )
        yield LLMClient(config), api_mock


# ============================================================================
# extract_concepts — 逗号分隔解析
# ============================================================================


class TestExtractConcepts:
    """extract_concepts 解析逻辑测试。"""

    def test_basic_comma_split(self, llm_client):
        client, api_mock = llm_client
        # mock chat 返回
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="Transformer,architecture,自注意力神经网络架构\nSelf-Attention,mechanism,序列元素相关性计算"
            ))],
            usage=None,
        )
        concepts = client.extract_concepts("test text")
        assert len(concepts) == 2
        assert concepts[0]["name"] == "Transformer"
        assert concepts[0]["type"] == "architecture"
        assert "自注意力" in concepts[0]["description"]
        assert concepts[1]["name"] == "Self-Attention"
        assert concepts[1]["type"] == "mechanism"

    def test_single_concept(self, llm_client):
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="RAG,technique,检索增强生成技术"
            ))],
            usage=None,
        )
        concepts = client.extract_concepts("RAG")
        assert len(concepts) == 1
        assert concepts[0]["name"] == "RAG"

    def test_strips_line_prefixes(self, llm_client):
        """去除行首的 - • * · 符号和序号。"""
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="1. Transformer,architecture,描述\n- BERT,model,预训练语言模型\n• GPT,model,生成式预训练"
            ))],
            usage=None,
        )
        concepts = client.extract_concepts("test")
        assert len(concepts) == 3
        names = [c["name"] for c in concepts]
        assert "Transformer" in names
        assert "BERT" in names
        assert "GPT" in names

    def test_empty_response(self, llm_client):
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))],
            usage=None,
        )
        concepts = client.extract_concepts("test")
        assert concepts == []

    def test_blank_lines_skipped(self, llm_client):
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="\n\nTransformer,architecture,描述\n\n\nBERT,model,描述"
            ))],
            usage=None,
        )
        concepts = client.extract_concepts("test")
        assert len(concepts) == 2

    def test_code_block_wrapper_removed(self, llm_client):
        """markdown 代码块行被跳过。"""
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="```\nTransformer,architecture,描述\n```"
            ))],
            usage=None,
        )
        concepts = client.extract_concepts("test")
        # ``` 行被跳过，中间的概念行正常解析
        assert len(concepts) == 1

    def test_missing_type_defaults_to_concept(self, llm_client):
        """类型缺失时默认 'concept'。"""
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="Transformer"
            ))],
            usage=None,
        )
        concepts = client.extract_concepts("test")
        assert len(concepts) == 1
        assert concepts[0]["type"] == "concept"
        assert concepts[0]["description"] == ""

    def test_mixed_markdown_prefixes(self, llm_client):
        """混合的序号样式都能正确处理。"""
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="1. Transformer,architecture,基础架构\n2) BERT,model,双向编码器\n3、GPT,model,自回归生成"
            ))],
            usage=None,
        )
        concepts = client.extract_concepts("test")
        assert len(concepts) == 3

    def test_whitespace_only_response(self, llm_client):
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="   \n  \n  "))],
            usage=None,
        )
        concepts = client.extract_concepts("test")
        assert concepts == []


# ============================================================================
# generate_query_variants — JSON 解析与回退
# ============================================================================


class TestGenerateQueryVariants:
    """generate_query_variants 测试。"""

    def test_json_parsing(self, llm_client):
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='["变体1", "变体2", "变体3", "变体4"]'
            ))],
            usage=None,
        )
        variants = client.generate_query_variants("什么是 Transformer？")
        assert len(variants) <= 4
        assert len(variants) > 0

    def test_json_with_markdown_wrapper(self, llm_client):
        """模型在 JSON 外包了 ```json ... ```。"""
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='```json\n["var1", "var2", "var3", "var4"]\n```'
            ))],
            usage=None,
        )
        variants = client.generate_query_variants("question?")
        assert len(variants) >= 1

    def test_json_fallback_line_split(self, llm_client):
        """JSON 解析失败时回退到行分割。"""
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="1. 什么是 Transformer 的定义？\n2. Transformer 的工作机制是什么？"
            ))],
            usage=None,
        )
        variants = client.generate_query_variants("question?")
        assert len(variants) >= 1
        assert isinstance(variants[0], str)

    def test_truncates_to_num_variants(self, llm_client):
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='["v1", "v2", "v3", "v4", "v5", "v6"]'
            ))],
            usage=None,
        )
        variants = client.generate_query_variants("q", num_variants=3)
        # 即使 LLM 返回 6 条，也只取前 3 条
        assert len(variants) == 3


# ============================================================================
# chat — 基本调用
# ============================================================================


class TestChat:
    """LLMClient.chat() 基本测试。"""

    def test_chat_returns_stripped_content(self, llm_client):
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="  Hello World  \n"))],
            usage=None,
        )
        result = client.chat([{"role": "user", "content": "hi"}])
        assert result == "Hello World"

    def test_chat_none_content(self, llm_client):
        """content 为 None 时返回空字符串。"""
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=None))],
            usage=None,
        )
        result = client.chat([{"role": "user", "content": "hi"}])
        assert result == ""

    def test_chat_captures_token_usage(self, llm_client):
        client, api_mock = llm_client
        api_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        client.chat([{"role": "user", "content": "hi"}])
        assert client.last_usage["prompt_tokens"] == 10
        assert client.last_usage["completion_tokens"] == 5
        assert client.last_usage["total_tokens"] == 15

    def test_model_property(self, llm_client):
        client, _ = llm_client
        assert client.model == "test-model"
