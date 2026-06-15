"""
QAEngine._normalize_method 静态方法测试。

覆盖所有合法的输入变体和不合法输入。
"""

import pytest

from src.engine.qa_engine import QAEngine


class TestNormalizeMethod:
    """_normalize_method 方法名规范化测试。"""

    # ---- 英文输入 ----

    def test_direct_lowercase(self):
        assert QAEngine._normalize_method("direct") == "direct"

    def test_mqe_lowercase(self):
        assert QAEngine._normalize_method("mqe") == "mqe"

    def test_hyde_lowercase(self):
        assert QAEngine._normalize_method("hyde") == "hyde"

    def test_combined_plus(self):
        assert QAEngine._normalize_method("mqe+hyde") == "mqe+hyde"

    # ---- 中文输入 ----

    def test_chinese_direct(self):
        assert QAEngine._normalize_method("直接检索") == "direct"

    # ---- 带空格 ----

    def test_with_spaces(self):
        assert QAEngine._normalize_method(" mqe+hyde ") == "mqe+hyde"

    # ---- 下划线分隔 ----

    def test_underscore_separator(self):
        assert QAEngine._normalize_method("mqe_hyde") == "mqe+hyde"

    # ---- 破折号分隔 ----

    def test_dash_separator(self):
        assert QAEngine._normalize_method("mqe-hyde") == "mqe+hyde"

    # ---- 大写输入 ----

    def test_uppercase_direct(self):
        assert QAEngine._normalize_method("DIRECT") == "direct"

    def test_uppercase_combined(self):
        assert QAEngine._normalize_method("MQE+HYDE") == "mqe+hyde"

    # ---- 紧凑写法 ----

    def test_compact_mqehyde(self):
        assert QAEngine._normalize_method("mqehyde") == "mqe+hyde"

    # ---- "combined" 别名 ----

    def test_combined_alias(self):
        assert QAEngine._normalize_method("combined") == "mqe+hyde"

    # ---- 异常输入 ----

    def test_invalid_method_raises_valueerror(self):
        with pytest.raises(ValueError, match="无效的检索方法"):
            QAEngine._normalize_method("invalid_method")

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError, match="无效的检索方法"):
            QAEngine._normalize_method("")

    def test_none_input_raises(self):
        """None 输入触发 AttributeError（在 .lower() 调用时）。"""
        with pytest.raises(AttributeError):
            QAEngine._normalize_method(None)  # type: ignore[arg-type]
