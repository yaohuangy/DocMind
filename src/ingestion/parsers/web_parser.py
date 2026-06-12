"""
网页抓取解析器。

使用 Trafilatura 作为首选引擎提取正文，
BeautifulSoup 作为回退方案。
自动过滤导航栏、广告、页脚等噪声内容。
"""

import logging

from llama_index.core.schema import Document as LlamaDocument

from src.ingestion.parsers.base_parser import BaseParser, make_doc_id

logger = logging.getLogger(__name__)


class WebParser(BaseParser):
    """网页解析器——Trafilatura 首选，BeautifulSoup 回退。

    Usage::

        parser = WebParser()
        docs = parser.parse("https://example.com/article")
        for d in docs:
            print(d.metadata["section_title"], d.text[:100])
    """

    @property
    def supported_format(self) -> str:
        return "web"

    def parse(self, source: str) -> list[LlamaDocument]:
        """抓取并解析网页。

        Args:
            source: 网页 URL（http/https）。

        Returns:
            LlamaDocument 列表。
        """
        doc_id = make_doc_id(source)
        doc_name = self._source_name(source)

        # 下载网页内容
        html = self._fetch_html(source)
        if html is None:
            logger.error("无法获取网页: %s", source)
            return []

        # 首选 Trafilatura
        result = self._parse_with_trafilatura(html, source)
        if result and result.strip():
            title = self._extract_title(html)
            doc = LlamaDocument(
                text=result.strip(),
                metadata={
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "source": source,
                    "format": "web",
                    "url": source,
                    "section_title": title or "",
                },
            )
            logger.info("网页解析完成 (Trafilatura): %s (%d 字符)", source, len(result))
            return [doc]

        # 回退 BeautifulSoup
        logger.info("Trafilatura 无有效输出，回退到 BeautifulSoup: %s", source)
        result = self._parse_with_bs4(html, source)
        if result and result.strip():
            title = self._extract_title(html)
            doc = LlamaDocument(
                text=result.strip(),
                metadata={
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "source": source,
                    "format": "web",
                    "url": source,
                    "section_title": title or "",
                },
            )
            logger.info("网页解析完成 (BeautifulSoup): %s (%d 字符)", source, len(result))
            return [doc]

        logger.warning("网页 %s 未提取到有效正文", source)
        return []

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_html(url: str, timeout: int = 15) -> str | None:
        """下载网页 HTML 源码。

        Args:
            url: 网页 URL。
            timeout: 超时秒数。

        Returns:
            HTML 字符串，失败返回 None。
        """
        try:
            import urllib.request

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # 检测编码
                content_type = resp.headers.get("Content-Type", "")
                charset = "utf-8"
                if "charset=" in content_type:
                    charset = content_type.split("charset=")[-1].strip()
                return resp.read().decode(charset, errors="replace")
        except Exception as e:
            logger.error("网页下载失败 %s: %s", url, e)
            return None

    @staticmethod
    def _extract_title(html: str) -> str:
        """从 HTML 中提取 <title> 文本。

        Args:
            html: HTML 字符串。

        Returns:
            标题文本，失败返回空字符串。
        """
        import re
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _parse_with_trafilatura(html: str, url: str = "") -> str:
        """使用 Trafilatura 提取正文。

        Args:
            html: HTML 字符串。
            url: 来源 URL（用于元数据）。

        Returns:
            提取的正文文本。
        """
        try:
            import trafilatura

            # 将 HTML 转为字节（trafilatura 需要）
            text = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
            return text or ""
        except ImportError:
            logger.warning("trafilatura 未安装")
            return ""
        except Exception as e:
            logger.warning("Trafilatura 解析失败: %s", e)
            return ""

    @staticmethod
    def _parse_with_bs4(html: str, url: str = "") -> str:
        """使用 BeautifulSoup 提取正文（回退方案）。

        Args:
            html: HTML 字符串。
            url: 来源 URL。

        Returns:
            提取的正文文本。
        """
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")
        except ImportError:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
            except ImportError:
                logger.warning("beautifulsoup4 未安装")
                return ""

        # 移除噪声标签
        for tag_name in ["script", "style", "nav", "footer", "header", "aside",
                         "noscript", "iframe", "form", "button"]:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 移除常见 class/id 噪声
        noise_patterns = [
            "nav", "menu", "sidebar", "footer", "header", "advertisement",
            "cookie", "banner", "comment", "widget", "related",
        ]
        for pattern in noise_patterns:
            for tag in soup.find_all(class_=lambda c: c and pattern in str(c).lower()):
                tag.decompose()
            for tag in soup.find_all(id=lambda i: i and pattern in str(i).lower()):
                tag.decompose()

        # 获取 body 文本
        body = soup.find("body")
        if body is None:
            body = soup

        text = body.get_text(separator="\n", strip=True)
        # 压缩多空行
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
