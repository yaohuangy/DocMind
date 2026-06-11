# ============================================================================
# DocMind — 智能文档问答助手 Docker 镜像
# ============================================================================
# 构建: docker build -t docmind .
# 运行: docker run -p 7860:7860 --env-file .env -v ./data:/app/data docmind
# ============================================================================

FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# 安装系统依赖（PyMuPDF 需要 libmupdf，Trafilatura 需要 libxml2 等）
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 安装 Python 依赖（利用 Docker 层缓存）
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# ---------------------------------------------------------------------------
# 复制项目代码（.dockerignore 排除不必要文件）
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# 创建运行时数据目录
# ---------------------------------------------------------------------------
RUN mkdir -p /app/data /app/data/chroma /app/data/uploads

# 暴露 Streamlit 默认端口
EXPOSE 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')" || exit 1

# ---------------------------------------------------------------------------
# 启动命令：绑定 0.0.0.0 以便容器外部访问
# ---------------------------------------------------------------------------
CMD ["streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "7860", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
