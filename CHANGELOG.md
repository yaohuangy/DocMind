# 📋 更新日志

## 2026-06-11

### ✨ 新增
- 🐳 **Docker 容器化**：新增 `Dockerfile`、`docker-compose.yml`、`.dockerignore`，`docker-compose up -d` 一键启动 Streamlit + Neo4j 全栈服务。数据通过 bind mount 持久化（`./data`、`./neo4j-data`），与本地 `python run.py` 共享同一份数据。compose 中含端口冲突故障说明。
- ☁️ **Railway 云端部署文档**：README 新增 Railway 部署步骤，10 分钟获得公网链接。

### 🔧 CI/CD
- ✅ **GitHub Actions**：新增 `.github/workflows/ci.yml`，Push/PR 自动跑 Ruff 代码检查 + mypy 类型检查 + pytest 单元测试（44/48 通过，4 个 e2e 因 Streamlit 框架限制在 CI 中跳过）。
- ✅ **代码质量配置**：新增 `pyproject.toml`（Ruff + mypy + pytest 配置）、`requirements-dev.txt`（开发依赖）。
- ✅ **Ruff 自动修复**：自动修复 512 个代码风格问题（现代类型注解、import 排序等），项目已通过零错误检查。

### ❌ 删除
- 🔄 **移除"重建记忆"功能**：该功能存在 bug（仅重建最后一条记录），且重建逻辑与记忆系统架构不兼容，已从回顾页面移除。
