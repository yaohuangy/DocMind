# 📋 更新日志

## 2026-06-11

### ✨ 新增
- 🐳 **Docker 容器化**：新增 `Dockerfile`、`docker-compose.yml`、`.dockerignore`，`docker-compose up -d` 一键启动 Streamlit + Neo4j 全栈服务。数据通过 bind mount 持久化（`./data`、`./neo4j-data`），与本地 `python run.py` 共享同一份数据。compose 中含端口冲突故障说明。
- ☁️ **Railway 云端部署文档**：README 新增 Railway 部署步骤，10 分钟获得公网链接。

### ❌ 删除
- 🔄 **移除"重建记忆"功能**：该功能存在 bug（仅重建最后一条记录），且重建逻辑与记忆系统架构不兼容，已从回顾页面移除。
