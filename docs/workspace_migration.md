# 工作区迁移说明

这是旧文档名称的兼容入口。当前面向人的完整设置说明已统一到
[workspace-setup.md](workspace-setup.md)，请优先阅读该文档。

当前入口约定是：

```text
$AGENT_DIR/AGENTS.md -> AgentForTritonCPU/AGENTS.md
```

它会继续路由到 `AgentForTritonCPU/agent/AGENTS.md`。不要在 `triton-cpu` 内创建
Agent 配置、软链或脚本副本；日志、缓存、测试状态和 benchmark 结果放到工作区
外部目录。
