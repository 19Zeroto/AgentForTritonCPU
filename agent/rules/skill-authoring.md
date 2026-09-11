# Triton CPU Skill 制作与迁移规则

本文件只记录 AgentForTritonCPU 的项目约束。通用 skill 目录、`SKILL.md` 写法、
渐进加载、初始化和元数据校验由 Codex 内置 `$skill-creator` 负责，不在本仓复制。
制作或实质修改 skill 时先使用 `$skill-creator`，再应用本文件。

## 保存与路由

- Triton CPU 专用 skill 保存到本仓 `skills/<skill-name>/`，不安装到全局 skill
  目录，也不要求 Codex 默认扫描发现。
- 工作区 `AGENTS.md` 经仓根 `AGENTS.md` 进入本仓 `agent/AGENTS.md`；任务路由表再指向对应
  `skills/<skill-name>/SKILL.md`。`SKILL.md` 是该能力的统一 AI 入口。
- 新增、重命名或改变 skill 适用范围时，必须同步更新 `agent/AGENTS.md` 的
  Skill routing 表和相关 playbook。没有文档路由的 skill 视为不可用。
- 正常调用只加载 `SKILL.md`，再按其指引执行 `scripts/` 或读取必要 reference；
  不默认读取完整脚本源码和设计文档。

## 内容边界

- 可重复、需要稳定流程、多个入口或领域资料的 Agent 工作整理为 skill。
- 一次性探针先放工作区 `cache/`；确认复用价值后再迁入本仓。
- 与 `triton-cpu` 产品实现或测试框架强耦合、需要随产品代码 review 的内容留在
  产品仓；只服务 Agent 工作流的脚本、设计和历史证据迁入本仓。
- 可执行代码和相邻配置放 `scripts/`；设计、完整用法和历史证据放
  `references/`；输出模板才放 `assets/`。
- 日志、缓存、PID、状态文件、benchmark 结果和生成报告保存到工作区 `logs/`、
  `cache/` 或 `/tmp`，不得作为 skill 源文件。
- 单个脚本同时遵守 `script-writing.md`。

## 迁移流程

1. 盘点源文件、未提交版本、权限、配置、设计文档和运行产物；重要迁移保存文件
   清单或校验值。
2. 先按内容边界归类再移动；不得丢失源仓未提交的最新版本。
3. 合并旧 README、入口说明和重复设计文档，由 `SKILL.md` 统一路由。
4. 从脚本自身位置推导 `AGENTFORTRITONCPU_DIR`；`AGENT_DIR` 默认
   `${HOME}/agent`。支持 `AGENT_DIR`、`TRITON_REPO_DIR`、`LLVM_INSTALL_DIR`
   和显式输出目录覆盖。
5. 清除旧仓名、旧目录层级、具体用户绝对路径和产品仓内 Agent 临时路径。
6. 更新脚本 usage、reference、仓 README、Agent 路由及所有相对链接。
7. 用户要求迁出时，确认目标内容完整后再删除源文件；不修改历史，不自动
   stage、commit 或 push。

## 项目验证

完成内置 `$skill-creator` 要求的结构和元数据校验后，继续执行：

1. Python 文件在工作区规定环境中运行 `py_compile` 或 `compileall`。
2. 所有 Shell 文件运行 `bash -n`。
3. 主要入口运行 `--help`、`--dry-run` 或无副作用 smoke test。
4. 至少执行一个代表性最小真实任务；昂贵或破坏性任务未执行时明确报告。
5. 扫描旧仓名、旧前缀、具体用户绝对路径、仓内默认产物和失效示例。
6. 检查 Markdown 链接、脚本权限、输出位置及全部文档路由。
7. 检查目标仓和源仓 Git 状态，区分迁移删除、用户修改和新文件；确认没有未授权
   staged、commit 或 push。
