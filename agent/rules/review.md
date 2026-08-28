# Review 与 Git 规则

- 处理 review 修改前，先列出 review 项、代码位置、真实意图、当前状态和处理计划，再实施或回复。
- commit message 使用英文，只描述当前 commit 的实际变更；不要写开发过程或把 pending 计划混入主体。
- 用户要求 review commit 前不得自动 push。用户要求独立 commit 时不得 squash；只有用户明确要求合并时才可 squash 或 amend。
- 涉及 CLA、PR 元数据或 commit 时，author/committer 必须使用用户当前 Git 配置，不得改成 Agent 身份。
