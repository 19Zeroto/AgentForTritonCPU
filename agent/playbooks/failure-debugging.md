# 失败调试 Playbook

## 使用条件

用于分析测试、benchmark、编译、lowering 或运行时失败。开始前阅读 `agent/rules/core.md`、`agent/rules/script-writing.md`、`agent/rules/testing.md` 和相关领域规则。

## 流程

1. 读取用户提供的完整日志或者自行运行测试的输出日志。
2. 定位第一个根因错误，不要只看最后一行。
3. 将错误归类到测试输入/配置、FlagGems 算子、Python runtime/JIT、TritonShared lowering、LLVM codegen 或 ARM/Kunpeng 后端配置。
4. 在 `agent/references/failure-library.md` 中查找同类错误和历史修复模式。
5. 只修改能解释当前失败的最小路径。
6. 给出原失败用例、可能被影响的 fallback 路径、验证命令和期望成功信号。
7. 已有稳定 pytest 复现且已知 good/bad commit 时，读取
   `skills/commit-bisect/SKILL.md`，按其安全检查和恢复要求执行 bisect。

## 产出

- 首个根因错误及证据
- 失败所属层级
- 最小修复范围
- 已执行与未执行验证
- 期望成功信号
