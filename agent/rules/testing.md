# 测试规则

测试命令和测试辅助脚本同时遵守 `agent/rules/script-writing.md`：不得依赖调用者当前目录或写死具体用户的 `/home/...` 路径。

- 不是所有任务都需要运行测试。只有在修改代码、复现失败、验证修复或用户明确要求测试时，才进入测试流程。
- 需要运行或建议用户运行测试前，先阅读 `agent/playbooks/testing.md`，按目标套件选择命令和环境变量，不要直接沿用记忆中的 pytest 参数。
- 所有测试命令必须显式设置 `OMP_NUM_THREADS=32` 或更低；不要依赖默认值。
- 涉及重新编译或安装 Triton CPU/MLIR 的验证时，整个流程的每条命令都必须继承与测试相同的 CPU/NUMA 绑定；必须用外层 `numactl --physcpubind=288-319 --membind=2` 启动重装脚本，不能只给最终 benchmark 绑核。绑定范围变化时，外层 `numactl`、脚本的 `--cpu-node/--mem-node/--cpu-list` 和 `OMP_NUM_THREADS` 必须同步修改。
- 使用 pytest-xdist 时，`pytest -n` 的并发数必须小于等于 32；需要更高并发时先征得用户确认，并说明资源风险。
- 运行前先确认目标用例属于 Triton `test_core`、`FlagGems/tests` 还是 benchmark；这些入口的 pytest 参数、marker、并发和环境变量要求不同。
- 只声明实际运行过的测试结果。targeted case 通过不能表述为全量测试通过。
- 如果测试依赖重新编译 LLVM、Triton Shared 或 triton-cpu，需要说明当前验证覆盖到哪一层。
- 处理失败日志时，先提炼首个根因错误，再决定是否需要扩大搜索范围。不要只根据最后一行错误做结论。
- 当前未运行测试时，不要在回复中声称已经验证通过。应明确说明未运行，并给出建议用户执行的命令或验证点。
- 用户提供复现命令时，必须使用相同环境和参数执行；涉及 `triton-cpu` 时，按测试 Playbook 加载本地环境并从仓库根目录运行。
- 全量或多用例验证使用 `-q --tb=no`；不要将 `-s -v` 与 pytest-xdist `-n` 组合使用。
- 调试单个失败时不使用 `-n`；只在检查日志、dump、print 输出或首个根因 traceback 时使用 `-s -v`。
- 含方括号的参数化 pytest case 名必须使用引号包裹。
- 修改 matcher 顺序、bitwidth 约束或 tiling action 后，验证计划必须同时覆盖原失败用例和可能被误匹配的 fallback 用例。
