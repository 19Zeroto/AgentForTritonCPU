# FlagGems Test Suite Tools

本 skill 保存 FlagGems correctness test runner。Python 入口仅依赖 CPython
stdlib，但实际 pytest workload 需要已准备的 Triton CPU、LLVM/MLIR 和 FlagGems。

## Entrypoints

- `scripts/run_triton_tests.py`：按 `(test file, pytest marker)` 并行执行。
- `scripts/run_simple_tests.py`：按文件并行执行，每个文件完成后清理 Triton cache。
- `scripts/summarize_results.py`：汇总 marker runner state。
- `scripts/run_all_flaggems_tests.sh`：逐文件批量 pytest。
- `scripts/run_flaggems_full.sh`：全量测试封装，支持后台运行。

默认测试目录为 `$TRITON_REPO_DIR/FlagGems/tests`：

状态文件路径相对 test 目录无关；state 中测试文件保存为 tests root 相对路径。
`--resume` 跳过已完成项，`--force` 清除 marker state 后重跑。

修改后只做语法/CLI 检查；除非当前机器已配置目标运行环境，不启动全量 pytest。
