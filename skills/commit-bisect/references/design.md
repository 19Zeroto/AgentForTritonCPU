# Triton Commit Bisect

入口：`scripts/bisect_triton_commit.sh`。脚本切换 Triton commit、重建并运行指定 pytest
目标，日志写入工作区 `logs/`。此工具会改变目标仓 checkout；使用前必须确认目标
仓无 tracked 修改，并按脚本 `--help` 配置 good/bad commit。
