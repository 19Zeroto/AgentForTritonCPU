# FlagGems Benchmark Shape 修改指南

这份文档帮助人类维护者判断：一个 benchmark 的 shape 应该改在算子代码、共享
YAML，还是 fallback。产品 benchmark 位于相邻的 `triton-cpu/FlagGems/benchmark/`。

## 先判断谁拥有 shape

1. 如果 benchmark 类重写了 `set_shapes()`，先看这个方法。它可能完全绕过共享
   `core_shapes.yaml`。
2. 如果使用基类 shape 流程，优先在 `core_shapes.yaml` 增加与算子名完全一致的
   配置：

   ```yaml
   silu_and_mul:
     shapes:
       - [1024, 4096]
       - [8192, 4096]
     shape_desc: "M, N"
   ```

3. 如果要影响一类 benchmark，再考虑类名配置或 `set_more_shapes()`。后者通常
   只在 `--level comprehensive` 下追加 shape。
4. 最后才修改 `attri_util.py` 的 `DEFAULT_SHAPES`；这是影响面最大的 fallback。

## 基类的查找顺序

普通 benchmark 大致按以下顺序工作：

1. `conftest.py` 读取 `--shape_file` 和 `--level`；
2. `Benchmark.run()` 初始化用户配置并调用 `set_shapes()`；
3. 基类按精确算子名、当前类/父类名称查找 YAML；
4. 找不到时使用 `DEFAULT_SHAPES`；
5. comprehensive 模式再追加 `set_more_shapes()`；
6. `get_input_iter()` 或 `input_fn` 将配置转换为真实 tensor。

因此，单算子修改应靠近该算子的 benchmark；只有明确需要批量影响时，才改类级
配置或 fallback。

## 修改前后的检查

- 确认 shape 描述与实际输入 rank、dtype 和 layout 一致；
- 运行前用 `--level core` 验证最小集合，再决定是否扩大到 comprehensive；
- 记录 `--shape_file`、warmup、iter、dtype、线程数和 CPU/NUMA 绑定；
- 不要把某台机器的 CPU 编号或 NUMA 节点写入共享 YAML；
- 如果 benchmark 自己生成输入，检查 `get_input_iter()`/`input_fn` 是否覆盖了
  YAML 的新字段。
