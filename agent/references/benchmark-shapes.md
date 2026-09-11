# FlagGems benchmark shape reference

FlagGems benchmark 的 shape 可能来自算子代码、YAML 配置、benchmark 类配置或
最终 fallback。修改前，先判断目标 benchmark 是否重写了 `set_shapes()`。

## 推荐修改顺序

### 1. Benchmark 已重写 `set_shapes()`

优先直接修改对应 benchmark 文件中的 `set_shapes()`。

如果该方法没有调用 `super().set_shapes()`，它通常会忽略 `--shape_file` 和
`core_shapes.yaml`。

例如：

- Flash Attention：`test_flash_attention_forward.py` 中
  `FlashAttentionForwardBenchmark.set_shapes()`。
- Fused MoE：`test_fused_moe.py` 中 `FusedMoEBenchmark.set_shapes()`。

若配置字段到实际张量 shape 的映射也要变化，同时修改该 benchmark 的
`get_input_iter()` 或 `input_fn`。

### 2. 为单个算子添加 YAML 配置

普通 benchmark 推荐在 `core_shapes.yaml` 中增加与 `op_name` 完全同名的配置：

```yaml
silu_and_mul:
  shapes:
    - [1024, 4096]
    - [8192, 4096]
  shape_desc: "M, N"
```

`op_name` 区分大小写，必须与创建 benchmark 时传入的名称一致。

该方式只影响同名算子，通常是默认单算子修改方式。若只想运行 YAML 中给出的
shape，建议使用 `--level core`，避免追加 comprehensive shape。

### 3. 修改一组算子

有两种方式：

- 自己编写 YAML，通过 `--shape_file <文件>` 指定。适合模型 shape、临时测试或
  一批明确算子。
- 修改对应 benchmark 类配置。可在 `core_shapes.yaml` 中添加类名配置，或修改
  `performance_utils.py` 中该类的 `set_more_shapes()`。

例如修改 `BlasBenchmark`，会影响所有没有算子专属配置、且继承该类的
benchmark。

注意：`set_more_shapes()` 只在 `--level comprehensive` 下追加。

### 4. 修改 fallback shape

Fallback 不建议作为单算子修改入口，因为影响范围较大。

两层 fallback：

- `core_shapes.yaml` 中的 `Benchmark`：多数普通 benchmark 的 YAML 兜底配置。
- `attri_util.py` 中 `DEFAULT_SHAPES`：YAML 中既无算子配置、也无继承类配置时
  使用。

## 正常读取流程

普通 benchmark 运行顺序：

1. `conftest.py` 中 `pytest_configure()` 读取 `--shape_file` 和 `--level`。
2. `performance_utils.py` 中 `Benchmark.run()` 调用 `init_user_config()`。
3. `init_user_config()` 调用 `set_shapes()`。
4. 基类 `Benchmark.set_shapes()` 按顺序查找：
   - 精确匹配 `op_name`；
   - 按 `type(self).__mro__` 查当前类和父类名称；
   - 使用 `DEFAULT_SHAPES`。
5. `--level comprehensive` 时，再调用 `set_more_shapes()` 追加额外 shape。
6. `get_input_iter()` 或 `input_fn` 将 shape 配置转换成实际输入张量。
7. `run()` 逐个 dtype、逐个输入执行 benchmark。

核心原则：先改离算子最近的位置；只有需要批量影响时，才修改类级或 fallback
配置。
