# FlagGems Fusion Metrics 公式推理说明

本文件依据`fusion_metrics_formulas.yaml`中的算子列表，整理两块内容：

1. **调用流程**：benchmark 如何构造输入、经由哪些产品入口和调度层启动主 kernel。
2. **指标推导**：从算子语义和 kernel 主体推导可直接代入的
   `logical_flops` 或 `logical_bytes` 公式。

本文覆盖 YAML 列出的全部算子：6 个计算量算子和 22 个访存量算子。说明详细度按
实现复杂度区分：attention、归一化和多阶段 kernel 展开推导；单次逐元素计算或
纯搬运算子合并说明调用语义与公式。3 个当前禁用的公式项仍保留推导，并明确记录
禁用原因。

---

## 1. 公共声明

### 1.1 推理依据

YAML 决定需要覆盖的算子列表，并提供当前运行时表达式。文档公式按以下顺序核对：

1. benchmark 实际传入的参数、shape 和 dtype；
2. 产品入口的参数约束和分支；
3. 主 kernel 的计算或 load/store 语义；
4. YAML 表达式及其 helper。

若 YAML 与实现语义不一致，文档先按实现给出推理，再单独记录差异，不直接照抄
YAML。

### 1.2 指标分类

| 类别 | 输出字段 | 统计对象 |
|---|---|---|
| 计算量类 | `logical_flops` | 算子的主要矩阵计算 |
| 访存量类 | `logical_bytes` | 语义上的输入读取与输出写回 |

分类用于选择 benchmark 报告指标，不表示任意 shape 下已经完成实际 roofline
瓶颈判定。

### 1.3 统一计数规则

- 一次乘加计作 2 FLOPs。
- 计算量只统计主要矩阵工作；scale、mask、softmax、激活、索引和 reduction
  等次要工作默认不计，算子章节另有说明时除外。
- `logical_bytes` 使用实际 dtype 的元素字节数：
  `nbytes(T) = T.numel() * T.element_size()`。
- 后文记 $B(T)=\operatorname{nbytes}(T)$；若输出沿用 tensor $T$ 的 dtype，
  但元素数为 $n$，记 $B_T(n)=n\times T.\operatorname{element\_size}()$。
- 访存量包含输入 tensor 的逻辑读取和输出 tensor 的逻辑写回。寄存器中间值、
  未物化的融合结果不计。
- 若所选产品路径明确对同一外部 tensor 做两遍完整扫描，则每一遍都计入；kernel
  内因 tile、广播或实现方式产生的重复加载仍不据此放大。
- padding、tile、尾块 mask、Split-K、临时 buffer、cache 行为和硬件事务放大属于
  实现成本，不改变 logical metric。
- logical bytes 是统一比较口径，不是 profiler 测得的真实 DRAM 流量；真实流量
  可能受重复加载、缓存、对齐和硬件事务粒度影响。

---

## 2. 计算量类

### 2.1 `flash_attention_forward`

涉及源码：

- test_flash_attention_forward.py
- /src/flag_gems/ops/attention.py

#### 2.1.1 调用流程

benchmark 构造以下输入：

| Tensor | Shape | 含义 |
|---|---|---|
| `Q` | `[B, Lq, Hq, D]` | query |
| `K` | `[B, Lk, Hkv, D]` | key |
| `V` | `[B, Lk, Hkv, D]` | value |

调用路径：

```text
test_flash_attention_forward
  └─ FlashAttentionForwardBenchmark / GenericBenchmark
       └─ gems_flash_attention_forward(q, k, v, ...)
            └─ flag_gems.ops.flash_attention_forward(...)
                 ├─ 检查 Q/K/V head dimension
                 ├─ 必要时 padding 到支持的 D
                 └─ mha_fwd(...)
                      ├─ flash_fwd_kernel
                      └─ 或 flash_fwd_splitkv_kernel
                           └─ flash_fwd_splitkv_combine_kernel
```

`mha_fwd` 根据 shape、并行任务数和 dropout/local-attention 状态选择普通或
Split-KV 路径。调度方式不同，逻辑 attention 计算相同。

#### 2.1.2 主计算语义

对每个 batch 和 query head，kernel 执行两个主要矩阵计算：

```text
S = Q @ Kᵀ
O = softmax(S) @ V
```

`Hq` 可以大于 `Hkv`，但必须能被 `Hkv` 整除。GQA 路径将多个 query head
映射到同一个 KV head；每个 query head 仍独立计算输出，因此公式使用 `Hq`，
不再额外乘 `Hkv`。

#### 2.1.3 有效 attention pair 数

设 query token 下标为 $i$，key token 下标为 $j$。当 $L_q$ 与 $L_k$ 不同时，
实现采用右对齐：

$$
c(i) = i + L_k - L_q
$$

令 $lo(i)$ 和 $hi(i)$ 为包含端点的有效 key 范围：

| 模式 | $lo(i)$ | $hi(i)$ |
|---|---:|---:|
| 全量 attention | $0$ | $L_k - 1$ |
| 因果 attention | $0$ | $c(i)$ |
| 局部 attention | $c(i)-w_l$ | $c(i)+w_r$ |

其中 $w_l$、$w_r$ 分别表示 `window_left`、`window_right`。入口会把因果
attention 的 $w_r$ 归一为 0；若同时给出有限 $w_l$，有效范围为
$[c(i)-w_l, c(i)]$。

裁剪到 $[0,L_k-1]$ 后，第 $i$ 行的有效 pair 数为：

$$
p(i)=\max\left(0,\min(L_k-1,hi(i))-\max(0,lo(i))+1\right)
$$

单个 batch、单个 query head 的总 pair 数为：

$$
P=\sum_{i=0}^{L_q-1}p(i)
$$

常见特例：

$$
P_{\text{full}}=L_qL_k
$$

$$
P_{\text{causal}},\;L_q=L_k=L=\frac{L(L+1)}{2}
$$

#### 2.1.4 FLOPs 推导

| 阶段 | 计算 | Logical FLOPs |
|---|---|---:|
| QK | 每个有效 pair 做长度为 $D$ 的点积 | $2BH_qPD$ |
| PV | 每个有效 pair 向长度为 $D$ 的输出累加 | $2BH_qPD$ |

最终公式：

$$
\boxed{\text{logical\_flops}=4BH_qPD}
$$

#### 2.1.5 benchmark 实例

取 benchmark 中：

```text
B=1, Hq=1, Hkv=1, Lq=128, Lk=2048, D=128, causal=True
```

则：

$$
c(i)=i+1920
$$

$$
P=1921+1922+\cdots+2048=254016
$$

$$
\text{logical\_flops}
=4\times1\times1\times254016\times128
=130056192\ \text{FLOPs}
$$

#### 2.1.6 特性与边界

- scale、mask/ALiBi、softmax、dropout 和 LSE 不计入当前主矩阵 FLOPs。
- head-dimension padding 属于实现开销；公式中的 $D$ 取原始
  `query.shape[-1]`。
- 普通 kernel 与 Split-KV kernel 使用同一 logical FLOPs 公式。
- 当前 YAML 的总 FLOPs 结构与上述推导一致。
- `attention_pair_count` 对 causal 或 local window 均加入 $L_k-L_q$
  偏移，全量 attention 不加偏移，与 kernel 的右对齐语义一致。

### 2.2 `flash_attn_varlen_func`

涉及源码：

- [benchmark 入口](./test_flash_attn_varlen_func.py)
- [产品入口与 attention kernel 调度](../src/flag_gems/ops/attention.py)

#### 2.2.1 调用流程与输入

benchmark 的主要输入为：

| Tensor | Shape | 含义 |
|---|---|---|
| `query` | $[T_q,H_q,D]$ | 所有序列拼接后的 query |
| `key_cache` / `value_cache` | paged layout | KV cache |
| `cu_seqlens_q` | $[B+1]$ | query 累积长度 |
| `seqused_k` | $[B]$ | 每条序列实际使用的 key 长度 |

调用路径可简化为：

```text
test_flash_attn_varlen_func
  └─ flash_attn_varlen_func(...)
       └─ flag_gems attention varlen 入口
            └─ mha_varlan_fwd(...)
                 └─ paged/varlen attention kernel
```

block table 只决定逻辑 key/value 位于哪个物理 cache block，不改变有效
query-key pair 数。

#### 2.2.2 分序列 pair 计数

第 $b$ 条序列的长度为：

$$
L_{q,b}=\operatorname{cu\_seqlens\_q}[b+1]
        -\operatorname{cu\_seqlens\_q}[b]
$$

$$
L_{k,b}=\operatorname{seqused\_k}[b]
$$

对每条序列分别复用 2.1.3 的右对齐规则。记：

$$
P_b=P(L_{q,b},L_{k,b},\text{causal},w_l,w_r)
$$

则整个 batch 的有效 pair 数为：

$$
P_{\mathrm{all}}=\sum_{b=0}^{B-1}P_b
$$

不能使用 $B\times L_q\times L_k$ 代替该求和，因为每条序列的 query/key
长度可以不同。

#### 2.2.3 FLOPs 推导

每个有效 pair、每个 query head 分别执行长度为 $D$ 的 QK 点积和 PV 累加：

$$
\text{QK}=2H_qDP_{\mathrm{all}}
$$

$$
\text{PV}=2H_qDP_{\mathrm{all}}
$$

最终公式：

$$
\boxed{
\text{logical\_flops}
=4H_qD\sum_{b=0}^{B-1}P(L_{q,b},L_{k,b},\text{causal},w_l,w_r)
}
$$

#### 2.2.4 特性与边界

- `query.shape[0]` 是拼接后的总 token 数，不能直接当作单条序列长度。
- GQA 中 KV head 可被多个 query head 共享，公式仍按 $H_q$ 统计。
- softmax、scale、block-table 查表和 scheduler metadata 不计入主矩阵 FLOPs。
- 当前 benchmark 使用 `causal=True`、`window_size=(-1,-1)`；helper 同时支持
  causal/local 的右对齐 pair 计数。

### 2.3 `flash_mla`

涉及源码：

- [benchmark 入口](./test_flash_mla.py)
- [产品入口与 kernel](../src/flag_gems/fused/flash_mla.py)

#### 2.3.1 调用与主计算

benchmark 固定 $B=128$、$S_q=1$、$H_q=128$、$H_{kv}=1$，输入维度
$D=576$，value/content 维度 $D_v=512$。产品入口将 paged KV cache 展平，
再由 `flash_mla_attn_kernel` 对每个 batch 和 query head 执行：

```text
QK: query[D] · key[D]
PV: probability[Lk] · value[Lk, Dv]
```

其中 value 是 cache 每行的前 $D_v$ 个元素；剩余 $D-D_v$ 个元素只参与
QK 的 RoPE 部分。

#### 2.3.2 FLOPs 推导

记第 $b$ 条序列的有效 cache 长度为 $L_b$。当前 decode benchmark 中
$S_q=1$，所以每个 query head 的有效 pair 数是 $L_b$。两个矩阵阶段分别为：

$$
\text{QK}=2S_qH_qD\sum_bL_b
$$

$$
\text{PV}=2S_qH_qD_v\sum_bL_b
$$

因此当前 benchmark 公式为：

$$
\boxed{
\text{logical\_flops}
=2S_qH_q\left(\sum_bL_b\right)(D+D_v)
}
$$

benchmark 中 $L_b=L_0+2b$，故：

$$
\sum_{b=0}^{127}L_b=128L_0+16256
$$

可直接代入上式。

#### 2.3.3 特性与边界

- `block_table`、page padding 和 `max_seqlen_pad` 只影响 cache 布局，不计入
  logical FLOPs。
- $H_{kv}=1$ 的数据被 $H_q$ 个 query head 共享；每个 query head 仍有独立
  QK/PV，所以公式使用 $H_q$。
- 产品入口要求 `causal=True`，当前实现和 benchmark 实际面向 $S_q=1$ 的
  decode。若扩展到 $S_q>1$ 的右对齐因果 attention，严格 pair 数应为
  $S_qL_b-S_q(S_q-1)/2$（假设 $L_b\ge S_q$），不能直接沿用
  $S_qL_b$。当前 YAML 公式是 decode 范围公式，不应外推到该未覆盖场景。

### 2.4 `flash_mla_sparse_fwd`

涉及源码：

- [benchmark 入口](./test_flash_mla_sparse_fwd.py)
- [稀疏 MLA 主 kernel](../src/flag_gems/fused/DSA/sparse_mla.py)

#### 2.4.1 调用与维度

benchmark 输入为：

| Tensor | Shape |
|---|---|
| `q` | $[S_q,H_q,D_{qk}]$ |
| `kv` | $[S_{kv},H_{kv},D_{qk}]$ |
| `indices` | $[S_q,H_{kv},K]$ |

公共入口把无 batch 的输入视为 $B=1$，核心计算复用 sparse MLA kernel。
$K=\operatorname{indices.shape[-1]}$ 是每个 query/KV-head group 的候选容量。

#### 2.4.2 FLOPs 推导

每个有效候选 key 对应：

| 阶段 | 点积/累加长度 | FLOPs |
|---|---:|---:|
| QK | $D_{qk}$ | $2D_{qk}$ |
| PV | $D_v$ | $2D_v$ |

若所有 $K$ 个候选均有效，则 query-head pair 总数为 $S_qH_qK$，因此：

$$
\boxed{
\text{logical\_flops}
=2S_qH_qK(D_{qk}+D_v)
}
$$

更严格地，令 $K_{i,g}^{\mathrm{eff}}$ 表示 query $i$、KV-head group $g$
中通过下标范围和 causal mask 的候选数，$G=H_q/H_{kv}$，则语义有效工作量为：

$$
\text{logical\_flops}_{\mathrm{valid}}
=2G(D_{qk}+D_v)
\sum_{i=0}^{S_q-1}\sum_{g=0}^{H_{kv}-1}K_{i,g}^{\mathrm{eff}}
$$

#### 2.4.3 特性与边界

- 当前 YAML 使用配置容量 $K$，是所有候选均参与时的上界，也是稳定的 benchmark
  归一化口径。
- 负下标、越界下标、`topk_length` 和 causal mask 会使实际有效工作量低于该上界。
- `attn_sink`、softmax、LSE 和索引判断不计入主矩阵 FLOPs。

### 2.5 `sparse_mla_fwd_interface`

涉及源码：

- [benchmark 入口](./test_sparse_mla_fwd_interface.py)
- [产品接口与主 kernel](../src/flag_gems/fused/DSA/sparse_mla.py)

#### 2.5.1 调用与公式

该 benchmark 直接调用 `triton_sparse_mla_fwd_interface`，输入带显式 batch：

```text
q       : [B, Sq, Hq, Dqk]
kv      : [B, Skv, Hkv, Dqk]
indices : [B, Sq, Hkv, K]
output  : [B, Sq, Hq, Dv]
```

kernel 将 $D_{qk}$ 拆成前 $D_v$ 维和尾部 $D_{qk}-D_v$ 维做两次 QK dot，
两者合计仍为长度 $D_{qk}$ 的点积；PV 只在前 $D_v$ 维累加。所有 $K$ 个
候选均有效时：

$$
\boxed{
\text{logical\_flops}
=2BS_qH_qK(D_{qk}+D_v)
}
$$

严格有效候选公式为：

$$
\text{logical\_flops}_{\mathrm{valid}}
=2\frac{H_q}{H_{kv}}(D_{qk}+D_v)
\sum_{b,i,g}K_{b,i,g}^{\mathrm{eff}}
$$

benchmark 先把 `indices` 填充为越界值，再写入当前位置之前的随机下标；因此部分
shape 下 $K_{b,i,g}^{\mathrm{eff}}<K$。当前 YAML 与 2.4 一样使用候选容量上界。

### 2.6 `rwkv_mm_sparsity`

涉及源码：

- [benchmark 入口](./test_rwkv_mm_sparsity.py)
- [产品入口与 kernel](../src/flag_gems/fused/rwkv_mm_sparsity.py)

#### 2.6.1 主计算与公式

输入 $k$ 的 shape 为 $[M]$，$v$ 的 shape 为 $[M,N]$，输出等价于：

$$
o=v^Tk
$$

kernel 先判断 $k_i\ne0$，只为非零行加载 $v_{i,:}$。每个非零 $k_i$ 对
$N$ 个输出执行一次乘加，因此：

$$
\boxed{
\text{logical\_flops}=2\operatorname{nnz}(k)N
=2\operatorname{count\_nonzero}(k)\times v.\operatorname{shape}[1]
}
$$

零值判断、mask 和输出写回不计入计算量类的主矩阵 FLOPs。公式按输入的实际
非零数计算，而不是按 benchmark 设定的期望稀疏率估算。

---

## 3. 访存量类

### 3.1 `silu_and_mul`

涉及源码：

- [benchmark 入口](./test_silu_and_mul.py)
- [产品入口与 scalar kernel](../src/flag_gems/fused/silu_and_mul.py)

#### 3.1.1 调用流程

`silu_and_mul_kernel` 和 `silu_and_mul_grad_kernel` 由
`@pointwise_dynamic` 包装，不由 benchmark 直接启动。

forward/autograd 路径：

```text
test_silu_and_mul
  └─ GenericBenchmark
       └─ flag_gems.silu_and_mul(A, B)
            └─ SiluAndMul.apply(A, B)
                 └─ SiluAndMul.forward(ctx, A, B)
                      ├─ ctx.save_for_backward(A, B)
                      └─ silu_and_mul_kernel(A, B)
                           └─ PointwiseDynamicFunction.__call__
                                ├─ prepare_args
                                │    ├─ dtype promotion
                                │    ├─ 输出分配
                                │    ├─ broadcast/task shape 推导
                                │    └─ StridedBuffer 包装
                                ├─ instantiate(ndim)
                                │    └─ 生成或复用对应 rank 的 wrapper/kernel
                                ├─ overload(*args, **kwargs)
                                │    └─ 启动生成的 @triton.jit kernel
                                └─ _unwrap
```

`SiluAndMul.apply` 执行时立即进入 forward 并提交前向 kernel，不会等到 backward
才触发。设备侧是否同步完成由运行后端和 benchmark 计时同步逻辑决定。

另有两条相关路径：

- backward：用户调用 `loss.backward()` 或 `torch.autograd.grad()` 后，
  autograd 引擎进入 `SiluAndMul.backward`，再调用
  `silu_and_mul_grad_kernel(A, B, grad_output)`。
- out 变体：`silu_and_mul_out(A, B, out)` 直接调用
  `silu_and_mul_kernel(A, B, out0=out)`，绕过 `SiluAndMul.apply`。

当前公式只统计 `silu_and_mul` 的 forward。

#### 3.1.2 前向 kernel 语义

每个输出元素执行：

$$
O=\operatorname{silu}(A)\times B
=\frac{A}{1+\exp(-A)}\times B
$$

当前 benchmark 的 `binary_input_fn` 生成同 shape、同 dtype 的 `A` 和 `B`；
输出 `O` 也具有相同 shape 和 dtype。

#### 3.1.3 Logical bytes 推导

记：

- $N=\prod_i\text{shape}_i$：每个 tensor 的元素数；
- $b=\operatorname{sizeof}(\text{dtype})$：单元素字节数；
- BF16/FP16 的 $b=2$，FP32 的 $b=4$。

当前 benchmark 每个输出元素的语义搬运：

| 步骤 | 操作 | Logical bytes |
|---|---|---:|
| 读 `A[i]` | input load | $b$ |
| 读 `B[i]` | input load | $b$ |
| 计算 `silu(A[i]) * B[i]` | 寄存器内计算 | $0$ |
| 写 `O[i]` | output store | $b$ |

逐元素合计为 $3b$，因此：

$$
\boxed{\text{logical\_bytes}=3Nb}
$$

其中：

$$
\text{Load}=2Nb,\qquad \text{Store}=Nb
$$

这与当前 YAML 的表达式等价：

$$
2\times\operatorname{nbytes}(A)+\operatorname{nbytes}(B)=3Nb
$$

更一般地，若输入或输出的 shape/dtype 不同：

$$
\text{logical\_bytes}
=N_A b_A+N_B b_B+N_O b_O
=\operatorname{nbytes}(A)+\operatorname{nbytes}(B)+\operatorname{nbytes}(O)
$$


#### 3.1.4 benchmark 实例

取 shape `(64, 64)`、dtype BF16：

$$
N=64\times64=4096,\qquad b=2\ \text{bytes}
$$

$$
\text{logical\_bytes}
=3\times4096\times2
=24576\ \text{bytes}
=24\ \text{KiB}
$$

#### 3.1.5 特性与边界

- `A -> fp32`、`exp` 和 `x_silu` 是融合 kernel 内部中间值，不形成独立
  tensor 搬运。
- `ctx.save_for_backward(A, B)` 保存 tensor 引用，不复制输入；backward 的读取
  与梯度写回不计入本次 forward。
- `silu_and_mul_out` 把结果写入外部 `out`，无需预读 `out`；在同
  shape/dtype 前提下仍为 $3Nb$。

### 3.2 `apply_repetition_penalties`

涉及源码：

- [benchmark 入口](./test_apply_repetition_penalties.py)
- [产品入口与 kernel](../src/flag_gems/fused/apply_repetition_penalties.py)

#### 3.2.1 调用语义与公式

kernel 对二维 `logits[num_seqs, vocab_size]` 原地更新。每个 sequence 读取一个
penalty，每个词表位置读取 `prompt_mask`、`output_mask` 和原 logits，再将新 logits
写回原地址：

$$
\boxed{
\text{logical\_bytes}
=2B(\text{logits})
+B(\text{prompt\_mask})
+B(\text{output\_mask})
+B(\text{penalties})
}
$$

其中 $2B(\text{logits})$ 分别表示一次读取和一次原地写回。正数除以 penalty、
非正数乘以 penalty 的分支只影响寄存器内计算，不改变访存公式。

### 3.3 `apply_rotary_pos_emb`

涉及源码：

- [benchmark 入口](./test_apply_rotary_pos_emb.py)
- [产品入口与 kernel](../src/flag_gems/fused/rotary_embedding.py)

#### 3.3.1 调用语义

每个 query/key 元素与其旋转配对元素组合：

$$
y_i=x_i\cos_i+x_{\operatorname{rotate}(i)}\sin_i
$$

benchmark 未传 `position_ids`，并走默认的非原地路径，分别生成与 `q`、`k`
同 shape/dtype 的输出。

#### 3.3.2 公式

按外部 tensor 的逻辑读写计数：

$$
\boxed{
\text{logical\_bytes}
=2B(q)+2B(k)+B(\cos)+B(\sin)
}
$$

`q` 和 `k` 的系数 2 表示输入读取与输出写回。kernel 中同一输入元素也可能作为
另一个位置的 rotated value 再次 load；这属于实现加载方式，不按物理 load 次数
放大 logical bytes。若调用方提供 `position_ids`，完整外部输入口径还应加上
$B(\text{position\_ids})$；当前 benchmark/YAML 未覆盖该分支。原地模式同样需要
一次读和一次写，因此主项不变。

### 3.4 `concat_and_cache_mla`

涉及源码：

- [benchmark 入口](./test_concat_and_cache_mla.py)
- [产品入口与 kernel](../src/flag_gems/fused/concat_and_cache_mla.py)

#### 3.4.1 调用语义与公式

对每个有效 token，kernel 根据 `slot_mapping` 找到 cache 位置，将 `kv_c` 和
`k_pe` 沿最后一维拼接后写入 `kv_cache`。拼接结果不物化为独立 tensor。

记 $n_c=\operatorname{numel}(kv_c)$、$n_p=\operatorname{numel}(k_{pe})$，
写回使用 cache 的 dtype，则：

$$
\boxed{
\begin{aligned}
\text{logical\_bytes}={}&B(kv_c)+B(k_{pe})+B(\text{slot\_mapping})\\
&+B_{kv\_cache}(n_c+n_p)
+\mathbb{1}_{\text{cache dtype}\ne\text{auto}}B(\text{scale})
\end{aligned}
}
$$

benchmark 使用 `kv_cache_dtype="auto"`，所以不计 `scale`。若 `slot_mapping` 含
负值，kernel 会跳过相应 token；上式按 benchmark 中所有 slot 有效的情况统计。
cache 已有内容无需预读。

### 3.5 `cross_entropy_loss`

涉及源码：

- [benchmark 入口](./test_cross_entropy_loss.py)
- [产品入口与 kernel](../src/flag_gems/fused/cross_entropy_loss.py)

#### 3.5.1 输出规模

前向读取 `logits`、`target` 和可选 `weight`。reduction 决定最终输出元素数：

$$
N_o=
\begin{cases}
\operatorname{numel}(\text{target}), & \text{reduction}=\text{"none"}\\
1, & \text{reduction}\in\{\text{"mean"},\text{"sum"}\}
\end{cases}
$$

输出 dtype 与 logits 相同，所以：

$$
\boxed{
\text{logical\_bytes}
=B(\text{logits})+B(\text{target})+B(\text{weight})
+B_{\text{logits}}(N_o)
}
$$

当 `weight=None` 时定义 $B(\text{weight})=0$。softmax/log-sum-exp 的内部临时值、
归约中间 buffer 和为 backward 保存的统计量不计为外部 tensor 搬运。

### 3.6 `dgeglu`

涉及源码：

- [benchmark 入口](./test_dgeglu.py)
- [产品入口与 backward kernel](../src/flag_gems/fused/geglu.py)

#### 3.6.1 调用语义与公式

若前向输入最后一维为 $2H$，则 `grad_output` 最后一维为 $H$，输出
`grad_input` 与原输入同 shape。backward 读取梯度和原输入的两半，写回两半梯度：

$$
\boxed{
\text{logical\_bytes}
=B(\text{grad\_output})+B(\text{input})+B(\text{grad\_input})
=B(\text{grad\_output})+2B(\text{input})
}
$$

第二个等号使用 `grad_input` 与 `input` 同 shape/dtype 的实现约束。

### 3.7 `dreglu`

涉及源码：

- [benchmark 入口](./test_dreglu.py)
- [产品入口与 backward kernel](../src/flag_gems/fused/reglu.py)

#### 3.7.1 调用语义与公式

`dreglu` 与 `dgeglu` 的 tensor 规模相同，只是门控导数由 GELU 换成 ReLU。
因此：

$$
\boxed{
\text{logical\_bytes}
=B(\text{grad\_output})+2B(\text{input})
}
$$

ReLU mask 和两个半区梯度在同一 kernel 内计算，不产生额外外部中间 tensor。

### 3.8 `fused_add_rms_norm`

涉及源码：

- [benchmark 入口](./test_fused_add_rms_norm.py)
- [产品入口与两个 kernel 路径](../src/flag_gems/fused/fused_add_rms_norm.py)

#### 3.8.1 分支与读写次数

该算子原地更新 `x` 和 `residual`：先计算 $z=x+residual$ 并写回 residual，
再计算 $x=\operatorname{RMSNorm}(z)\times weight$。

记：

$$
N=\prod\text{normalized\_shape}
$$

| 路径 | `x` | `residual` | 原因 |
|---|---:|---:|---|
| $N<4096$ | 1 读 + 1 写 | 1 读 + 1 写 | 单遍 persistent kernel |
| $N\ge4096$ | 2 读 + 1 写 | 2 读 + 1 写 | 第一遍求方差，第二遍归一化并写回 |

权重按 logical tensor 口径读取一次。令：

$$
c(N)=
\begin{cases}
2,&N<4096\\
3,&N\ge4096
\end{cases}
$$

最终公式：

$$
\boxed{
\text{logical\_bytes}
=c(N)B(x)+c(N)B(\text{residual})+B(\text{weight})
}
$$

这里的大 shape 系数 3 来自实现中明确存在的两遍完整扫描，不是 tile padding 或
cache miss 的估算。

### 3.9 `geglu`

涉及源码：

- [benchmark 入口](./test_geglu.py)
- [产品入口与 kernel](../src/flag_gems/fused/geglu.py)

#### 3.9.1 调用语义与公式

输入最后一维被拆成两个等长部分 $x_a,x_b$，输出为
$\operatorname{GELU}(x_a)\times x_b$，所以输出元素数是输入的一半：

$$
\boxed{
\text{logical\_bytes}
=B(x)+B_x\left(\frac{\operatorname{numel}(x)}{2}\right)
}
$$

若输入和输出 dtype 相同，也可写为 $\frac{3}{2}B(x)$。

### 3.10 `gelu_and_mul`

涉及源码：

- [benchmark 入口](./test_gelu_and_mul.py)
- [产品入口与 scalar kernel](../src/flag_gems/fused/gelu_and_mul.py)

#### 3.10.1 调用语义与公式

两个同 shape 输入执行 $o=\operatorname{GELU}(x)\times y$，输出与 $x$
同 shape/dtype：

$$
\boxed{
\text{logical\_bytes}=2B(x)+B(y)
}
$$

两个 $B(x)$ 分别表示读取 `x` 和写回 `output`；GELU 的精确/近似分支不改变
外部 tensor 规模。

### 3.11 `get_scheduler_metadata`

涉及源码：

- [benchmark 入口](./test_get_scheduler_metadata.py)
- [产品入口、启发式与 metadata kernel](../src/flag_gems/ops/get_scheduler_metadata.py)

#### 3.11.1 调用语义

benchmark 的 tensor 输入只有 `seqused_k[B]`，其余主要参数是 Python 标量或
`None`。产品入口根据 sequence 长度、head 维度和设备启发式生成 int32 的
`scheduler_metadata`。

令 $S$ 为最终 split 数，动态 split 条件为 $B\le992$；再令
$\sigma=\mathbb{1}_{\text{arch}\ge90\ \lor\ S>1}$ 表示是否需要 semaphore。
输出元素数为：

$$
A=\sigma+B\mathbb{1}_{B\le992}
$$

第一项是 split semaphore，第二项是每条序列的动态 split metadata。在 CPU
路径中 `arch=0`，所以 $\sigma=\mathbb{1}_{S>1}$。

#### 3.11.2 实现推导公式与 YAML 差异

输出 dtype 固定为 int32，每元素 4 bytes，因此 benchmark 当前输入分支的完整
外部读写公式是：

$$
\boxed{
\text{logical\_bytes}=B(\text{seqused\_k})+4A
}
$$

当前 YAML 只有 $B(\text{seqused\_k})$，遗漏了 `scheduler_metadata` 的输出写回。
若后续 benchmark 传入 `seqused_q`、累积长度或 `leftpad_k` 等可选 tensor，还应
把实际读取的这些 tensor 加入公式。入口内部创建的 block-count 临时 tensor 不计。

### 3.12 `instance_norm`

涉及源码：

- [benchmark 入口](./test_instance_norm.py)
- [产品入口与 kernel 分支](../src/flag_gems/fused/instance_norm.py)

#### 3.12.1 空间规模与分支

输入 shape 为 $[B,C,\ldots]$，每个 instance/channel 的空间元素数为：

$$
N=\frac{\operatorname{numel}(x)}{B\times C}
$$

$N\le4096$ 时 persistent kernel 一次读取输入并写输出；$N>4096$ 时 loop
kernel 第一遍统计均值/方差，第二遍重新读取输入并写输出。定义：

$$
c_x(N)=
\begin{cases}
2,&N\le4096\\
3,&N>4096
\end{cases}
$$

#### 3.12.2 running stats 与最终公式

`use_input_stats=True` 且提供 running stats 时，更新 kernel 对 running mean/var
各读写一次；`use_input_stats=False` 时只读取它们。因此：

$$
c_r=
\begin{cases}
2,&\text{use\_input\_stats=True}\\
1,&\text{use\_input\_stats=False}
\end{cases}
$$

最终公式：

$$
\boxed{
\begin{aligned}
\text{logical\_bytes}={}&c_x(N)B(x)+B(\text{weight})+B(\text{bias})\\
&+c_rB(\text{running\_mean})+c_rB(\text{running\_var})
\end{aligned}
}
$$

`None` 的 tensor 按 0 bytes 处理。为 backward 保存的 `mean/rstd` 是实现内部
统计量，不纳入当前前向 logical bytes。

### 3.13 `moe_align_block_size_triton`（当前禁用）

涉及源码：

- [benchmark 入口](./test_moe_align_block_size_triton.py)
- [产品入口与四阶段 kernel](../src/flag_gems/fused/moe_align_block_size.py)

#### 3.13.1 外部 tensor 公式

算子读取 `topk_ids`，按 expert 对 token 排序并按 `block_size` padding，写入三个
调用方提供的输出 buffer：

$$
\boxed{
\begin{aligned}
\text{logical\_bytes}={}&B(\text{topk\_ids})+B(\text{sorted\_token\_ids})\\
&+B(\text{expert\_ids})+B(\text{num\_tokens\_post\_pad})
\end{aligned}
}
$$

该式按最终外部结果各写一次统计；四阶段实现中的 `tokens_cnts`、`cumsum` 和输出
预填充值属于内部/初始化成本，不在简化 logical 口径中展开。

#### 3.13.2 禁用原因

当前 YAML 将该项设为 disabled：benchmark 分配的输出 buffer capacity 与产品
路径要求尚未同步，可能不足。在修复输入/输出容量前保留公式，但不报告带宽指标。

### 3.14 `moe_sum`

涉及源码：

- [benchmark 入口](./test_moe_sum.py)
- [产品入口与 kernel](../src/flag_gems/fused/moe_sum.py)

#### 3.14.1 调用语义与公式

输入 shape 为 $[T,K,H]$，kernel 沿 top-k expert 维求和，写入预分配的
$[T,H]$ 输出：

$$
\boxed{
\text{logical\_bytes}=B(\text{input})+B(\text{output})
}
$$

累加器位于寄存器中；输出 buffer 不需预读。

### 3.15 `reglu`

涉及源码：

- [benchmark 入口](./test_reglu.py)
- [产品入口与 kernel](../src/flag_gems/fused/reglu.py)

#### 3.15.1 调用语义与公式

与 `geglu` 相同，输入最后一维拆成两个半区；差别仅是门函数改为 ReLU：

$$
o=\operatorname{ReLU}(x_a)\times x_b
$$

输出元素数是输入的一半，因此：

$$
\boxed{
\text{logical\_bytes}
=B(x)+B_x\left(\frac{\operatorname{numel}(x)}{2}\right)
}
$$

### 3.16 `reshape_and_cache`

涉及源码：

- [benchmark 入口](./test_reshape_and_cache.py)
- [产品入口与 kernel](../src/flag_gems/fused/reshape_and_cache.py)

#### 3.16.1 调用语义与公式

kernel 读取 `slot_mapping`，将每个 token 的 key/value 写入 paged cache 的指定
slot。cache 的 dtype 可能与输入不同，故写回字节数必须使用目标 cache 的
`element_size()`：

$$
\boxed{
\begin{aligned}
\text{logical\_bytes}={}&B(k)+B(v)+B(\text{slot\_mapping})\\
&+B_{k\_cache}(\operatorname{numel}(k))
+B_{v\_cache}(\operatorname{numel}(v))
\end{aligned}
}
$$

cache 旧值不需读取。当前 benchmark 使用 `kv_cache_dtype="auto"`；若启用量化
cache 并在 kernel 中读取 `k_scale/v_scale`，完整公式还应加对应 scale tensor
的读取字节数，当前 YAML 未覆盖该分支。

### 3.17 `reshape_and_cache_flash`

涉及源码：

- [benchmark 入口](./test_reshape_and_cache_flash.py)
- [产品入口与 kernel](../src/flag_gems/fused/reshape_and_cache_flash.py)

#### 3.17.1 调用语义与公式

该变体把 key/value 写入 FlashAttention 使用的 cache layout。物理下标计算不同，
但外部 tensor 读写与 3.16 相同：

$$
\boxed{
\begin{aligned}
\text{logical\_bytes}={}&B(k)+B(v)+B(\text{slot\_mapping})\\
&+B_{k\_cache}(\operatorname{numel}(k))
+B_{v\_cache}(\operatorname{numel}(v))
\end{aligned}
}
$$

block layout、stride 和 slot 到 block/offset 的换算不增加 logical bytes。

### 3.18 `rwkv_ka_fusion`

涉及源码：

- [benchmark 入口](./test_rwkv_ka_fusion.py)
- [产品入口与 kernel](../src/flag_gems/fused/rwkv_ka_fusion.py)

#### 3.18.1 主语义

benchmark 中 $k,a$ 的 shape 为 $[T,C]$，$kk,ka$ 为 $[C]$，并生成三个与
$k$ 同 shape/dtype 的输出：

$$
o_{kk}=\operatorname{normalize}(k\times kk)
$$

$$
o_k=k\times\left(1+(a-1)\times ka\right)
$$

$$
o_{kka}=o_{kk}\times a
$$

$o_{kk}$ 在 kernel 内直接复用于 $o_{kka}$，不作为额外外部读入。

#### 3.18.2 公式

输入读取为 $B(k)+B(kk)+B(a)+B(ka)$，三个输出共 $3B(k)$，所以：

$$
\boxed{
\text{logical\_bytes}
=4B(k)+B(kk)+B(a)+B(ka)
}
$$

归一化的平方和保存在寄存器中，不物化临时 tensor。

### 3.19 `silu_and_mul_out`

涉及源码：

- [benchmark 入口](./test_silu_and_mul.py)
- [产品入口与 scalar kernel](../src/flag_gems/fused/silu_and_mul.py)

#### 3.19.1 与 `silu_and_mul` 的差异

该接口由调用方预先分配 `out`，再通过 `out0=out` 传给同一个 pointwise wrapper。
将两个输入记作 $X,Y$；`out` 只写不读，因此与 3.1 的前向公式相同：

$$
\boxed{
\text{logical\_bytes}=B(X)+B(Y)+B(\text{out})
=2B(X)+B(Y)
}
$$

第二个等号使用 benchmark 中 `out` 与 $X$ 同 shape/dtype 的条件。预分配本身不计
为 kernel 的访存流量。

### 3.20 `skip_layer_norm`

涉及源码：

- [benchmark 入口](./test_skip_layer_norm.py)
- [产品入口与两个 kernel 路径](../src/flag_gems/fused/skip_layernorm.py)

#### 3.20.1 分支读写

算子计算 $y=\operatorname{LayerNorm}(x+residual)$。记
$N=\prod\text{normalized\_shape}$：

| 路径 | `x` 与输出 `y` | `residual` | 说明 |
|---|---:|---:|---|
| $N<4096$ | 1 读 + 1 写，共 2 份 | 1 读 | 单遍 kernel，不写 residual |
| $N\ge4096$ | 2 读 + 1 写，共 3 份 | 2 读 + 1 写，共 3 份 | 两遍扫描，第二遍还写 residual |

令：

$$
c_x(N)=\begin{cases}2,&N<4096\\3,&N\ge4096\end{cases},\qquad
c_r(N)=\begin{cases}1,&N<4096\\3,&N\ge4096\end{cases}
$$

最终公式：

$$
\boxed{
\text{logical\_bytes}
=c_x(N)B(x)+c_r(N)B(\text{residual})+B(\text{weight})+B(\text{bias})
}
$$

注意当前实现的 loop 路径会把 $x+residual$ 写回 `residual`，而小 shape 路径
不会；这也是两个分支 residual 系数不同的原因。

### 3.21 `topk_softmax`（当前禁用）

涉及源码：

- [benchmark 入口](./test_topk_softmax.py)
- [产品入口与 kernel](../src/flag_gems/fused/topk_softmax.py)

#### 3.21.1 外部 tensor 公式

kernel 读取 `gating_output[num_tokens, num_experts]`，对每行做 softmax 并选择
top-k，写入权重、expert 下标和 source-row 下标三个预分配 buffer：

$$
\boxed{
\begin{aligned}
\text{logical\_bytes}={}&B(\text{gating\_output})+B(\text{topk\_weights})\\
&+B(\text{topk\_indices})+B(\text{token\_expert\_indices})
\end{aligned}
}
$$

#### 3.21.2 禁用原因

benchmark 当前额外传入 `renormalize`，而产品 `topk_softmax` 接口只接收四个
tensor 参数，参数表不匹配。因此 YAML 将该项禁用；公式保留用于接口同步后的
覆盖，不在当前运行中报告。

### 3.22 `weight_norm`（当前禁用）

涉及源码：

- [benchmark 入口](./test_weight_norm_interface.py)
- [产品入口与 kernel](../src/flag_gems/ops/weightnorm.py)

#### 3.22.1 实现读写推导

前向先沿指定维度对 $v^2$ 求和得到 norm，再次扫描 $v$ 并计算：

$$
w=g\frac{v}{\lVert v\rVert}
$$

产品 kernel 对外部 $v$ 有两遍完整读取，并写出一个与 $v$ 同 shape/dtype 的
结果；$g$ 逻辑读取一次。因此按 1.3 的多遍扫描规则：

$$
\boxed{
\text{logical\_bytes}_{\mathrm{impl}}=3B(v)+B(g)
}
$$

内部 `norm` buffer 不作为外部 tensor 统计。当前 YAML 写为
$2B(v)+B(g)$，只包含一次语义输入读取和一次输出写回，少计了实现的第二遍
$v$ 扫描；文档采用上面的实现推导式。

#### 3.22.2 禁用原因

当前 accuracy coverage 已跳过，且 SVE lowering 路径失败，所以该公式项保持
disabled。修复执行路径后，还应同步 YAML 中的 $v$ 系数再启用带宽指标。

---

## 4. 覆盖与差异汇总

| 类别 | YAML 算子数 | 已展开 | 当前启用 | 当前禁用 |
|---|---:|---:|---:|---:|
| 计算量类 | 6 | 6 | 6 | 0 |
| 访存量类 | 22 | 22 | 19 | 3 |

需要在维护公式时重点保留的边界：

- `flash_mla` 当前公式限定 decode 的 $S_q=1$；多 token causal 场景需要有效
  pair 计数。
- 两个 sparse MLA 公式使用配置的 top-k 容量作为稳定上界；若要统计语义有效
  工作量，应改用通过下标和 causal mask 的实际候选数。
- `get_scheduler_metadata` 的实现推导式比 YAML 多输出 metadata 写回。
- `reshape_and_cache*` 的当前公式对应 benchmark 的 `kv_cache_dtype="auto"`；量化
  cache 分支需要加入 scale 读取。
- `weight_norm` 的实现推导式比 YAML 多一遍 $v$ 扫描，且该算子当前处于禁用状态。
