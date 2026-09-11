# Perf interpretation and report contract

Read this reference before turning perf output into a bottleneck conclusion.

## Three denominators

| Scope | Source | Denominator | Correct question |
|---|---|---|---|
| Process Self | `perf report --no-children` | All recorded process periods | Where was the sampled instruction pointer? |
| Process inclusive | `perf report` Children | All recorded process periods | How much process cost passed through this call tree? |
| Symbol local | `perf annotate ... local-period` | Periods attributed directly to one symbol | Which instructions dominate this symbol's Self? |

For a kernel row such as `Children=58%`, `Self=9%`:

- 9% is direct execution in the kernel symbol.
- 58% is kernel Self plus descendants.
- 42% is outside that kernel tree, subject to call-chain quality.
- Do not compute 67%; Children already includes Self.

Percentages from different denominators belong in separate table columns. A load
that is 40% kernel-local does not mean it is 40% of the pytest process. Its rough
process share would require the kernel Self share and matching period data; do not
multiply rounded percentages when exact periods are available.

## Period is the cost weight

`perf annotate --show-total-period --percent-type local-period` prints a period
weight at each sampled instruction. Sum the period column, then divide each
instruction or mnemonic family by the symbol total:

```text
instruction local share = instruction period / symbol period
family local share      = sum(family instruction periods) / symbol period
```

This is not the number of static instructions and not the fraction of raw sample
records. With `cycles:u`, period is sampled user-space CPU-cycle weight: a useful
time/cycle proxy, but not exact wall-clock duration. Sampling skid can place
period on a nearby instruction, especially around tight loops and branches;
interpret neighboring instructions as a hotspot region when appropriate.

## Caller, callee, and memory instructions

Perf Self attribution follows the current instruction address:

```text
kernel symbol
├── ldr/str at kernel addresses       -> kernel Self
└── bl helper
    └── ldr/str at helper addresses   -> helper Self
```

The helper Self is included in the kernel's inclusive Children if the call chain
shows the helper below the kernel. The `bl` call-site instruction does not absorb
the time spent in the helper. Consequently:

- Aggregate `ldr`, `ld1*`, `str`, and `st1*` separately inside each symbol.
- Do not move a child's memory periods into kernel Self.
- Explain both views when the user asks for “kernel total”: direct kernel code and
  the inclusive call tree.

`memrefCopy` is a common generated/runtime helper, but its meaning must be checked
against the actual call tree and implementation. Its loads/stores may include
descriptor, index, stride, stack, and data-copy traffic. They are not automatically
the original Triton `tl.load`/`tl.store` operations.

## Explaining the remainder

Compute the outside-tree share only when `CALLCHAIN` is recorded and usable:

```text
outside kernel tree = 100% - kernel Children
```

To decompose it, inspect callgraph roots that do not descend from the kernel.
Typical categories include input generation, JIT/compiler tools, Python/pytest,
allocation/GC, and runtime scheduling. These are hypotheses until the call chain
connects each symbol to that category.

Do not sum flat DSO or symbol rows to explain the remainder without checking their
parents. A libc/OpenMP/helper row can be inside the kernel tree. Inclusive rows
also overlap with ancestors and descendants, so they are not additive.

If symbols or stacks are unresolved, report the unresolved share and the resulting
limit. A precise disjoint decomposition may require a precompiled standalone
harness or profiling markers around only the measured operator/kernel interval.

## From hotspot to bottleneck conclusion

Use precise language:

- “Load/store instructions account for X% of kernel-local sampled cycles” is
  supported by annotation periods.
- “The kernel is memory-bandwidth-bound” additionally needs bandwidth, cache,
  IPC/stall counters, scaling, or a controlled code variant.
- “`memrefCopy` costs X% of the process and is a confirmed kernel child” needs
  both flat Self and call-chain evidence.
- “Input generation accounts for X% outside the kernel” needs a non-kernel call
  chain, not just a hot `randn`/`normal_fill` flat row.

Separate confirmed facts, interpretation, and follow-up hypotheses.

## Concise final report

Start with collection context and data quality:

| Shape | Event | Frequency | Call graph | Lost samples | Repeat status |
|---|---|---:|---|---:|---|
| `<shape>` | `cycles:u` | `<Hz>` | `dwarf/fp` | `<N>` | `stable/unstable/N/A` |

Then report attribution without mixing denominators:

| Shape | Scope | Symbol/instruction | Hotspot type | Period | Share | Denominator | Conclusion |
|---|---|---|---|---:|---:|---|---|
| `<shape>` | Kernel Self | `<address> ldr ...` | load | `<period>` | `<X%>` | kernel-local | `<fact>` |
| `<shape>` | Child Self | `memrefCopy: ldur ...` | load | `<period>` | `<Y%>` | child-local | `<fact>` |
| `<shape>` | Process Self | `memrefCopy` | helper | `<period>` | `<Z%>` | whole process | confirmed child |
| `<shape>` | Process inclusive | `<kernel>` | call tree | `N/A` | `<K%>` | whole process | includes Self and descendants |
| `<shape>` | Outside tree | `N/A` | setup/JIT/etc. | `N/A` | `<100-K%>` | whole process | decompose only with call chains |

Include the exact collection and analysis commands. Link each conclusion to the
raw report or annotation line when the delivery medium supports file links.

## Repetition and validity checks

A report is complete only when:

- Matching repeats use the same event, frequency, callgraph mode, shape, dtype,
  warmup/iterations, affinity, NUMA placement, cache policy, and source revision.
- Kernel symbol resolution succeeds and call chains exist for inclusive claims.
- Lost samples are zero or their effect is explained.
- Top hotspot ranking and shares are reasonably consistent; instability is
  reported rather than averaged away.
- Permission failures, unsupported events, missing symbols, and missing call
  chains appear as `N/A`, not zero.
- Generated artifacts stay outside source repositories.
