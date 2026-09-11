# Historical failure evidence

> Non-normative index derived from previously merged `triton-cpu` MRs. Use the
> current code and `failure-library.md` for decisions. MR numbers are search
> hints, not proof that an old patch still applies.

| Area | Representative MRs |
| --- | --- |
| CPU-only API and backend isolation | `!26`, `!27`, `!32`–`!34`, `!37`, `!39`, `!42`, `!49`, `!56`, `!81`, `!111`–`!113`, `!155`, `!162` |
| Numerical semantics, dtype, masks, layout | `!31`, `!35`, `!44`, `!54`, `!58`, `!60`, `!67`, `!69`, `!70`, `!82`, `!83`, `!96`, `!97`, `!119`, `!122`, `!123`, `!127`, `!129`, `!130`, `!132`, `!145`, `!154` |
| OOB, segfault, pointer/index analysis | `!64`, `!72`, `!86`, `!90`, `!93`, `!95`, `!98`, `!100`, `!110`, `!120`, `!134`, `!135`, `!149`, `!151`, `!161`, `!163` |
| TritonShared/MLIR/LLVM lowering | `!13`, `!15`, `!28`, `!40`, `!43`, `!47`, `!51`, `!53`, `!68`, `!90`, `!91`, `!100`, `!103`, `!105`, `!106`, `!109`, `!121`, `!134`, `!158`, `!159` |
| Bufferization and ownership | `!13`, `!28`, `!66`, `!71`, `!92`, `!102`, `!104` |
| FlagGems operators and test harness | `!4`, `!36`, `!38`, `!41`, `!45`, `!48`, `!59`, `!61`, `!63`, `!64`, `!75`, `!77`, `!78`, `!95`, `!98`, `!101`, `!108`, `!115`, `!125`, `!139`, `!166`, `!167`, `!170`, `!171`, `!173` |
| Concurrency, debug, compile/runtime performance | `!21`, `!25`, `!62`, `!79`, `!114`, `!136`, `!147`, `!148`, `!150`, `!152`, `!153`, `!157`, `!159`, `!170`–`!172` |
| Upstream sync, backport, and revert | `!1`–`!3`, `!18`, `!19`, `!22`, `!23`, `!50`, `!57`, `!73`, `!84`, `!87`, `!133` |

When consulting an MR, record the exact commit/diff and verify that the current
failure has the same layer, operation, dtype/layout, and backend conditions.
