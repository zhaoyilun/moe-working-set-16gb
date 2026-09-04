# Prefill 专家分窗 Staging —— 正式门禁报告

日期：2026-09-04（Asia/Shanghai）。数据根：`results\prefill-grouped-gate\`
二进制：checkpoint `2d4f3154` + staging 补丁（`LLAMA_PREFILL_STAGING=1` 开启；默认关闭=原路径）
机制：router 后锚点节点 → eval 回调 barrier → 该 ubatch 每层 unique experts 并行打包进 pinned arena → 大块 DMA 进共享 VRAM 窗口（角色×类型共 6 窗、512 slot、2,837.5 MiB）→ 改写映射 → GPU 分组 MMQ；大 batch（≥32 token）图只含 GPU 路径。机制侦破与设计限制见 CODE_NOTES §五–§十。

## 结论总览

| # | 问题 | 收敛结论 | 标签 |
|---:|---|---|---|
| 1 | 16K prefill | **725.667 tok/s**（3 轮均值，CV 0.13%；22.5 s）| Measured |
| 2 | 同口径对照 | 同 build 同 ubatch(6144) control 516.9 → **+40.4%**；对照最优配置(8192) 566.7 → **+28.1%** | Measured |
| 3 | 数值一致性 | 首 token argmax 与 8-token greedy 序列三轮全等；logits cosine 0.9948（16K=3 ubatch，KV 两轮传播放大；单 ubatch 2K 为 0.9999928）| Measured |
| 4 | decode 回归 | 无二进制回归：2K decode 18.979（旧二进制 19.531，-2.8% 环境内）；16K plan 模式 3×500 = 15.0 记为环境方差（连续 20 h 重载后）| Measured |
| 5 | 262K | **staging NO-GO**：窗口+KV 实测 15,840 MiB、余量 536 MiB < 1,024 安全线 → WDDM 蠕行（40 min 未完成已终止）；262K prefill 维持对照路径 331.8–364.2 tok/s | Measured |
| 6 | 增量路线（950+ 投影） | **结构性死亡**：每层专家权重独立，跨 ubatch 持久化需每层独立窗口 = 22–41 GiB | Measured（哨兵定凶）+ 分析 |

## 正式测量

| 配置 | 轮 | tok/s | 首 token |
|---|---:|---:|---|
| 16K/6144 staging | 1 | 726.048 | 248046 ✓ |
| 16K/6144 staging | 2 | 724.537 | 248046 ✓ |
| 16K/6144 staging | 3 | 726.416 | 248046 ✓ |
| 16K/6144 control（同 build） | 1 | 516.929 | 248046 ✓ |
| 16K/8192 control（历史最优配置） | — | 566.7 | 248046 ✓ |

适用范围：≤16K 上下文、ubatch 6144、f16 KV（实测组合）。32K 及以上未测；262K 已证 NO-GO（显存包络）。8192 ubatch 在 16K 亦超包络（WDDM）。

## 262K 边界记录

q4-q4/2048：总占用 15,840 MiB（模型 4.46 + 窗口 2.84 + KV 2.38 + compute/图 5+ + 基线 0.8），余量 536 MiB；运行 40 min 未完成（GPU 满载但吞吐病态），按 WDDM 边界判 NO-GO 并终止。缩小窗口（slot<384）则 unique 溢出（2048-ubatch p95 unique≈419），ubatch 降至 1024 又使搬运总量膨胀（~9.7 TB）——262K 档无可行 staging 配置。

## 产物

- 数据：`staging\final-16k-u6144-run{1,2,3}\`、`ab\control-u6144\`、`decode-regression\plan-8g-3x500\`、侦探工具全部输出（v* 系列）
- 脚本：`profiling\run_prefill_staging.ps1`（含 KV 量化参数）、`run_prefill_ab.ps1`、`run_hybrid_decode.ps1`
- 过程记录：`CODE_NOTES.md` §五–§十（机制、负实验、reserve 污染修复、双读裁决、哨兵定凶、delta 判决）
- 代码：llama.cpp-routing-trace 工作树（未提交；提交指引见 CODE_NOTES/会话记录——先清 mmq.cu 裸打印）

## 范围边界

- 未更改硬件/驱动/系统设置；262K staging 运行手动终止（唯一主动干预）；
- staging 模式为 prefill 专用配置（decode 走原 plan 模式）；
- 16K decode 15.0 的环境方差解释基于 2K 对照（18.98），发布前建议冷启动复测 decode headline；
- 四类证据标签沿用；headline 全部 Measured。
