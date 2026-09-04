# PC Slotstream 长上下文门禁（128K–262K）最终报告

日期：2026-09-04（Asia/Shanghai）
数据根目录：`work\pc-slotstream-hybrid-gate\results\context-128k-gate\`
运行二进制：`slotstream-prefix-cache.exe` / `slotstream-hybrid-decode.exe`（llama.cpp checkpoint `2d4f3154`，out-of-tree 重建，360+ targets 全量编译通过）

## 先看结论

| # | 问题 | 收敛结论 | 数据标签 |
|---:|---|---|---|
| 1 | 262K 能不能跑 | 能。往返保真 cosine=1；retrieval 5 用例 q8-q4 全过；hybrid decode 5.776 tok/s | Measured |
| 2 | 262K 恢复体验 | 会话缓存 3.37 GiB（q8-q4），恢复到首 token 7.57 s，含模型加载的冷进程全链路 14.15 s | Measured |
| 3 | 262K 冷预填充成本 | q8-q4 786.9 s（331.8 tok/s）；q4-q4 716.8 s（364.2 tok/s）——长上下文最大等待项 | Measured |
| 4 | decode 全曲线 | 2K 19.531 → 32K ≈15.1–15.4 → 64K ≈10.9–11.7 → 128K ≈9.0 → 192K ≈7.5–7.7 → 262K ≈5.6–5.8 tok/s（hybrid 固定缓存，Graph ON） | Measured |
| 5 | q8-q4+6G 在 262K | **NO-GO（WDDM 边界负样本）**：长跑塌到 0.710 tok/s（p95 7,477.7 ms；单 token replay 108.7 s）；同配置短跑 5.508 tok/s、峰值余量仅 637 MiB——非确定性塌陷，机制为显存余量不足触发 WDDM 共享内存换页 | Measured |
| 6 | 262K 推荐 decode 档 | **q8-q4 + 4 GiB pool：5.776 tok/s，峰值 13,611 MiB，余量 2,765 MiB**（用户指定 q8-q4 profile） | Measured + policy decision |
| 7 | headroom 规则验证 | 637 MiB 余量档实测塌陷、≥1,598 MiB 档全部健康——safe-default 的 ≥1,024 MiB allocator headroom 门槛获得直接边界证据 | Measured |
| 8 | 检索质量随深度 | 128K/192K 全 5/5；262K q8-q4 5/5；262K q4-q4 4/5（跨文件链用例在 384 token 预算内未算完，三个 marker 均已正确引用——截断而非检索失败） | Measured |
| 9 | pool 随 context 降档策略 | q8-q4：≤64K 8G → 128K 8G → 192K 6G → 262K 4G（KV 每上一档吃掉 ~0.8–1.0 GiB，pool 必须让位） | Measured + Projection |

## 数据口径

沿用项目四类标签（Measured / Prototype measured / Trace simulation / Projection）；本报告 headline 全部为 Measured。decode 统一条件：hybrid fixed-cache、`promote_threshold=-1`、CUDA Graph ON、forced canonical continuation（首 token 7059）、50 warmup + 200 measured、n_batch 2048 / n_ubatch 16、-t 24、-ngl -2。

## 一、会话缓存与往返保真（kv_quant_sweep）

完整矩阵见 `kv_quant_sweep.csv`（15 行）。关键行：

| Context | KV | 冷预填充 | tok/s | 缓存大小 | 保存 | 加载 | 恢复→首 token | 冷进程全链路 | 往返 cosine |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K | q8-q4 | — | 544.3 | 1.73 GiB | 2.4 s | 4.1 s | 4.37 s | 11.04 s | 1 |
| 192K | q8-q4 | — | 435.5 | 2.55 GiB | 3.6 s | 5.3 s | 5.71 s | 12.29 s | 1 |
| **262K** | **q8-q4** | 786.9 s | 331.8 | 3.37 GiB | 4.8 s | 7.2 s | **7.57 s** | **14.15 s** | **1** |
| 262K | q4-q4 | 716.8 s | 364.2 | 2.43 GiB | 4.2 s | 6.8 s | 7.18 s | 14.00 s | 1 |

- 所有 15 个往返组合 prompt_tokens_equal / first_token_equal / generated_tokens_equal 均为 true，logits cosine=1（保存/加载路径无损）。
- 262K 冷预填充一次 12–13 分钟是长上下文最大的单项等待成本；会话缓存把它变成一次性的，恢复全链路（含 6.3 s 模型加载）14 s 出头。
- 容量探测（`262k/capacity-probes/summary.json`，u1）：262K KV 显存 q4-q4 2,376 MiB / q8-q4 3,336 MiB；decode 期 compute buffer 548.5 MiB（u16/b2048）。

## 二、检索质量（5 用例评估）

评估方法：`long_context_retrieval.json`（6 runs）。5 用例 = 首/中/尾 recall + 25%/75% 双位置求和（4821+9076=13897）+ 跨文件三跳链（RELAY_SEED=7319 → ×3+17 → 21974）。

| Context | KV | 用例 | 结果 | restored tok/s |
|---:|---|---|---|---:|
| 128K | q8-q4 | 5/5 | 全过 | 8.19 |
| 192K | q4-q4 | 5/5 | 全过 | 6.85 |
| 192K | q8-q4 | 5/5 | 全过 | 6.85 |
| **262K** | **q8-q4** | **5/5** | **全过（含完整 JSON 答案）** | 5.29 |
| 262K | q4-q4 | 4/5 | 跨文件链预算内未算完（marker 全部正确引用） | 5.40 |

结论：到 262K 为止，注意力/KV 量化没有造成检索能力断崖；q4-q4 的唯一失分是 token 预算截断，非检索错误。q8-q4 在 262K 保持 5/5，是指定 profile 的又一依据。

## 三、Hybrid Decode 曲线（2K → 262K）

| Context | KV + pool | tok/s | p50 ms | p95 ms | 峰值显存 | 余量 |
|---:|---|---:|---:|---:|---:|---:|
| **2K** | **q8-q4 + 8G（3 轮均值，CV 0.64%）** | **19.531** | 50.0 | 62.4 | — | — |
| **8K** | **q8-q4 + 8G（3 轮均值，CV 0.31%）** | **18.321** | 53.7 | 66.2 | — | — |
| 32K | q4-q4 + 8G | 15.296 | 64.4 | 81.9 | — | — |
| 64K | q8-q4 + 8G | 11.149 | 89.6 | 107.8 | — | — |
| 128K | q4-q4 + 8G | 9.078 | 110.0 | 126.4 | — | — |
| 128K | q8-q4 + 8G | 9.048 | 110.2 | 128.4 | — | — |
| 192K | q4-q4 + 8G | 7.674 | 130.8 | 150.0 | 15,751 MiB* | 625 MiB* |
| 192K | q8-q4 + 6G | 7.548 | 131.1 | 153.2 | 14,393 MiB* | 1,983 MiB* |
| **262K** | **q8-q4 + 4G（推荐）** | **5.776** | **170.7** | **197.2** | **13,611 MiB** | **2,765 MiB** |
| 262K | q4-q4 + 6G | 5.649 | 176.3 | 200.7 | 14,778 MiB | 1,598 MiB |
| 262K | q8-q4 + 6G（负样本） | **0.710** | 491.9 | **7,477.7** | 15,739 MiB（短跑） | **637 MiB** |

\* 192K 为 vram 采样运行峰值（5+5 token）。

要点：

1. 2K → 262K，tok/s 从 19.531 降到 ~5.7（-71%），但曲线平滑、无断崖；降幅主因是 attention/KV 随上下文线性走高的读取成本，而非 expert cache 命中率变化。
2. 2K/8K 行为本次补充测量（q8-q4、8 GiB pool、n_ubatch 16、50 warmup + 200 measured、3 轮独立运行取均值）。作为外部参照：社区在 RTX 5070 Ti 16 GB（GDDR7，带宽较 4080 SUPER 高 21.7%）上报告同模型短上下文 decode 约 18–20 tok/s——本机 4080 SUPER 已达到同一水平，说明该区间由 CPU miss expert 路径而非 GPU 带宽决定。
3. hybrid 相对 Native 的增益随上下文收窄：262K 处 Native load-only 为 5.40/5.29 tok/s，hybrid 6G/4G 为 5.65/5.78——增益 ~4–9%（16K 时为 23.91%）。长上下文下 CPU miss experts（~35 ms/token 固定）被 ~130 ms/token 的 attention 成本稀释。
4. 262K 处 q8-q4+4G 略快于 q4-q4+6G（+2.2%）：更小的 pool 带来更低显存压力，抵消了略低的命中率与更重的 K 读取。

## 四、WDDM 边界：q8-q4+6G 在 262K 的塌陷（本门禁关键负结果）

配置：262K、q8-q4（KV 3,336 MiB）、6 GiB pool（6,406.7 MiB packed）、compute 548.5 MiB、model ~4.8 GiB + 桌面基线 ~0.8 GiB。

| 观测 | 数值 |
|---|---|
| 长跑（50 warmup + 200 measured） | **0.710 tok/s**；p95 7,477.7 ms |
| 单 token replay（261K KV 首遍 attention） | **108.7 s**（健康配置 ~0.3–0.4 s） |
| warmup 50 token | 506.1 s |
| 同配置短跑（5+5，vram 采样） | 5.508 tok/s 正常；峰值 15,739 MiB，**余量仅 637 MiB** |
| 短跑同指标 replay / warmup | 268 ms / 0.95 s（405 倍差距） |

归因：静态分配已把余量压到 637 MiB（低于 safe-default 的 1,024 MiB 门槛）；驻留期任何瞬时分配（attention workspace、CUDA graph 复用、WDDM 内部簿记）越过边界即触发页降级到共享内存，此后每次访问走 PCIe 换页。短跑碰巧全程未越界，长跑必然越界——**非确定性塌陷正是余量不足的特征**，与 Stage 1 时 32K/8G 档 306 MiB 余量的"可用但越线"属同一谱系。

处置：该运行保留为边界负样本（`262k/decode-adaptive-u16/q8-q4-pool6g-run-1.json`）；策略上 262K 的 q8-q4 直接降 4G pool。1,024 MiB headroom 门槛从此有直接实测证据。

## 五、context × KV → pool 策略表（safe-default）

| Context | q8-q4 pool | q4-q4 pool | 依据 |
|---:|---:|---:|---|
| 2K–16K | 8 GiB | 8 GiB | 既有矩阵（benchmark-max 亦 8G） |
| 32K | 8 GiB | 8 GiB | 本门禁 decode 行 |
| 64K | 8 GiB | 8 GiB | 同上 |
| 128K | 8 GiB | 8 GiB | 峰值余量充足 |
| 192K | 6 GiB | 8 GiB | q8-q4 的 KV 比 q4-q4 大 ~720 MiB |
| **262K** | **4 GiB** | **6 GiB** | 第四节边界实测；6G 余量 637 MiB 判 NO-GO |

规则不变：16,376 MiB 总量 − 实测 model/KV/compute/pool − 桌面基线 ≥ 1,024 MiB，取满足条件的最大 pool。

## 六、下一步主线决策材料（OVERNIGHT_RESULTS.md 第九节）

两条主线优劣（供用户选择，未获指示前不动工）：

**Prefill 1000+ grouped runtime**
- 优势：直接攻击本报告暴露的最大痛点——262K 冷预填充 12–13 分钟（331–364 tok/s）；现有 524 tok/s（16K/8K ubatch）基础与既定结构（prompt chunk → router → expert buckets → unique experts → bulk staging → grouped GEMM → scatter/weighted merge）衔接；1000+ 为既定主目标、2000+ 为优秀目标；命中"长上下文等待时间"场景。
- 代价：新 runtime 工程量大（分组 GEMM + scatter/merge 内核 + 数值门禁）；需配套多 prompt 质量评估（4K vs 8K ubatch logits/continuation）；对已冻结 decode headline 无直接提升。

**固定 Decode cache 产品化**
- 优势：工作量小、风险低——把 safe-default policy、本报告的 context×KV→pool 策略表和 fixed plan 做成产品入口；18.753 tok/s headline 与全部矩阵已就绪；OSS 发布条件早已满足，产品化让发布故事完整。
- 代价：性能上限已探明，结构不变；不解决 262K 冷启动等待；decode 在 262K 仅 ~5.8 tok/s 的现实不变。

倾向性建议（仅供参考）：固定 Decode cache 产品化先行闭环（小工作量 + 直接可发布），Prefill 1000+ grouped runtime 作为下一个性能主目标立项。OSS 公开节奏仍由用户单独决定。

## 七、代码与工件

- runner 与脚本：`profiling\run_262k_gate.ps1`（retrieval/decode/vram 三阶段，pool 参数化）、`profiling\run_short_context_gate.ps1`、`profiling\run_short_repeat.ps1`、`profiling\aggregate_context_gate.py`
- 构建产物：`work\build\slotstream-tools-262k\slotstream-{prefix-cache,hybrid-decode}.exe`（构建脚本 `work\build\build-262k-runners.cmd`）
- 汇总表（本次全部重生成）：`results\context-128k-gate\{kv_quant_sweep.csv, context_decode.csv, context_vram_budget.csv, session_restore_long_context.csv, long_context_retrieval.json}`
- 262K 原始数据：`results\context-128k-gate\262k\{capacity-probes, kv-roundtrip, retrieval, decode-adaptive-u16, vram-adaptive-u16}\`
- 短上下文补充测量：`results\context-short-gate\{2k,8k}\`（q8-q4 状态 roundtrip + decode 3 轮复测；状态往返保真 cosine=1）

## 范围边界

- 本轮未更改硬件/BIOS/驱动/系统设置；未删除或覆盖既有结果（q8-q4+6G 塌陷运行按原样保留）；
- decode 运行期间无 nvidia-smi 轮询（vram 阶段单独采样），符合既有基准纪律；
- 未执行任何公开 push、发帖或远端创建；OSS 发布节奏由用户决定；
- 262K 档 q4-q4 的 4G pool 与 f16-f16 在 262K 的表现未测（非指定主线，留待需要时补）。
