# Decode Runtime Final

## 最终结构

本轮完整模型 decode 主线收敛为：

`router → fixed GPU expert cache → hit 走 GPU expert → miss 走 Native CPU expert`

16 GiB 显存被作为一个全局预算，而不是把 Expert Pool 叠加到粗粒度 `-ngl 10` placement 上。固定模型、KV、workspace 与 expert pool共同竞争该预算。

## 三条独立基线

| 数据标签 | 基线 | 结果 | 解释 |
|---|---|---:|---|
| Measured | 原始粗粒度 layer offload，`-ngl 10` | ≈6.54 tok/s | 0–38 层的大量 attention / norm / shared-dense 等算子留在 CPU，是主要结构性问题 |
| Measured | Native expert-only CPU offload | ≈15–16 tok/s | 非 expert tensor 全 GPU，routed experts 位于 CPU/RAM；这是新的 Native baseline |
| Measured | Phase-Adaptive Hybrid Stage 1，8 GiB fixed pool | **18.753 tok/s** | canonical 16K、3×500 token，较同路径 Native 15.135 tok/s 提高 23.91% |

早期 routed-MoE prototype 的 26.921 ms/token 与 `-ngl 10` 的 llama.cpp wall time属于不同端到端执行路径，最终报告不再做直接相减。

## 动态缓存门禁

Stage 2 的相邻 token selective H2D 把 hit rate最高推到 61.78%，但最高动态吞吐仅 12.77 tok/s，低于 Stage 1。主因是每 token route D2H 同步与碎片化 expert slice H2D。该路径完成了真实集成和负门禁，默认关闭。

## 冻结项

- 默认 pool：固定 plan；最终容量由 Context-Adaptive Gate 选择；
- CUDA Graph：ON；
- CPU fallback：Native routed expert；
- 动态提升：实验项，默认阈值 -1；
- 正确性：Stage 1 前 32 token 序列一致；Stage 2 T0/T2 末端 logits cosine 0.999814702、argmax 一致。

## 下一问题

1. 在不同 context 下，KV 增长后哪一个固定 expert pool 给出最高且留有运行余量的端到端吞吐；
2. Cold Prefill 是否能从当前 352.433 tok/s 推进到 500+，再向 1000+ 演进。
