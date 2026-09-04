# Hybrid Decode Runtime — Stage 2 Final

## 结论

**Stage 2 判定：NO-GO。** 相邻 token、同层路由驱动的 selective H2D / dynamic promotion 提高了 GPU expert hit rate，但在当前实现里没有转化为端到端吞吐收益。最终运行时默认保持 Stage 1 固定专家池，动态提升路径处于实验关闭状态。

## 同路径阈值扫描

统一条件：canonical 16K session、6 GiB expert pool、CUDA Graph ON、100 token warmup、100 token measured、同一 forced-token continuation。

| 数据标签 | 阈值 | 行为 | tok/s | p50 | p95 | hit rate | CPU fallback layers/token | H2D experts/token | H2D weight/token | promotion/token |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Measured | T0 | packed route stats only | **12.767** | 78.03 ms | 91.98 ms | 31.11% | 47.72 | 0.00 | 0.00 MB | 0.00 ms |
| Measured | T1 | promote miss-set ≤1 | 12.269 | 79.83 ms | 99.51 ms | 32.73% | 47.08 | 1.39 | 3.53 MB | 0.87 ms |
| Measured | T2 | promote miss-set ≤2 | 11.551 | 83.88 ms | 110.00 ms | 39.86% | 45.08 | 8.48 | 20.67 MB | 4.83 ms |
| Measured | T3 | promote miss-set ≤3 | 12.010 | 82.01 ms | 102.27 ms | 53.69% | 42.08 | 26.70 | 64.38 MB | 13.60 ms |
| Measured | T4 | promote miss-set ≤4 | 11.152 | 88.70 ms | 104.96 ms | **61.78%** | 40.11 | 47.55 | 114.45 MB | 22.89 ms |

T0 是这组带运行时计数路径中的最高吞吐，但仍显著低于 Stage 1 冻结结果 **18.753 tok/s**。T1–T4 全部低于 T0，因此停止更长的 3×500 promotion 稳定性运行；短门禁已经给出方向一致的负结果。

## 为什么命中率升高、速度反而下降

每个 decode token 的控制链是：

`GPU router → 48×10 route IDs 打包 → 一次 D2H 读取 → scheduler synchronize → CPU cache policy → 多个 expert slice H2D → map/mask/LRU 更新 → 下一 token 使用`

Stage 2 已经把 480 次离散 route 读取收敛成一次打包读取，但这一同步点仍切断 CPU 与 GPU 的流水。阈值提高后，又叠加三类成本：

1. **搬运量迅速增加**：T4 每 token 搬运 114.45 MB expert weights；
2. **搬运高度碎片化**：gate / up / down slices 分别更新，事务数量与命中收益一起上升；
3. **提升是滞后一拍的预测**：本 token 观察到的路由服务于下一 token，路由变化会造成提升后仍 miss 或很快被替换。

这说明本轮机制并非“PCIe 带宽不足”这一单变量问题，而是 **同步边界 + 碎片化事务 + 替换抖动** 的组合税。

## CUDA Graph

所有 T0–T4 运行均记录 `cuda_graphs=true`，日志显示 graph ID 被复用。Stage 2 没有通过全局关闭图来换取动态性；性能下降来自新增控制面，而非 Graph 退化为普通 launch。

## 数值正确性

T0 与 T2 使用同一 61-token forced continuation，比较 promotion window 末端 logits：

| 数据标签 | cosine | max abs | mean abs | RMSE | argmax |
|---|---:|---:|---:|---:|---:|
| Measured | **0.999814702** | 0.35210 | 0.03872 | 0.04908 | T0 = T2 = 2128 |

因此本轮负结论指向性能机制，而非明显的输出路径破坏。

## 性能归因边界

- **Measured**：阈值扫描、hit/miss、H2D 数量与字节、promotion 内部耗时、端到端 token wall time、logits 对比。
- Nsight Systems 2023.3.3 两次短采集均出现目标进程退出后 collector 保持运行，本轮没有形成可引用的 `.nsys-rep` 数值证据。
- Stage 2 归因使用轻量内部计数。`promotion_ms_per_token` 位于提升例程内部，和整 token wall time存在重叠与后续 GPU 节省关系，所以只做机制解释，不做逐项硬相加。

## 冻结决策

默认 Runtime 固化为：

`router → fixed cache lookup → GPU hit / Native CPU miss`

- expert pool 固定在总 VRAM 预算内；
- CUDA Graph 保持开启；
- `promote_threshold=-1` 为默认；
- T0 统计模式与 T1–T4 动态提升作为诊断/实验开关；
- bulk transfer、double buffer 与更复杂替换器不进入当前主线，因为最小动态机制已经低于固定缓存基线。

## 证据文件

- `results/hybrid-decode-runtime/stage2/threshold_sweep.csv`
- `results/hybrid-decode-runtime/stage2/attribution.json`
- `results/hybrid-decode-runtime/stage2/correctness/logits-comparison.json`
- `results/hybrid-decode-runtime/stage2/gate-6g-t0/t0.json`
- `results/hybrid-decode-runtime/stage2/gate-6g-t1/t1.json`
- `results/hybrid-decode-runtime/stage2/gate-6g-t2/t2.json`
- `results/hybrid-decode-runtime/stage2/gate-6g-t3/t3.json`
- `results/hybrid-decode-runtime/stage2/gate-6g-t4/t4.json`
