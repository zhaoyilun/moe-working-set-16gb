# 16GB 独显如何运行 100GB+ MoE：从整层卸载到专家工作集虚拟化

## 一、问题的本质不是“模型放进哪张卡”

对于稀疏 MoE，完整权重很大，但每个 token 实际只访问少量专家。真正的问题因此不是把整套模型一次性塞进显存，而是把系统分成两层：96GB 内存保存完整容量，16GB 显存保存当前高频工作集。

这和操作系统的虚拟内存很像：容量层负责“都有”，工作集层负责“现在快”。差异在于，模型专家不是普通内存页；一次迁移还会影响量化布局、GPU kernel、CUDA Graph 和 CPU/GPU 同步。

## 二、为什么整层卸载只有 6.5 tok/s

最早的 `-ngl 10` 把 0–38 层整体留在 CPU。问题在于每层除了 routed experts，还有 attention、norm、router、shared/dense FFN 等适合 GPU 的算子。整层卸载把这些算子也一起带回 CPU，造成 148 ms/token 量级的 Decode。

结构感知 placement 只把 routed expert 权重留在内存，其余适合的 tensor 放到 GPU，同机 Decode 随即进入约 15–16 tok/s 区间。

## 三、显存应该保存“热专家”，而不是更多完整层

下一步是在显存中建立固定专家工作集：router 选出的专家若在池里，就走 GPU；其余仍走 Native CPU expert。16K canonical session 的三轮 500-token 实测为：

- 4 GiB：16.096 tok/s
- 6 GiB：17.275 tok/s
- 8 GiB：18.753 tok/s

8 GiB 相比同路径 Native 15.135 tok/s 提高 23.91%。同时，CUDA Graph ON/OFF 为 18.753 对 13.567 tok/s，说明“保留图复用”与“提高专家命中”同样重要。

## 四、命中率更高为什么反而更慢

动态 selective H2D 实验把 hit rate 从约 31% 推到约 62%，吞吐却降到 11–12.8 tok/s。原因是每个 token 都新增了一条控制链：读取路由、同步 CPU、做替换决策、分片搬运 gate/up/down 权重、更新映射，再进入下一 token。

这里得到一个可迁移规律：**缓存优化看的是端到端服务时间，而不是单独命中率。** 当 miss 处理破坏流水或 CUDA Graph 时，更高命中率也可能对应更低吞吐。

## 五、Context 与 Expert Pool 争夺同一块显存

2K 到 32K 的 4/6/8 GiB 矩阵显示，8 GiB 始终是短跑吞吐冠军；但 16K 的设备实测余量只有 386 MiB，32K 的 allocator 余量只有 306 MiB。因此最终策略分两档：

- 追求短跑峰值：8 GiB；
- 日常安全默认：2K/8K 用 8 GiB，16K/32K 用 6 GiB。

这就是 Context–Expert Cache Coupling：KV 增长会直接压缩专家工作集空间。

## 六、Prefill 与 Decode 应该采用不同策略

Decode 适合长期保存热专家；Prefill 更适合让大量 prompt token 共享一次专家权重搬运。把 physical ubatch 从 2K 扩到 8K 后，同一 16K prompt 的吞吐从 287.387 提升到三轮均值 524.003 tok/s，Cold Prefill 约 31.23 秒。32K 单轮为 617.342 tok/s。

下一阶段若要向 1000+ 推进，核心结构应是：token 按 expert 聚桶、权重连续搬运、grouped GEMM、最后做 weighted scatter/merge。

## 七、当前结论

普通 PC 上运行超显存 MoE 的关键抽象已经从“offload 几层”变成了 **Model Working-Set Virtualization**：

```text
RAM = 完整模型容量
VRAM = 高速工作集
GPU = 主计算
CPU = 冷专家回退
Prefill / Decode = 两套阶段策略
```

当前最有价值的结果不是某个单点数字，而是三条边界都已经实测：固定专家池有收益；动态调度可能被同步税反噬；长上下文会通过 KV 与 Expert Pool 的竞争改变最优配置。
