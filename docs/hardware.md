# Reference Hardware and Software

| Item | Reference configuration |
|---|---|
| CPU | Intel Core i9-14900KF |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| VRAM | 16,376 MiB reported by NVML |
| RAM | 96 GiB DDR5-4000 |
| PCIe | PCIe 4.0 x16, CPU-connected |
| OS | Windows |
| llama.cpp base | `4e97ac86ebe2c4cb8212d98d2641ad6768810896` |
| Local experiment checkpoint | `2d4f315` |
| Model architecture | Qwen4exp / Qwen3.8-Flash-Next family |
| Model parameters | 176.94B total, sparse top-10 of 512 experts per layer |
| Quantization | GGUF IQ4_XS family, mixed quantized tensors |
| Contexts | 2,027 / 8,171 / 16,363 / 32,747 actual tokens |
| Threads | 24 |
| CUDA Graph | ON except the explicit Graph-OFF control |

The model files are split GGUFs totaling about 87.24 GiB. Weights are excluded from this staging area and remain subject to their own distribution terms.

Background GPU use was low before the context sweep: 274 MiB used and 2% utilization. The system remained in its normal daily software state rather than a stripped benchmark OS.
