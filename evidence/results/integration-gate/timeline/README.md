# Hybrid Layer Timeline

Representative p50-nearest run (microseconds):

```text
activation_d2h          0.0 ->     24.5  (   24.5)
gpu_resident            0.0 ->    461.8  (  461.8)
gpu_h2d                 0.0 ->    365.5  (  365.5)
gpu_dequant           365.5 ->    438.3  (   72.8)
gpu_promoted          461.8 ->    562.2  (  100.4)
cpu                   437.2 ->    869.3  (  432.1)
cpu_return_merge      890.9 ->    939.2  (   48.4)
host_total             0.0 ->   1042.0
```

Critical path: host launch submission delays CPU start; the CPU cold branch finishes after the GPU branch, then CPU return/merge closes the p50 critical path
