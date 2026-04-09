# DAS Architecture Efficiency and Dimensional Suppression Report

## Experiment Setup
- baseline layer: nn.Linear(1024, 1024)
- baseline profile: spectral_decay
- DAS rank: 32
- rotor steps: 16

## Parameter and Memory Suppression
| item | baseline | DAS |
|---|---:|---:|
| params | 1049600 | 66608 |
| param bytes | 4198400 | 266432 |
| param size | 4.003906 MiB | 0.254089 MiB |
| complexity | O(1024*1024) | O(rank*(1024+1024) + rotor_steps) |

- compression ratio: 15.7579x
- parameter reduction: 93.6540%

## Reconstruction Quality
- MSE: 1.187448e-02
- MAE: 8.661311e-02
- max abs error: 5.431747e-01
- relative Frobenius error: 6.112773e-01

## Riemann Sphere Structural Alignment
- target manifold: stereographic lift from Euclidean output space to unit Riemann sphere
- mean geodesic error: 2.711498e-01 rad
- p95 geodesic error: 3.713835e-01 rad
- interaction Gram relative Frobenius error: 8.694941e-02

## Inference Runtime
- compute device (request/resolved): auto -> cpu
- rotor kernel: staged
- torch.compile enabled: False
- matmul precision: medium
- selected torch threads: 2
- autotune timings (ms): {'scalar': 0.24011665000216453, 'staged': 0.10261874999741849, 'plan::cpu:t1:scalar': 0.2213430666718826, 'plan::cpu:t1:staged': 0.09013053334759509, 'plan::cpu:t2:scalar': 0.2203097000043878, 'plan::cpu:t2:staged': 0.08987779998885041, 'plan::cpu:t4:scalar': 0.2256346999881013, 'plan::cpu:t4:staged': 0.09254446667910088, 'plan::cpu:t6:scalar': 0.2423305666828431, 'plan::cpu:t6:staged': 0.10178053332007646, 'plan::cpu:t8:scalar': 0.24515553335125637, 'plan::cpu:t8:staged': 0.10099026667376165, 'plan::cpu:t10:scalar': 0.2481944333339925, 'plan::cpu:t10:staged': 0.10716390000501026, 'plan::mps:t1:scalar': 6.408377766638296, 'plan::mps:t1:staged': 0.820793033320418}
- device: cpu
- batch size: 128
- baseline mean latency: 0.326349 ms
- DAS mean latency: 0.188208 ms
- speedup (baseline / DAS): 1.733985x
- baseline peak memory: N/A
- DAS peak memory: N/A

## Memory-Wall and Friction Analysis
- DAS lazy-path inference avoids dense matrix unfolding and lowers weight traffic from O(N^2) to O(rank*N).
- On current GPUs, rotor/path kernels can face architectural friction because hardware is tuned for dense GEMM throughput.
- A future dedicated 3D reversible chip should fuse path contraction and rotor primitives in one on-chip dataflow.
