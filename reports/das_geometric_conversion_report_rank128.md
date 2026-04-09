# DAS Architecture Efficiency and Dimensional Suppression Report

## Experiment Setup
- baseline layer: nn.Linear(1024, 1024)
- baseline profile: spectral_decay
- DAS rank: 128
- rotor steps: 32

## Parameter and Memory Suppression
| item | baseline | DAS |
|---|---:|---:|
| params | 1049600 | 263328 |
| param bytes | 4198400 | 1053312 |
| param size | 4.003906 MiB | 1.004517 MiB |
| complexity | O(1024*1024) | O(rank*(1024+1024) + rotor_steps) |

- compression ratio: 3.9859x
- parameter reduction: 74.9116%

## Reconstruction Quality
- MSE: 9.871845e-04
- MAE: 2.499609e-02
- max abs error: 1.721723e-01
- relative Frobenius error: 1.762810e-01

## Riemann Sphere Structural Alignment
- target manifold: stereographic lift from Euclidean output space to unit Riemann sphere
- mean geodesic error: 6.318342e-02 rad
- p95 geodesic error: 8.354236e-02 rad
- interaction Gram relative Frobenius error: 4.991035e-03

## Inference Runtime
- compute device (request/resolved): auto -> cpu
- rotor kernel: staged
- torch.compile enabled: False
- matmul precision: medium
- selected torch threads: 1
- autotune timings (ms): {'scalar': 0.49705835003805987, 'staged': 0.1338332999694103, 'plan::cpu:t1:scalar': 0.417647233310466, 'plan::cpu:t1:staged': 0.1282861000011811, 'plan::cpu:t2:scalar': 0.42770000000018626, 'plan::cpu:t2:staged': 0.13416803330983385, 'plan::cpu:t4:scalar': 0.4316041666849439, 'plan::cpu:t4:staged': 0.14345136666330896, 'plan::cpu:t6:scalar': 0.43791666663916357, 'plan::cpu:t6:staged': 0.15114860001024985, 'plan::cpu:t8:scalar': 0.44771109999904485, 'plan::cpu:t8:staged': 0.1534916666666201, 'plan::cpu:t10:scalar': 0.4534139000194652, 'plan::cpu:t10:staged': 0.14729166665953622, 'plan::mps:t1:scalar': 13.148688899976454, 'plan::mps:t1:staged': 1.5783958333486225}
- device: cpu
- batch size: 128
- baseline mean latency: 0.269433 ms
- DAS mean latency: 0.142244 ms
- speedup (baseline / DAS): 1.894164x
- baseline peak memory: N/A
- DAS peak memory: N/A

## Memory-Wall and Friction Analysis
- DAS lazy-path inference avoids dense matrix unfolding and lowers weight traffic from O(N^2) to O(rank*N).
- On current GPUs, rotor/path kernels can face architectural friction because hardware is tuned for dense GEMM throughput.
- A future dedicated 3D reversible chip should fuse path contraction and rotor primitives in one on-chip dataflow.
