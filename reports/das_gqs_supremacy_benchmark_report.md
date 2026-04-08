# DAS-GQS Supremacy Basis-Crossing Benchmark

## Complexity Summary
- baseline memory: O(2^n)
- baseline time (single observable): O(2^n)
- DAS memory: O(n + e + d) where e is link count and d is local-rotor depth
- DAS memory on GHZ chain: O(n)

## Mandatory 20-Qubit Head-to-Head
- baseline ran: True
- baseline expectation: 0.0
- DAS expectation: 2.83276944882399e-16
- abs delta: 2.83276944882399e-16
- baseline est state bytes: 16.00 MB
- DAS est bytes: 944.00 B

## Scaling Table
| n | baseline_ran | baseline_exp | das_exp | abs_delta | baseline_state_bytes | das_bytes | baseline_time_s | das_time_s | skip_reason |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20 | True | 0.0 | 2.83276944882399e-16 | 2.83276944882399e-16 | 16777216 | 944 | 0.33643620799921337 | 0.0008982499966805335 | None |
| 21 | True | 0.0 | 2.83276944882399e-16 | 2.83276944882399e-16 | 33554432 | 992 | 0.6700554999952146 | 0.00011208300202270038 | None |
| 22 | True | 0.0 | 2.83276944882399e-16 | 2.83276944882399e-16 | 67108864 | 1040 | 1.3627784169984807 | 0.00011258400263614021 | None |
| 23 | True | 0.0 | 2.83276944882399e-16 | 2.83276944882399e-16 | 134217728 | 1088 | 2.635335873997974 | 0.0001139169980888255 | None |
| 24 | True | 0.0 | 2.83276944882399e-16 | 2.83276944882399e-16 | 268435456 | 1136 | 5.314678667000408 | 0.0002672499977052212 | None |
| 25 | True | 0.0 | 2.83276944882399e-16 | 2.83276944882399e-16 | 536870912 | 1184 | 10.834873292002158 | 0.00031045899959281087 | None |
| 26 | True | 0.0 | 2.83276944882399e-16 | 2.83276944882399e-16 | 1073741824 | 1232 | 22.56059704099971 | 0.0003492089999781456 | None |
| 27 | True | 0.0 | 2.83276944882399e-16 | 2.83276944882399e-16 | 2147483648 | 1280 | 47.28941641599886 | 0.00040116600575856864 | None |
| 28 | False | None | 2.83276944882399e-16 | None | 4294967296 | 1328 | None | 4.3207997805438936e-05 | state_vector_exceeds_memory_cap |
| 29 | False | None | 2.83276944882399e-16 | None | 8589934592 | 1376 | None | 3.749999814317562e-05 | state_vector_exceeds_memory_cap |
| 30 | False | None | 2.83276944882399e-16 | None | 17179869184 | 1424 | None | 3.5208999179303646e-05 | state_vector_exceeds_memory_cap |

## Interpretation Guardrail
- These results demonstrate a large practical scaling gap for this GHZ benchmark setup.
- They do not by themselves constitute a universal mathematical proof that all quantum-supremacy claims are basis artifacts.
- A universal claim would require formal equivalence and complexity proofs over broader circuit families beyond GHZ-like constructions.
