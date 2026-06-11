# Flash Attention — Study Repo

## Context

This is Hriday's preparation repo for an ML systems research project with Prof. Jinyang Li.

**Background:** Completed the MLC (Machine Learning Compilation) course at mlc.ai. That covered TensorIR, Relax, IRModule, MetaSchedule, GPU kernel optimization, operator fusion, and hardware specialization (tensorization, external C kernels). Notes/code at https://github.com/FruitPnchSamuraiG/learning-ml-compilers.

**Current phase:** Preparation — deep-diving Flash Attention before the research project kicks off.

## Goals

1. Read and understand all 4 FlashAttention papers by Tri Dao
2. Read the forward and backward pass implementations in:
   - Triton
   - TileLang
   - CUDA Tile
3. Develop enough intuition to contribute to an ML systems research project

## The 4 Papers (in order)

- **FlashAttention v1** — Dao et al., 2022. IO-aware exact attention with tiling and recomputation.
- **FlashAttention v2** — Dao, 2023. Better parallelism, fewer non-matmul FLOPs, work partitioning.
- **FlashAttention v3** — Shah et al., 2024. Hopper GPU optimizations (WGMMA, TMA, pingpong scheduling).
- **FlashAttention-3 / ThunderKittens** — or the 4th paper as assigned by Prof. Li.

## Key Concepts to Build Up

- Why standard attention is memory-bandwidth bound (not compute bound)
- Tiling strategy: how blocks of Q, K, V are loaded to SRAM
- Online softmax: how running max/sum enable single-pass numerically stable softmax
- Recomputation in backward pass (trading compute for memory)
- How this maps onto GPU execution model (threads, warps, shared memory, global memory)

## Reference Implementation

Original repo: https://github.com/Dao-AILab/flash-attention
