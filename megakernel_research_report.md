# Megakernel 研究综述：从 Karpathy 到 2026，技术脉络、开源实现与业界落地

> 研究时间：2026-10-26  
> 研究范围：学术论文、开源实现、工业界案例、硬件适配

---

## Executive Summary

**Megakernel 是什么**：Megakernel 将 GPU 上前百个独立 kernel launch 融合为一个持久运行的巨型 kernel，由 GPU 内部通过指令解释器或调度器控制 SM 执行，消除 kernel launch 开销、内存管道气泡和全局同步屏障，让不同硬件资源（HBM 带宽、Tensor Core、NVLink）在 SM 粒度上同时被使用。

**Karpathy 的角色**：Andrej Karpathy 并未直接撰写 megakernel 论文，但他在 2026 年 3 月发布的 [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) 仓库（"agent 改文件 → 固定评估 → keep/revert" 循环）开创了 **"agent 驱动的迭代优化范式"**，这直接催生了 AutoKernel、ForgeMegakernel、AutoMegaKernel 等一系列将 LLM agent 与 megakernel 合成结合的研究线。

**当前状态（2026 年底）**：megakernel 已从纯学术探索进入工程化阶段，形成三大派系：
1. **手动设计 + 指令解释器**（Hazy Research ThunderKittens / Megakernels）
2. **编译器自动生成**（Event Tensor / MPK / GPUOS）
3. **Agent 自动合成**（AutoKernel / ForgeMegakernel / AutoMegaKernel）

实测效果：H100 上 Llama-1B 延迟降低 1.5-2.5x，Llama-70B 吞吐提升 22%，MoE decode 提速 1.23x，微批处理提升至 **8.7x GPU 利用率**。

---

## 一、Karpathy 原始贡献与"agent 驱动 megakernel"学术线

### 1.1 Karpathy 的 Autoresearch：范式而非直接论文

2026 年 3 月 7 日，Karpathy 在 GitHub 发布 [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) 仓库（MIT License，72k+ stars）[cite:1]。核心机制极简：

```
agent 编辑 train.py（固定范围）
    → git commit
    → 固定 5 分钟评估
    → 若 val_bpb 提升则 keep，否则 git reset --hard
    → 循环
```

在单张 GPU 上，agent 2 天内跑了 **700 次实验**，发现了包括注意力实现 bug 在内的 **20 项训练优化**[cite:2]。这是 megakernel 学术线的**范式触发点**——证明了"固定评估 harness + 可控 agent 循环"能在单 GPU 上自主发现工程优化。

### 1.2 直接承接 Autoresearch 的 megakernel 研究

| 论文 | 年份 | 与 Karpathy 的关系 | arxiv / 来源 |
|------|------|-------------------|-------------|
| **AutoKernel** | 2026 | 直接引用 Karpathy autoresearch，将"agent 循环"应用于 GPU kernel 优化（Triton / CUDA C++） | arxiv:2603.21331 |
| **ForgeMegakernel** | 2026 | 用 coding agent 生成 per-model decode megakernel，配 10 个渐进里程碑知识库 | arxiv:2609.12379 |
| **AutoMegaKernel (AMK)** | 2026 | 静态校验 schedule-IR + autoresearch 循环，自优化 megakernel；int4/int8 量化 | arxiv:2606.09682 |
| **Kernel Forge** | 2026 | MCTS 多路径搜索，GUI 监控，支持 vision/diffusion/LLM 三类 workload | arxiv:2607.24762 |
| **KernelEvolve (Meta)** | 2026 | Meta 内部系统，MCTS + 多工具 profiling（TritonBench/NCU/Proton） | ISCA 2026 |

这些工作共享同一范式：**agent 修改一个可控 scope 的 kernel 文件/IR → 固定评估 harness → keep/revert**。

### 1.3 手动设计派（与 Karpathy 无直接关系，但同属 megakernel 领域）

| 研究 | 年份 | 核心贡献 |
|------|------|---------|
| **Hazy Research Low-Latency Megakernel** | 2025.05 | 首个 Llama-1B 全融合 megakernel，H100 达到 78% 内存带宽 |
| **ThunderMLA** | 2025.03 | FlashMLA 的 megakernel 版本，20-35% 更快 |
| **Hazy Tensor-Parallel Llama-70B** | 2025.09 | 8-GPU TP megakernel，超 SGLang 22% 吞吐 |
| **ThunderKittens / ThunderKittens 2.0** | 2024.10 / 2026.02 | CUDA-embedded DSL，megakernel 指令模板基础设施 |
| **MonoMoE (FlashInfer)** | 2026.09 | 量化 MoE decode megakernel，FlashInfer v0.7.0 集成 |
| **Mixture-of-Kittens** | 2026 | NVL72 MoE 训练 megakernel |

### 1.4 编译器自动生成派

| 研究 | 年份 | 核心贡献 | 出处 |
|------|------|---------|------|
| **Event Tensor / ETC** | 2026 | 统一抽象编码 SM 粒度依赖，静态+动态调度生成 persistent kernel | MLSys 2026, arxiv:2604.13327 |
| **Mirage Persistent Kernel (MPK)** | 2025 | SM-level graph 自动将多 GPU inference 转为单个 mega-kernel | arxiv:2512.22219 |
| **GPUOS** | 2026 | persistent runtime + NVRTC JIT operator injection，透明融合任意 PyTorch 代码 | arxiv:2604.17861 |
| **Ada-MK** | 2026 | MLIR DAG 离线搜索 + 3D shared-memory 约束，TensorRT-LLM 混合嵌入 | arxiv:2605.11581 |
| **Fleet** | 2026 | 多级 task abstraction（wavefront → CU → Chiplet → Device），chiplet-aware 调度 | arxiv:2604.15379 |
| **RaMP** | 2026 | Runtime-Aware Megakernel Polymorphism，MoE 根据 batch size + expert routing 选择配置 | arxiv:2604.26039 |
| **Perseus** | 2026 | 消除多节点 megakernel communication 的 hidden serialization | arxiv:2605.00686 |

### 1.5 学术前史："Persistent Kernel" 传统

Megakernel 并非全新概念。可追溯到 GPU 早期：

| 年代 | 论文/系统 | 贡献 |
|------|----------|------|
| **2009** | Aila & Laine (NVIDIA) [cite:3] | 首次提出 persistent threads，用 BVH 遍历 ray tracing，单 kernel 跑所有任务 |
| **2010** | NVIDIA OptiX | 将 persistent kernel 状态机模型产品化，支持函数调用/递归/异常 |
| **2012** | Gupta et al. | unstructured grid solver 上验证 persistent kernel 的动态工作生成 |
| **2022** | PERKS [cite:4] | 将 time loop 移进 persistent kernel，消除 host 同步开销 |

**与 LLM megakernel 的区别**：早期 persistent kernel 主要面向不规则科学计算（ray tracing、流体模拟），解决动态负载均衡问题；LLM megakernel 解决的是**静态计算图上的 kernel launch overhead + 跨操作 pipeline 机会**，目标更明确，规模更大（全模型融合）。

---

## 二、开源实现与对应论文

### 2.1 完整开源的 megakernel 项目

| 项目 | GitHub | 对应论文 | 后端 | 状态 |
|------|--------|---------|------|------|
| **HazyResearch/Megakernels** | 12.7k ★ | [Look Ma, No Bubbles!](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [TP Llama-70B](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main) | CUDA (ThunderKittens DSL) | 活跃维护 |
| **ThunderKittens** | 7.9k ★ | arxiv:2410.20399, 2026.02 TK 2.0 blog | CUDA-embedded DSL | 活跃维护 |
| **AutoKernel** | 880 ★ | arxiv:2603.21331 | Triton + CUDA C++ | 活跃维护 |
| **Event Tensor Compiler (ETC)** | et-compiler (CMU) | MLSys 2026, arxiv:2604.13327 | 自研编译器 | 活跃维护 |
| **FlashInfer (MegaMoE)** | 24.6k ★ | MonoMoE arxiv:2609.04244 | CUDA + CuTe DSL | 商业级产品 |
| **MPK (Mirage)** | 开源编译器 | arxiv:2512.22219 | TVM/CUTLASS | 研究原型 |
| **Fleet** | 开源 | arxiv:2604.15379 | HIP (AMD) | 研究原型 |

### 2.2 主要组件架构

**Hazy Research Megakernels**（最完整的参考实现）：

```
┌──────────────────────────────────────────────────┐
│        per-SM Instruction Queue (pre-scheduled)    │
│   [RMS+QKV] → [Attn] → [O+Res] → ...              │
└────────────┬─────────────────────────────────────┘
             │ launch single persistent kernel
             ▼
┌──────────────────────────────────────────────────┐
│  On-GPU Interpreter (CUDA Template)               │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐   │
│  │ Load   │  │ Compute│  │ Store  │  │ Pager  │   │
│  │ Stage  │  │ Stage  │  │ Stage  │  │ Stage  │   │
│  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘   │
│      │           │           │           │        │
│  shared memory pages (paged, 13 × 16KiB on H100) │
│  global memory counters (dependency wait/signal)   │
└──────────────────────────────────────────────────┘
```

**Event Tensor (ETC)**（编译器方案）：
- **抽象层**：Event Tensor = 带依赖关系的 tile task tensor
- **调度层**：静态调度（低延迟）+ 动态调度（MoE 等不规则场景）
- **运行时**：最小化 runtime，依赖 counter 驱动 SM 工作循环
- **硬件目标**：Hopper/H100、Grace Hopper、AMD MI300X

**GPUOS**（运行时方案）：
- **架构**：`init()` → launch persistent worker kernel；`fuse()` 将小操作组合后注入
- **动态算子更新**：NVRTC JIT + dual-slot aliasing scheme，kernel 不重启即可换算子
- **兼容性**：透明接入 PyTorch，支持任意 shape/stride/dtype/broadcasting

**AutoKernel / ForgeMegakernel / AutoMegaKernel**（agent 驱动）：
- **共享骨架**：固定 evaluation harness + 单文件 kernel scope + git commit/revert
- **AutoKernel**：Triton / CUDA C++，profile → extract → orchestrate（Amdahl 定律调度）
- **ForgeMegakernel**：10 个渐进里程碑（SM 指令流 → 细粒度依赖 counter → shared-memory buffer pool）
- **AutoMegaKernel**：静态 schedule-IR 校验器（1150 行），deadlock-free + race-free 保证

### 2.3 Megakernel 执行模型对比

| 特性 | Hazy 手动 | ETC 编译器 | GPUOS 运行时 | Agent 驱动 |
|------|----------|-----------|-------------|-----------|
| 需要手动编码 | 是 | 否（自动生成） | 否（透明注入） | 否（agent 生成） |
| 动态 shape 支持 | 有限 | 原生（Event Tensor） | 原生（通用 tensor） | 取决于 harness |
| 多 GPU 支持 | 是（NVLink overlap） | 是（SM-level graph） | 否（单 GPU 设计） | 是（profile → inject） |
| 正确性保证 | 人工校验 | 编译期校验 | runtime validator | 5-stage correctness harness |
| 跨模型可复用 | 低（per-model 手动） | 高（编译时适配） | 极高（运行时通用） | 高（agent 自主适配） |

---

## 三、业界使用案例与实际效果

### 3.1 学术基准上的实测结果

#### H100 上低延迟 Megakernel（Llama-3.2-1B, batch=1, bfloat16）[cite:5]

| 系统 | 延迟 (μs / forward pass) | 相对 vLLM | 相对 SGLang |
|------|------------------------|-----------|-------------|
| vLLM (CUDA graphs) | ~1,000+ | 1.0x | — |
| SGLang | ~800 | — | 1.0x |
| **Hazy Megakernel** | **~400** | **2.5x faster** | **1.5x faster** |
| **理论上限** (memory-bandwidth only) | ~740 | — | — |

关键指标：H100 上达到 **78% 内存带宽利用率**（vLLM/SGLang ≤ 50%），首次实现 1B+ LLM forward pass 低于 **1ms**。

#### B200 上 Llama-1B

| 系统 | 延迟 |
|------|------|
| **Hazy Megakernel** | **< 680 μs / forward pass** |

#### Llama-70B TP Megakernel（8×H100s, throughput 优化）[cite:6]

| 系统 | 吞吐 (tokens/s, ShareGPT 65,536 prompts) | 相对 SGLang |
|------|----------------------------------------|-------------|
| SGLang | baseline | 1.0x |
| **Hazy Tokasaurus Megakernel** | — | **> 22% faster** |

#### Event Tensor Compiler（ETC）结果[cite:7]

| 模型 | 指标 | 相对 vLLM |
|------|------|-----------|
| Llama-2/3, Phi-3 | 首 token 延迟（TTFT） | **降低 18-32%** |
| — | 尾 token 延迟 | **降低 27-41%** |
| — | 系统 warmup 时间 | **减少 90% (<50ms)** |
| MoE layer | 吞吐 | **提升 1.23x** |

#### GPUOS 结果[cite:8]

| workload | GPU | speedup vs PyTorch |
|----------|-----|-------------------|
| 微批 element-wise | H100 | **15.3x** |
| 微批 inference | H100 | **8.7x GPU 利用率** |
| Attention pattern | H100 | **显著超过 CUDA Graphs** |
| 通用 element-wise | RTX 5090 | **11.3x** |
| 通用 element-wise | GB10 | **2.8x** |

#### Fleet（AMD MI350, Qwen3-8B）[cite:9]

| 配置 | 指标 | 结果 |
|------|------|------|
| batch=1-8 | decode latency | **降低 1.3-1.5x vs vLLM** |
| batch=32 | L2 hit rate | 12% → **54%**（提升 3.25x） |
| batch=64 | L2 hit rate | 39% → **61%** |
| batch=64 | HBM traffic | **减少 37%** |
| batch=64 | vs chiplet-unaware megakernel | **1.27-1.30x speedup** |

#### MonoMoE（FlashInfer v0.7.0）[cite:10]
- 开源，已集成进 FlashInfer
- 融合 dispatch + activation quantization + grouped GEMM + combine 为单个 persistent kernel
- weight-major 布局消除 expert-local token materialization，减少 padded arithmetic
- symmetric memory 实现 communication-computation overlap

#### Ada-MK（NVIDIA Ada，TensorRT-LLM 混合嵌入）[cite:11]
- 3D shared-memory 约束模型 + K-dimension splitting → **峰值共享内存降低 50%**
- MLIR 离线 DAG 搜索 → 消除 runtime branching
- heterogeneous hybrid engine：prefill 用 Megakernel，decode 用 TensorRT-LLM 高吞吐路径

### 3.2 生产环境使用案例

#### FlashInfer MegaMoE（最成熟工业集成）
FlashInfer v0.7.0 正式将 MegaMoE megakernel 作为官方 API 暴露[cite:10]。MATE（摩尔线程 AI Tensor Engine）也已将 MegaMoE 纳入其算子库，表明该模式正从学术走向**多硬件供应商的标准算子**。

#### DeepSeek DeepGEMM / Mega MoE
DeepGEMM 仓库（2025 年 2 月创建）[cite:12] 实现 Mega MoE 融合 kernel：dispatch + expert GEMM + combine 融合为单个 persistent kernel，通信与 tensor-core 执行重叠，达到 **1000+ TFLOP/s 级别** 的专家并行 GEMM 性能。

#### Meta KernelEvolve
Meta 内部系统，MCTS 搜索 + 多工具 profiling（TritonBench / Torch Profiler / NCU / Proton / MTIA Insight），用于在 **Ascend / NPU 等异构加速器** 上生成 megakernel，用于 MetaKernet Tree Search 优化[cite:13]。

#### DeepSeek DualPipe + SWA
DeepSeek-V4.1 在 vLLM 社区中用 CUDA graphs 实现 SWA bounded replay（一种半 megakernel 做法），**TTFT 降低 30%**，吞吐在 150 TPS 约束下 **5.3x 提升**[cite:14]。

---

## 四、硬件覆盖与收益总结

### 4.1 硬件矩阵

| GPU 架构 | 代表型号 | 研究项目 | Megakernel 特性利用 |
|----------|---------|---------|-------------------|
| **NVIDIA Hopper** | H100, H100 PCIe, H100 80GB | Hazy Megakernel, ETC, Fleet | TMA, async copies, 50MB L2, 3.35 TB/s |
| **NVIDIA Blackwell** | B200, GB10, RTX 5090 | Hazy Megakernel, GPUOS | TMA + WGMMA, 2x die chiplet, FP8 峰值翻倍 |
| **NVIDIA Ampere** | A100, RTX 4090 | AutoKernel 参考 | TF32 tensor cores, async global→shared |
| **NVIDIA Ada** | L20, L2, RTX 4090 | Ada-MK | Ada 架构 small SM → megakernel 缓解 |
| **AMD CDNA3** | MI300X | Fleet (chiplet-aware) | 8 XCD chiplets, private L2, Infinity Fabric |
| **AMD CDNA4** | MI350X, MI355X | Fleet | 5 TB/s+ HBM, chiplet-aware scheduling 核心场景 |
| **AMD RDNA3** | — | HipKittens | ThunderKittens AMD 移植 |
| **Ascend NPU** | Ascend 950DT | KernelEvolve (华为 CANNBot) | AscendC operator generation |
| **Intel Gaudi** | Gaudi 3 | [unverified] | 社区关注中 |

### 4.2 关键收益量化汇总

| 指标 | 最优结果 | 来源 |
|------|---------|------|
| H100 单 GPU Llama-1B 延迟 | **2.5x vs vLLM**, **1.5x vs SGLang** | Hazy 2025 |
| B200 单 GPU Llama-1B 延迟 | **<680 μs / forward pass** | Hazy 2025 |
| 8×H100 Llama-70B 吞吐 | **22%+ vs SGLang** | Hazy 2025 |
| TTFT 降低 | **18-32%** vs vLLM | ETC 2026 |
| MoE layer 吞吐 | **1.23x** | ETC 2026 |
| 微批 element-wise 提速 | **15.3x** vs PyTorch | GPUOS 2026 |
| L2 hit rate 提升 (MI350) | **12% → 54%** | Fleet 2026 |
| warmup 时间减少 | **90% (<50ms)** | ETC 2026 |
| 多节点通信隐藏 | **消除 hidden serialization** | Perseus 2026 |
| int8/int4 量化 + megakernel | **1.12x / 2.42x** 速度提升 | AutoMegaKernel 2026 |

### 4.3 Megakernel vs CUDA Graphs 的本质区别

| 维度 | CUDA Graphs | Megakernel |
|------|-------------|------------|
| launch 边界 | 仍然是 per-kernel launch 的批量回放 | **零 launch 边界**：一个 persistent kernel |
| 同步粒度 | kernel 级全局同步 | **SM 粒度细粒度同步**（counter barrier） |
| 动态 shape | 每 batch size 需独立 graph + fallthrough | Event Tensor 原生支持动态 shape |
| 依赖等待 | 必须等所有 thread block 完成才能开始下一个 kernel | 下一条指令**可在最后一个 tile 完成前开始** |
| weight prefetch | kernel 边界无法 pipeline | **共享内存 paging** 支持跨指令 prefetch |
| 实现复杂度 | 低（自动 capture） | 高（需要指令解释器或编译器） |
| 通用性 | 高（任何 CUDA 代码） | 低（per-model / per-architecture） |

---

## 五、技术发展脉络与趋势

### 5.1 时间线（简化）

```
2009 ─── Aila & Laine (NVIDIA) 首次提出 persistent threads for ray tracing
2010 ─── OptiX 将 persistent kernel 状态机产品化
2012 ─── Gupta et al. 在 unstructured grid solver 验证 persistent kernel
  │
  │  （漫长的科学计算时期，LLM 尚未出现）
  │
2022 ─── PERKS (Pers KernelS) 将 time loop 移入 persistent kernel
2024 ─── ThunderKittens (Hazy Research) — CUDA DSL 基础设施就绪
2025.03 ─── ThunderMLA — FlashMLA megakernel 版本
2025.05 ─── Hazy Look Ma, No Bubbles! — 首个 Llama-1B 全融合 megakernel 里程碑
2025.09 ─── Hazy TP Llama-70B — 8-GPU tensor-parallel megakernel
2025.10 ─── Neptune (Neptune advanced operator fusion)
2025.12 ─── Mirage MPK — 首个自动生成 mega-kernel 的编译器
  │
  │  ── Karpathy autoresearch 发布，agent 驱动线开始 ──
  │
2026.03 ─── AutoKernel (autoresearch → GPU kernel optimization)
2026.03 ─── ThunderKittens 2.0
2026.04 ─── Event Tensor (ETC) — MLSys 2026 论文
2026.04 ─── GPUOS — persistent runtime + NVRTC JIT
2026.04 ─── Fleet — chiplet-aware megakernel
2026.04 ─── RaMP — runtime-aware megakernel polymorphism
2026.05 ─── Ada-MK — MLIR DAG 搜索 + TensorRT-LLM 混合
2026.05 ─── Perseus — 多节点 hidden serialization 消除
2026.06 ─── AutoMegaKernel (AMK) — 静态校验 schedule-IR
2026.06 ─── Kernel Forge (KernelEvolve / META)
2026.09 ─── MegaMoE (FlashInfer v0.7.0) — 工业级集成里程碑
2026.09 ─── MonoMoE (FlashInfer) + Mixture-of-Kittens
2026.09 ─── ForgeMegakernel — 10 里程碑知识库 + coding agent
```

### 5.2 从三条技术线看发展脉络

**脉络 1：手动设计 → 编译器自动生成 → Agent 自动合成**

| 阶段 | 代表 | 瓶颈 | 2026 年状态 |
|------|------|------|------------|
| 手动设计 | Hazy Megakernel, ThunderKittens | 每个新模型需要 expert 级 CUDA 编程，对硬件代际敏感 | 开源参考实现，工业界学习 |
| 编译器自动 | Event Tensor / MPK / GPUOS | 需要定义新抽象（Event Tensor, SM-level graph），编译器基础设施重 | ETC MLSys 2026，GPUOS H100 15.3x |
| Agent 自动 | AutoKernel / ForgeMegakernel / AMK | correctness harness 是关键；探索空间大，结果不稳定 | AutoKernel 论文，ForgeMegakernel 10 里程碑 |

**脉络 2：单 GPU → 多 GPU → 多 chiplet / 多 die**

| 阶段 | 代表 | 核心挑战 |
|------|------|---------|
| 单 GPU | Hazy Llama-1B | SM 间同步、共享内存管理 |
| 多 GPU (TP) | Hazy Llama-70B, MPK | NVLink overlap, 跨 GPU collective 融合 |
| 多 chiplet | Fleet (AMD MI350) | NUMA-like L2 分区，Infinity Fabric traffic |
| 多节点 | Perseus | hidden serialization 消除 |

**脉络 3：静态 workload → 动态 shape / 路由**

| 阶段 | 代表 | 解决的动态性 |
|------|------|------------|
| 纯静态 | 早期 Hazy Llama-1B | 固定 shape / batch size |
| 动态 shape | ETC Event Tensor | 任意 batch size，speculative decoding |
| 动态路由 | MoE megakernels (RaMP, MonoMoE) | token expert routing 随 batch 变 |
| 运行时策略 | RaMP | 根据 batch size + routing 分布动态选 kernel 配置 |

### 5.3 当前趋势判断

#### 趋势 1：**megakernel 正在从 "Research Concept" 成为 "Production Operator"**
FlashInfer 0.7.0 将 MegaMoE 作为官方 API 对外暴露，摩尔线程 MATE 算子库跟进 MegaMoE，说明"融合式 persistent kernel"正成为 LLM 推理引擎的标准实现模式之一——不是取代 vLLM/SGLang，而是作为其内核中**特定模块（MoE decode、微批 element-wise、低延迟 LLM）的可选加速路径**。

#### 趋势 2：**Agent-Driven Synthesis 与 Megakernel 编译器在收敛**
AutoMegaKernel 的 schedule-IR 静态校验 + autoresearch 循环，实际上就是一个**带正确性约束的编译器**。未来的方向很可能是：agent 生成候选 megakernel → 编译器静态校验 + 调度 → 固定 harness 评估 → keep/revert。

#### 趋势 3：**Chiplet GPU 正在推动 megakernel 成为必要而非优化**
AMD CDNA3/CDNA4 和 NVIDIA Blackwell 都是多 chiplet 设计。传统 CUDA/HIP 编程模型假设统一 L2 cache，在 chiplet GPU 上导致 **37% HBM 流量浪费**（Fleet 结果）。Megakernel + chiplet-aware scheduling 不是"锦上添花"，而是**必须**的优化。

#### 趋势 4：**量化 + Megakernel 正在成为新组合**
AutoMegaKernel 自动生成 int8/int4 weight-only GEMV 并 fused dequant 入 megakernel，FlashInfer MonoMoE 同样量化 MoE decode。这解决了传统量化的两个痛点：(a) dequant kernel launch 开销 (b) dequant + GEMM 之间的中间 memory traffic。

#### 趋势 5：**Runtime-Level Persistent Kernel 正在涌现**
GPUOS 是一个很有启发性的方向：它不要求你重写模型，而是 transparent 地把 PyTorch 中的小操作注入 persistent runtime。虽然目前只在 small-op workload 上有 15.3x 效果，但这可能是**PyTorch/Triton 下一个版本的重要扩展方向**。

### 5.4 未解决问题

| 问题 | 现状 |
|------|------|
| 通用跨模型 megakernel 生成 | 仅对特定模型架构（Transformer）有效，Diffusion / MLP-Mixer 等尚未覆盖 |
| 正确性保证 | 静态校验器（AMK 1150 行）覆盖有限，5-stage harness 仍运行时验证 |
| 可移植性 | 手动设计版本硬件敏感，编译器版本硬件适配仍需手动调优 |
| 动态图（torch.compile + data-dependent control flow） | Event Tensor 有初步支持，但完整动态图仍是开放问题 |
| 调试困难 | megakernel 内部复杂，profiling 工具（Nsight Systems, Nsight Compute）支持有限 |
| 与 PyTorch 生态兼容 | GPUOS 透明集成好但性能受限；其他方案多为独立内核 |

---

## Sources

| [cite:N] | 来源 |
|----------|------|
| 1 | karpathy/autoresearch, https://github.com/karpathy/autoresearch (2026) |
| 2 | AutoKernel, arxiv:2603.21331, Section 2.4 |
| 3 | T. Aila & S. Laine, "Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs", HPG 2009 |
| 4 | PERKS, arxiv:2204.02064 |
| 5 | Hazy Research, "Look Ma, No Bubbles!", https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles (2025.05.27) |
| 6 | Hazy Research, "We Bought the Whole GPU", https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main (2025.09.28) |
| 7 | Event Tensor / ETC, MLSys 2026, arxiv:2604.13327 |
| 8 | GPUOS, arxiv:2604.17861 |
| 9 | Fleet, arxiv:2604.15379 |
| 10 | FlashInfer v0.7.0 Release, https://flashinfer.ai/2026/09/22/flashinfer-v07.html; MonoMoE arxiv:2609.04244 |
| 11 | Ada-MK, arxiv:2605.11581 |
| 12 | DeepGEMM, https://github.com/deepseek-ai/DeepGEMM |
| 13 | KernelEvolve (Meta), ISCA 2026, https://gangliao.me/assets/pdf/kernelevolve_isca26.pdf |
| 14 | vLLM Blog, "DeepSeek-V4.1-Flash on vLLM", https://vllm.ai/blog/2026-10-07-deepseek-v41-flash |
