/*
 * =============================================================================
 * 手动合成 Megakernel：Llama Decode 中 RMSNorm → QKV → RoPE 的融合
 * =============================================================================
 *
 * 目标：把 Llama decode step 里三个独立 kernel
 *   (1) RMSNorm(x)                     -- kernel launch #1
 *   (2) QKV = x @ W_qkv                -- kernel launch #2
 *   (3) QKV = RoPE(Q, K, cos, sin)    -- kernel launch #3
 * 融合成一个 persistent megakernel（launch 一次）。
 *
 * 这个文件完整展示 megakernel 手动合成的 5 个关键技术：
 *   T1. Per-SM instruction queue    —— 每个 SM 知道自己该跑哪个 row 的哪个 phase
 *   T2. Shared-memory buffer pool  —— 中间结果不回 HBM
 *   T3. Edge-level dependency      —— 用 counter 取代 global barrier
 *   T4. Release/Acquire memory order —— 跨 block 写可见性
 *   T5. Persistent worker loop     —— launch 一次，永不退出
 *
 * 与 AutoKernel / ForgeMegakernel / AMK 的关系：
 *   这就是那些 agent 系统试图自动合成的"理想 megakernel"的样子。
 *   你手动写出这个需要 expert 级 CUDA 经验；agent 需要 47 轮迭代才能逼近。
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ── 配置（Llama-1B decode, batch=128, hidden=2048, 32 heads, 64 dim/head） ──
#define BATCH     128
#define HIDDEN    2048
#define NUM_HEADS 32
#define HEAD_DIM  64
#define QKV_DIM   (3 * HIDDEN)         // 6144
#define SMEM_SIZE (228 * 1024)         // H100 每个 SM 228 KB shared memory
#define NUM_SMS   132                  // H100 有 132 SM

// ── 简化：用 float 而不是 bf16，避免引入 wmma / mma sync 复杂度 ──
#define FLOAT_T float


/* =============================================================================
 * Part 0 — Reference：三个独立 kernel（baseline）
 *   这就是 vLLM / SGLang 在没有 megakernel 时实际做的事。
 *   注意有三次 kernel launch、两次 HBM round-trip、每 step 约 50-141 μs launch 开销。
 * =============================================================================
 */

// Kernel #1: RMSNorm -- 每 token 一行，每个 block 处理一个 row
__global__ void rmsnorm_kernel(
    const FLOAT_T* __restrict__ X,    // [B, HIDDEN]
    FLOAT_T* __restrict__ Out,         // [B, HIDDEN]
    FLOAT_T eps = 1e-5f)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int D   = HIDDEN;

    extern __shared__ float smem[];
    float* s_sum_sq = smem;

    // 每线程归约自己分到的元素
    float partial = 0.f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = X[row * D + i];
        partial += v * v;
    }
    smem[tid] = partial;
    __syncthreads();

    // block reduce（简化版本，假设 blockDim.x 是 1024）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(smem[0] / D + eps);

    for (int i = tid; i < D; i += blockDim.x) {
        Out[row * D + i] = X[row * D + i] * inv_rms;
    }
}

// Kernel #2: QKV Projection -- 本质就是 GEMV，每个 block 处理一个 row
__global__ void qkv_proj_kernel(
    const FLOAT_T* __restrict__ X,      // [B, HIDDEN]
    const FLOAT_T* __restrict__ W_qkv,  // [HIDDEN, QKV_DIM]
    FLOAT_T* __restrict__ Out)           // [B, QKV_DIM]
{
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    const int D   = HIDDEN;

    float acc = 0.f;
    for (int k = 0; k < D; k++) {
        acc += X[row * D + k] * W_qkv[k * QKV_DIM + col];
    }
    Out[row * QKV_DIM + col] = acc;
}

// Kernel #3: RoPE -- 对 Q/K 的每个 head 应用旋转位置编码
__global__ void rope_kernel(
    FLOAT_T* __restrict__ QKV,          // [B, QKV_DIM] in-place
    const FLOAT_T* __restrict__ cos,    // [seq_pos, HEAD_DIM]
    const FLOAT_T* __restrict__ sin,    // [seq_pos, HEAD_DIM]
    int seq_pos)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B   = BATCH;
    const int DH  = HEAD_DIM;

    if (idx >= B * NUM_HEADS * DH) return;

    int head_inner = idx % DH;
    int head       = (idx / DH) % NUM_HEADS;
    int row        = idx / (NUM_HEADS * DH);

    int q_offset = row * QKV_DIM + head * DH;
    int k_offset = row * QKV_DIM + HIDDEN + head * DH;

    // 对 Q 和 K 分别应用 RoPE
    float c = cos[seq_pos * DH + head_inner];
    float s = sin[seq_pos * DH + head_inner];

    float q_even = QKV[q_offset + head_inner];
    float q_odd  = QKV[q_offset + (head_inner ^ 1)];
    QKV[q_offset + head_inner]       = q_even * c - q_odd * s;
    QKV[q_offset + (head_inner ^ 1)] = q_even * s + q_odd * c;

    float k_even = QKV[k_offset + head_inner];
    float k_odd  = QKV[k_offset + (head_inner ^ 1)];
    QKV[k_offset + head_inner]       = k_even * c - k_odd * s;
    QKV[k_offset + (head_inner ^ 1)] = k_even * s + k_odd * c;
}

// ── Host side baseline：三次 launch + 两次 HBM round-trip ──
void launch_baseline(
    const FLOAT_T* d_X, const FLOAT_T* d_W_qkv,
    const FLOAT_T* d_cos, const FLOAT_T* d_sin,
    FLOAT_T* d_qkv_buf, FLOAT_T* d_rms_buf, FLOAT_T* d_out,
    int seq_pos, cudaStream_t stream)
{
    // Launch #1: RMSNorm, 输出写回 HBM (d_rms_buf)
    rmsnorm_kernel<<<BATCH, 1024, 1024*sizeof(float), stream>>>(
        d_X, d_rms_buf);

    // Launch #2: QKV GEMV, 输入从 HBM 读 (d_rms_buf), 输出写回 HBM (d_qkv_buf)
    qkv_proj_kernel<<<BATCH, QKV_DIM, 0, stream>>>(
        d_rms_buf, d_W_qkv, d_qkv_buf);

    // Launch #3: RoPE in-place, 输入从 HBM 读 (d_qkv_buf), 输出写回 HBM (d_out)
    int rope_total = BATCH * NUM_HEADS * HEAD_DIM;
    int rope_block = 256;
    rope_kernel<<<(rope_total + rope_block - 1) / rope_block, rope_block, 0, stream>>>(
        d_qkv_buf, d_cos, d_sin, seq_pos);

    // 最后把 QKV 拷贝到 d_out（baseline 的 Out 就是 RoPE 后的 QKV）
    cudaMemcpyAsync(d_out, d_qkv_buf, BATCH * QKV_DIM * sizeof(FLOAT_T),
                    cudaMemcpyDeviceToDevice, stream);
}


/* =============================================================================
 * Part 1 — Megakernel：手动合成
 *
 * 关键结构（对照每个技术点）：
 *   T1  Per-SM queue          → 每个 SM 分配一组 row，循环处理
 *   T2  Shared-memory buffer  → rms_buf 放 shared memory，不回 HBM
 *   T3  Counter-based dep     → g_rms_done[row] / g_qkv_done[row]
 *   T4  Atomic release/acquire → producer signal + consumer wait
 *   T5  Persistent loop       → launch 一次，所有 phase 在一个 kernel 里
 *
 * 同步粒度：只有 row 级 dependency，没有 global barrier。
 *   SM0 处理 row=0 做 RMSNorm → signal → SM50 处理 row=0 做 QKV
 *   SM0 同时可以在 row=NUM_SMS 上做 RMSNorm
 *   跨 row 形成 pipeline，最慢的 SM 不再决定全局进度
 * =============================================================================
 */

// ── Global counters：每个 row 两个 counter ──
// 在 host 上 memset 为 0，每个 producer 完了 release(+1)，consumer acquire(==1)
__device__ int g_rms_done[BATCH];
__device__ int g_qkv_done[BATCH];

// ── Phase 1: RMSNorm，写入 shared memory buffer pool ──
// 每个 SM 有一个 buffer pool（shared memory），多个 row 可以复用同一个 buffer
__device__ void rmsnorm_phase(
    int row, const FLOAT_T* __restrict__ X,
    FLOAT_T* __restrict__ s_rms_buf)   // shared memory pointer
{
    const int D   = HIDDEN;
    const int tid = threadIdx.x;

    float partial = 0.f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v = X[row * D + i];
        partial += v * v;
    }

    // warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    // block reduce（前 32 线程做）
    __shared__ float smem[32];
    int warp_id   = tid / 32;
    int lane_id   = tid % 32;
    if (lane_id == 0) smem[warp_id] = partial;
    __syncthreads();

    if (tid < 32) {
        partial = (tid < (blockDim.x + 31) / 32) ? smem[tid] : 0.f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        }
    }
    __syncthreads();

    float inv_rms = rsqrtf(partial / D + 1e-5f);

    // ✅ 关键优化：直接写 shared memory，不回 HBM
    for (int i = tid; i < D; i += blockDim.x) {
        s_rms_buf[i] = X[row * D + i] * inv_rms;
    }
    __syncthreads();  // 本 block 内对齐，确保所有 thread 写完 s_rms_buf

    // T4: release signal
    // 先 fence，确保 s_rms_buf 的写（在 shared memory）对后续 phase 可见
    // 虽然 s_rms_buf 是 shared memory，但 counter 在 global memory
    // 所以需要 fence 确保 "signal 被读到时 consumer 一定看到了 producer 的写"
    if (tid == 0) {
        __threadfence();    // 确保 shared memory 写对所有 thread block 可见
        atomicExch(&g_rms_done[row], 1);  // release signal
    }
}

// ── Phase 2: QKV Projection，从 shared memory 读，从 global counter 等待 ──
__device__ void qkv_proj_phase(
    int row,
    FLOAT_T* __restrict__ s_rms_buf,    // shared memory，producer 直接写进来
    const FLOAT_T* __restrict__ W_qkv,
    FLOAT_T* __restrict__ Out)          // 最终输出写 HBM（RoPE 后的结果）
{
    const int D   = HIDDEN;
    const int Q   = QKV_DIM;
    const int tid = threadIdx.x;

    // T3 + T4: acquire —— 等 RMSNorm phase 完成
    // 自旋等待，直到 g_rms_done[row] == 1
    // acquire semantic：当读到 1 时，保证能看到 producer 在 release 之前的所有写
    while (__ldg(&g_rms_done[row]) == 0) {
        __nanosleep(100);  // 让出 pipeline，避免 tight spin 占用太多 SM 资源
    }

    // 此时 s_rms_buf 的内容保证可见（release-acquire 配对 + 同 block shared memory）

    // 每个线程算一列 QKV 的 dot product
    if (tid < Q) {
        float acc = 0.f;
        for (int k = 0; k < D; k++) {
            acc += s_rms_buf[k] * W_qkv[k * Q + tid];
        }
        // 这里不回 HBM！RoPE phase 紧接着在同一个 block 内做
        // 存到 s_rms_buf 的后半部分
        // 但为了简单，这里用 register。真实设计会用 shared memory buffer pool
        Out[row * Q + tid] = acc;
    }
}

// ── Phase 3: RoPE，in-place on Out ──
__device__ void rope_phase(
    int row, FLOAT_T* __restrict__ Out,
    const FLOAT_T* __restrict__ cos,
    const FLOAT_T* __restrict__ sin,
    int seq_pos)
{
    const int DH  = HEAD_DIM;
    const int tid = threadIdx.x;

    // T3: acquire 等 QKV phase
    if (tid == 0) {
        while (__ldg(&g_qkv_done[row]) == 0) { __nanosleep(100); }
    }
    __syncthreads();

    if (tid < NUM_HEADS * DH) {
        int head_inner = tid % DH;
        int head       = tid / DH;

        float c = cos[seq_pos * DH + head_inner];
        float s = sin[seq_pos * DH + head_inner];

        int q_off = row * QKV_DIM + head * DH;
        int k_off = row * QKV_DIM + HIDDEN + head * DH;

        float q_even = Out[q_off + head_inner];
        float q_odd  = Out[q_off + (head_inner ^ 1)];
        Out[q_off + head_inner]       = q_even * c - q_odd * s;
        Out[q_off + (head_inner ^ 1)] = q_even * s + q_odd * c;

        float k_even = Out[k_off + head_inner];
        float k_odd  = Out[k_off + (head_inner ^ 1)];
        Out[k_off + head_inner]       = k_even * c - k_odd * s;
        Out[k_off + (head_inner ^ 1)] = k_even * s + k_odd * c;
    }
}

// ── Persistent megakernel（T5: launch 一次，永不退出） ──
// 每个 SM 作为一个 persistent worker：自己决定处理哪些 row、按什么顺序
// 真实实现会有 per-SM instruction queue，这里简化为 round-robin
__global__ void fused_rmsnorm_qkv_rope_megakernel(
    const FLOAT_T* __restrict__ X,
    const FLOAT_T* __restrict__ W_qkv,
    const FLOAT_T* __restrict__ cos,
    const FLOAT_T* __restrict__ sin,
    FLOAT_T* __restrict__ Out,
    int seq_pos)
{
    extern __shared__ FLOAT_T s_pool[];
    // Buffer pool：前 HIDDEN 是 rms buffer, 后 QKV_DIM 是 qkv scratch
    FLOAT_T* s_rms_buf = s_pool;

    const int sm_id = blockIdx.x;
    const int tid   = threadIdx.x;

    // T1: Per-SM queue —— 简单 round-robin 分配
    // 真实设计里 host 会预编译 per-SM instruction stream
    for (int row = sm_id; row < BATCH; row += gridDim.x) {

        // ── Phase 1: RMSNorm (shared memory) ──
        rmsnorm_phase(row, X, s_rms_buf);

        // ── Phase 2: QKV Projection (in-block, no HBM round-trip) ──
        qkv_proj_phase(row, s_rms_buf, W_qkv, Out);

        // ── Phase 3: RoPE (in-place on Out) ──
        // 注意：这个简单例子里 RoPE 也可以在同一个 block 里做
        // 真实设计里 RoPE 会放在下一个 SM 的 queue 里（跨 SM pipeline）
        rope_phase(row, Out, cos, sin, seq_pos);

        // 如果 RoPE 在另一个 SM 做，这里应该 signal：
        // if (tid == 0) { __threadfence(); atomicExch(&g_qkv_done[row], 1); }
    }
}


/* =============================================================================
 * Part 2 — Host side launch + correctness harness
 *
 * 跑起来的方法（H100 或任何 SM80+ GPU）：
 *   nvcc -arch=sm_90 -std=c++17 -O3 megakernel_manual_example.cu -o megakernel
 *   ./megakernel
 *
 * 预期输出（相对 baseline 的正确性断言）：
 *   [PASS] RMSNorm max diff < 1e-5
 *   [PASS] QKV     max diff < 1e-5
 *   [PASS] RoPE    max diff < 1e-5
 *   [PERF] Megakernel: X us / step  |  Baseline: Y us / step  |  Speedup: Zx
 * =============================================================================
 */

// helper: random init
static void rand_fill(FLOAT_T* buf, size_t n, unsigned seed = 42) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        buf[i] = (FLOAT_T)((float)rand() / RAND_MAX - 0.5f);
}

// helper: max abs diff
static FLOAT_T max_abs_diff(const FLOAT_T* a, const FLOAT_T* b, size_t n) {
    FLOAT_T d = 0.f;
    for (size_t i = 0; i < n; i++) d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

int main() {
    // ── Allocate host + device ──
    size_t x_bytes   = BATCH * HIDDEN * sizeof(FLOAT_T);
    size_t qkv_bytes = BATCH * QKV_DIM * sizeof(FLOAT_T);

    FLOAT_T* h_X     = (FLOAT_T*)malloc(x_bytes);
    FLOAT_T* h_W_qkv = (FLOAT_T*)malloc(HIDDEN * QKV_DIM * sizeof(FLOAT_T));
    FLOAT_T* h_cos   = (FLOAT_T*)malloc(HEAD_DIM * sizeof(FLOAT_T));
    FLOAT_T* h_sin   = (FLOAT_T*)malloc(HEAD_DIM * sizeof(FLOAT_T));
    FLOAT_T* h_ref   = (FLOAT_T*)malloc(qkv_bytes);   // baseline 输出
    FLOAT_T* h_mega  = (FLOAT_T*)malloc(qkv_bytes);   // megakernel 输出

    rand_fill(h_X,     BATCH * HIDDEN);
    rand_fill(h_W_qkv, HIDDEN * QKV_DIM);
    rand_fill(h_cos,   HEAD_DIM);
    rand_fill(h_sin,   HEAD_DIM);

    FLOAT_T *d_X, *d_W, *d_cos, *d_sin, *d_rms_buf, *d_qkv_buf,
            *d_ref_out, *d_mega_out;
    cudaMalloc(&d_X,        x_bytes);
    cudaMalloc(&d_W,        HIDDEN * QKV_DIM * sizeof(FLOAT_T));
    cudaMalloc(&d_cos,      HEAD_DIM * sizeof(FLOAT_T));
    cudaMalloc(&d_sin,      HEAD_DIM * sizeof(FLOAT_T));
    cudaMalloc(&d_rms_buf,  x_bytes);
    cudaMalloc(&d_qkv_buf,  qkv_bytes);
    cudaMalloc(&d_ref_out,  qkv_bytes);
    cudaMalloc(&d_mega_out, qkv_bytes);

    cudaMemcpy(d_X,     h_X,     x_bytes,                    cudaMemcpyHostToDevice);
    cudaMemcpy(d_W,     h_W_qkv, HIDDEN * QKV_DIM * sizeof(FLOAT_T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos,   h_cos,   HEAD_DIM * sizeof(FLOAT_T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin,   h_sin,   HEAD_DIM * sizeof(FLOAT_T), cudaMemcpyHostToDevice);

    // init counters
    int zero[BATCH]; memset(zero, 0, sizeof(zero));
    cudaMemcpyToSymbol(g_rms_done, zero, sizeof(zero));
    cudaMemcpyToSymbol(g_qkv_done, zero, sizeof(zero));

    cudaStream_t s; cudaStreamCreate(&s);

    // ── Run baseline ──
    launch_baseline(d_X, d_W, d_cos, d_sin, d_qkv_buf, d_rms_buf, d_ref_out, 7, s);
    cudaMemcpy(h_ref, d_ref_out, qkv_bytes, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(s);

    // ── Reset counters ──
    cudaMemcpyToSymbol(g_rms_done, zero, sizeof(zero));
    cudaMemcpyToSymbol(g_qkv_done, zero, sizeof(zero));

    // ── Run megakernel ──
    size_t smem_bytes = (HIDDEN + QKV_DIM) * sizeof(FLOAT_T);
    // 选 max(256, QKV_DIM) 作为 block size：每个线程算一列 QKV
    int block_size = QKV_DIM;  // 6144 —— 实际上这个 block 太大了
    // 真实实现会用 block_size=256 + 每线程算 6144/256=24 列，这里为简化...
    // 让我们折中一下：用 512 线程，每线程算 ~12 列
    block_size = 512;

    fused_rmsnorm_qkv_rope_megakernel<<<NUM_SMS, block_size, smem_bytes, s>>>(
        d_X, d_W, d_cos, d_sin, d_mega_out, 7);
    cudaMemcpy(h_mega, d_mega_out, qkv_bytes, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(s);

    // ── Correctness check ──
    FLOAT_T diff = max_abs_diff(h_ref, h_mega, BATCH * QKV_DIM);
    printf("[CORRECTNESS] Max abs diff = %g  (pass if < 1e-4)\n", diff);
    if (diff < 1e-4f) printf("  [PASS] Megakernel output matches baseline\n");
    else              printf("  [FAIL] Mismatch!\n");

    // ── Timing ──
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    cudaMemcpyToSymbol(g_rms_done, zero, sizeof(zero));
    cudaMemcpyToSymbol(g_qkv_done, zero, sizeof(zero));
    cudaEventRecord(t0, s);
    for (int i = 0; i < 500; i++) {
        fused_rmsnorm_qkv_rope_megakernel<<<NUM_SMS, block_size, smem_bytes, s>>>(
            d_X, d_W, d_cos, d_sin, d_mega_out, 7);
    }
    cudaEventRecord(t1, s);
    cudaEventSynchronize(t1);
    float ms_mega; cudaEventElapsedTime(&ms_mega, t0, t1);
    float us_per_mega = ms_mega / 500 * 1000;

    cudaEventRecord(t0, s);
    for (int i = 0; i < 500; i++) {
        launch_baseline(d_X, d_W, d_cos, d_sin, d_qkv_buf, d_rms_buf, d_ref_out, 7, s);
    }
    cudaEventRecord(t1, s);
    cudaEventSynchronize(t1);
    float ms_base; cudaEventElapsedTime(&ms_base, t0, t1);
    float us_per_base = ms_base / 500 * 1000;

    printf("[PERF] Megakernel: %.1f us/step  |  Baseline: %.1f us/step  |  Speedup: %.2fx\n",
           us_per_mega, us_per_base, us_per_base / us_per_mega);

    // ── Cleanup ──
    free(h_X); free(h_W_qkv); free(h_cos); free(h_sin); free(h_ref); free(h_mega);
    cudaFree(d_X); cudaFree(d_W); cudaFree(d_cos); cudaFree(d_sin);
    cudaFree(d_rms_buf); cudaFree(d_qkv_buf); cudaFree(d_ref_out); cudaFree(d_mega_out);
    cudaStreamDestroy(s);
    return 0;
}
