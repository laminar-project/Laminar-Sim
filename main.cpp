#include <immintrin.h>
#include <x86intrin.h>
#include <cpuid.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>

static volatile uint64_t g_sink = 0;

/* ------------ pin to CPU0 ------------ */
static void pin_thread_to_cpu0() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

/* ------------ serialized TSC ------------ */
static inline __attribute__((always_inline)) void serialize_cpu() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);
}

static inline __attribute__((always_inline)) uint64_t tsc_start() {
    serialize_cpu();
    return __rdtsc();
}

static inline __attribute__((always_inline)) uint64_t tsc_stop() {
    unsigned int aux;
    uint64_t t = __rdtscp(&aux);
    serialize_cpu();
    return t;
}

/* ------------ monotonic clock (for estimating TSC Hz) ------------ */
static inline uint64_t qpc_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline uint64_t qpc_freq() {
    return 1000000000ull;
}

static double estimate_tsc_hz(double seconds = 0.25) {
    uint64_t fq = qpc_freq();
    uint64_t wait_ticks = (uint64_t)(seconds * (double)fq);

    uint64_t q0 = qpc_now();
    uint64_t c0 = tsc_start();
    while ((qpc_now() - q0) < wait_ticks) _mm_pause();
    uint64_t c1 = tsc_stop();
    uint64_t q1 = qpc_now();

    double dt = (double)(q1 - q0) / (double)fq;
    double dc = (double)(c1 - c0);
    return dc / dt;
}

/* ------------ stats ------------ */
static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)std::llround(p * (v.size() - 1));
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

struct BenchResult {
    double avg_cycles = 0, p50_cycles = 0, p99_cycles = 0;
    double avg_ns = 0, p50_ns = 0, p99_ns = 0;
};

template <typename Fn>
static BenchResult bench_blocks(
    const char* name, Fn&& fn,
    double tsc_hz,
    int warmup_blocks, int measure_blocks,
    uint64_t iters_per_block,
    uint64_t ops_per_iter
) {
    for (int b = 0; b < warmup_blocks; ++b) fn(iters_per_block);

    std::vector<double> cyc;
    cyc.reserve(measure_blocks);

    uint64_t total_cycles = 0, total_ops = 0;
    for (int b = 0; b < measure_blocks; ++b) {
        uint64_t c0 = tsc_start();
        fn(iters_per_block);
        uint64_t c1 = tsc_stop();
        uint64_t dc = c1 - c0;
        uint64_t ops = iters_per_block * ops_per_iter;
        total_cycles += dc;
        total_ops += ops;
        cyc.push_back((double)dc / (double)ops);
    }

    BenchResult r;
    r.avg_cycles = (double)total_cycles / (double)total_ops;
    auto tmp = cyc; r.p50_cycles = percentile(tmp, 0.50);
    tmp = cyc;      r.p99_cycles = percentile(tmp, 0.99);

    auto to_ns = [&](double c) { return c * (1e9 / tsc_hz); };
    r.avg_ns = to_ns(r.avg_cycles);
    r.p50_ns = to_ns(r.p50_cycles);
    r.p99_ns = to_ns(r.p99_cycles);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << name << "\n";
    std::cout << "  cycles/op  avg " << r.avg_cycles
              << ", p50 " << r.p50_cycles
              << ", p99 " << r.p99_cycles << "\n";
    std::cout << "  ns/op      avg " << r.avg_ns
              << ", p50 " << r.p50_ns
              << ", p99 " << r.p99_ns << "\n";
    return r;
}

/* ============================================================
   1) Pure-instruction style (throughput-ish, for sanity check)
   ============================================================ */
static void pure_andnot(uint64_t iters) {
    __m256i a = _mm256_set_epi64x(
        0x0123456789ABCDEFull, 0x0FEDCBA987654321ull,
        0x1111111111111111ull, 0x2222222222222222ull);
    __m256i b = _mm256_set_epi64x(
        0x13579BDF2468ACE0ull, 0x02468ACE13579BDFull,
        0x3333333333333333ull, 0x4444444444444444ull);
    __m256i acc = _mm256_setzero_si256();
    const __m256i k = _mm256_set1_epi64x(0x9E3779B97F4A7C15ull);

    for (uint64_t i = 0; i < iters; ++i) {
        a = _mm256_add_epi64(a, k);
        b = _mm256_xor_si256(b, a);
        __m256i c = _mm256_andnot_si256(a, b);
        acc = _mm256_add_epi64(acc, c);
    }

    alignas(32) uint64_t out[4];
    _mm256_store_si256((__m256i*)out, acc);
    g_sink ^= (out[0] + out[1] + out[2] + out[3]);
}

/* fast ln (bsr + LUT) */
static constexpr int LUT_BITS = 8;
static constexpr int LUT_SIZE = 1 << LUT_BITS;
static float g_log_lut[LUT_SIZE];

static void init_log_lut() {
    for (int i = 0; i < LUT_SIZE; ++i) {
        float x = 1.0f + (float)i / (float)LUT_SIZE;
        g_log_lut[i] = std::log(x);
    }
}

static inline __attribute__((always_inline)) float fast_ln_u32(uint32_t x) {
    if (x == 0) return -INFINITY;
    uint32_t e = 31u - (uint32_t)__builtin_clz(x);
    uint32_t shift = (e > LUT_BITS) ? (e - LUT_BITS) : 0;
    uint32_t mant = x >> shift;
    uint32_t lut_idx = (mant & (LUT_SIZE - 1));
    const float ln2 = 0.6931471805599453f;
    return (float)e * ln2 + g_log_lut[lut_idx];
}

static void pure_fastlog(uint64_t iters) {
    uint32_t x = 0x12345678u;
    float acc = 0.0f;
    for (uint64_t i = 0; i < iters; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        acc += fast_ln_u32(x | 1u);
    }
    g_sink ^= (uint64_t)acc;
}

/* ============================================================
   2) Micro-kernels (mapped to Laminar parameters)
   - mask: single-node bitmap update primitive
   - fastlog: single-node DA score primitive
   - gamma_h3: single-zone gamma*H^3 update primitive
   ============================================================ */

static constexpr size_t WORKSET_BYTES = 1u << 20;
static constexpr size_t VEC_BYTES = 32;
static constexpr size_t VEC_COUNT = WORKSET_BYTES / VEC_BYTES;

alignas(32) static __m256i g_A[VEC_COUNT];
alignas(32) static __m256i g_B[VEC_COUNT];
alignas(32) static __m256i g_C[VEC_COUNT];

static void init_masks() {
    for (size_t i = 0; i < VEC_COUNT; ++i) {
        uint64_t x = (uint64_t)i * 0x9E3779B97F4A7C15ull + 0xD1B54A32D192ED03ull;
        g_A[i] = _mm256_set_epi64x(
            (long long)(x ^ 0x1111), (long long)(x ^ 0x2222),
            (long long)(x ^ 0x3333), (long long)(x ^ 0x4444));
        g_B[i] = _mm256_set_epi64x(
            (long long)(x ^ 0xAAAA), (long long)(x ^ 0xBBBB),
            (long long)(x ^ 0xCCCC), (long long)(x ^ 0xDDDD));
        g_C[i] = _mm256_setzero_si256();
    }
}

/* one op = one node-mask load + andnot + store */
static void kernel_mask_andnot(uint64_t iters) {
    size_t idx = (size_t)(g_sink % VEC_COUNT);
    __m256i dep = _mm256_set1_epi64x((long long)g_sink);

    for (uint64_t t = 0; t < iters; ++t) {
        __m256i a = _mm256_load_si256(&g_A[idx]);
        __m256i b = _mm256_load_si256(&g_B[idx]);
        a = _mm256_xor_si256(a, dep);
        __m256i c = _mm256_andnot_si256(a, b);
        _mm256_store_si256(&g_C[idx], c);

        alignas(32) uint64_t tmp[4];
        _mm256_store_si256((__m256i*)tmp, c);
        idx = (idx + (tmp[0] & (VEC_COUNT - 1))) & (VEC_COUNT - 1);
        dep = _mm256_xor_si256(dep, c);
    }

    alignas(32) uint64_t out[4];
    _mm256_store_si256((__m256i*)out, dep);
    g_sink ^= (out[0] ^ out[1] ^ out[2] ^ out[3]);
}

/* one op = one node score evaluation */
static constexpr size_t INT_COUNT = 1u << 18;
alignas(64) static uint32_t g_X[INT_COUNT];
alignas(64) static float g_Y[INT_COUNT];

static void init_ints() {
    uint32_t x = 0x12345678u;
    for (size_t i = 0; i < INT_COUNT; ++i) {
        x = x * 1664525u + 1013904223u;
        g_X[i] = x | 1u;
        g_Y[i] = 0.0f;
    }
}

static inline uint32_t rotl32(uint32_t x, unsigned r) {
    return (x << r) | (x >> (32 - r));
}

static void kernel_fastlog(uint64_t iters) {
    size_t idx = (size_t)(g_sink % INT_COUNT);
    uint32_t dep = (uint32_t)g_sink;

    for (uint64_t t = 0; t < iters; ++t) {
        uint32_t x = g_X[idx] ^ dep;
        float y = fast_ln_u32(x | 1u);
        g_Y[idx] = y;

        uint32_t bits = 0;
        memcpy(&bits, &y, sizeof(bits));
        bits = rotl32(bits, 13);

        idx = (idx + (bits & (INT_COUNT - 1))) & (INT_COUNT - 1);
        dep ^= bits + 0x9E3779B9u;
    }

    g_sink ^= (uint64_t)dep;
}

/* one op = one zone gamma*H^3 update */
static constexpr int Z = 256;
alignas(32) static float g_H[Z];
alignas(32) static float g_S[Z];
alignas(32) static float g_O[Z];

static void init_tensor() {
    for (int i = 0; i < Z; ++i) {
        g_H[i] = 0.001f * (float)(i + 1);
        g_S[i] = 0.002f * (float)(Z - i);
        g_O[i] = 0.0f;
    }
}

static void gamma_h3_single_zone_update() {
    const __m256 gamma = _mm256_set1_ps(1.2345f);
    const __m256 beta  = _mm256_set1_ps(0.9876f);
    const __m256 eps   = _mm256_set1_ps(1e-6f);

    for (int i = 0; i < Z; i += 8) {
        __m256 h = _mm256_load_ps(&g_H[i]);
        __m256 s = _mm256_load_ps(&g_S[i]);
        __m256 h2 = _mm256_mul_ps(h, h);
        __m256 h3 = _mm256_mul_ps(h2, h);
        __m256 term = _mm256_add_ps(_mm256_mul_ps(gamma, h3), _mm256_mul_ps(beta, s));
        __m256 o = _mm256_load_ps(&g_O[i]);
        o = _mm256_add_ps(o, term);
        _mm256_store_ps(&g_O[i], o);

        h = _mm256_add_ps(h, eps);
        _mm256_store_ps(&g_H[i], h);
    }
}

static void kernel_gamma_h3_update(uint64_t iters) {
    for (uint64_t t = 0; t < iters; ++t) {
        gamma_h3_single_zone_update();
    }
    float sum = 0.0f;
    for (int i = 0; i < Z; ++i) sum += g_O[i];
    g_sink ^= (uint64_t)sum;
}

/* ============================================================
   main
   ============================================================ */
int main() {
    pin_thread_to_cpu0();
    init_log_lut();
    init_masks();
    init_ints();
    init_tensor();

    std::cout << "Clock freq = " << qpc_freq() << " ticks/sec\n";
    double tsc_hz = estimate_tsc_hz(0.25);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Estimated TSC = " << (tsc_hz / 1e9) << " GHz\n\n";

    const int warm = 8;
    const int blocks = 60;

    std::cout << "=== A) Pure instruction style (throughput-ish, sanity check) ===\n";
    bench_blocks("Pure AVX2 andnot",
                 pure_andnot, tsc_hz,
                 warm, blocks,
                 8000000, 1);

    bench_blocks("Pure fast ln (bsr+LUT)",
                 pure_fastlog, tsc_hz,
                 warm, blocks,
                 8000000, 1);

    std::cout << "\n=== B) Micro-kernels (mapped to Laminar parameters) ===\n";
    bench_blocks("Mask micro-kernel: single-node mask update",
                 kernel_mask_andnot, tsc_hz,
                 warm, blocks,
                 2000000, 1);

    bench_blocks("Fastlog micro-kernel: single-node score",
                 kernel_fastlog, tsc_hz,
                 warm, blocks,
                 2000000, 1);

    bench_blocks("Gamma*H^3 micro-kernel: single-zone update",
                 kernel_gamma_h3_update,
                 tsc_hz, warm, blocks,
                 200000, 1);

    std::cout << "\nsink = " << g_sink << "\n";
    return 0;
}