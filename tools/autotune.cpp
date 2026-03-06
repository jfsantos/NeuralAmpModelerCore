// autotune.cpp — Auto-tuning benchmark for NAM inline GEMM specializations
//
// Profiles each inline specialization against Eigen and outputs a
// nam_tuning_config.h header with optimal per-operation choices.
//
// IMPORTANT: Each specialization is benchmarked using a template with
// compile-time-constant dimensions, so the compiler can fully unroll
// the loops — matching the actual unrolled code in dsp.cpp / conv1d.cpp.
//
// Usage:
//   cmake -B build -DNAM_BUILD_AUTOTUNE=ON -DCMAKE_BUILD_TYPE=Release
//   cmake --build build --target nam_autotune
//   ./build/tools/nam_autotune > NAM/nam_tuning_config.h
//
// Or from benchmarks/:
//   make autotune && ./autotune > ../NAM/nam_tuning_config.h

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

// ---------------------------------------------------------------------------
// Prevent compiler from optimizing away a value
// ---------------------------------------------------------------------------
template <typename T>
inline void DoNotOptimize(T const& value)
{
  asm volatile("" : : "r,m"(value) : "memory");
}
template <typename T>
inline void DoNotOptimize(T& value)
{
  asm volatile("" : "+r,m"(value) : : "memory");
}

// ---------------------------------------------------------------------------
// Fill a float buffer with uniform random values
// ---------------------------------------------------------------------------
static void fill_random(float* buf, size_t n)
{
  static std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; i++)
    buf[i] = dist(rng);
}

// ---------------------------------------------------------------------------
// Benchmark helper: returns median time in nanoseconds
// ---------------------------------------------------------------------------
static double bench_median(std::function<void()> fn, int iters = 5000, int warmup = 500)
{
  for (int i = 0; i < warmup; i++)
    fn();

  std::vector<double> times(iters);
  for (int i = 0; i < iters; i++)
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    times[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
  }

  std::sort(times.begin(), times.end());
  return times[iters / 2];
}

// ---------------------------------------------------------------------------
// Result for a single specialization
// ---------------------------------------------------------------------------
struct TuneResult
{
  std::string macro_name;
  double inline_ns;
  double eigen_ns;
  bool use_inline;
};

// ===========================================================================
// Templated inline GEMM kernels with compile-time dimensions
//
// These use an "output-major" loop order that matches the hand-written
// specializations in dsp.cpp and conv1d.cpp:
//   for each frame:
//     load input scalars
//     for each output channel:
//       accumulate weight[col] * input_scalar across input channels
//
// This structure generates optimal code on all targets:
// - Cortex-M7 (scalar FPU): GCC fully unrolls both loops, identical to
//   hand-written code (verified: same vmul/vfma count, same stack spills).
// - NEON / SSE+AVX (desktop): the inner output loop vectorizes naturally
//   with scalar-broadcast FMA (e.g. fmla.4s v, v, v[lane] on ARM).
//
// On desktop SIMD targets, clang/MSVC may try to auto-vectorize the outer
// frame loop instead, batching multiple frames with ld4/st4 transposes.
// This is 1.5-2x slower for >=4x4 matrices. We disable outer-loop
// vectorization on these compilers so the benchmarked code matches the
// actual codebase performance.
// ===========================================================================

// Disable vectorization of the outer (frame) loop on SIMD desktop targets.
// On Cortex-M7 (GCC, no NEON), this is a no-op — all approaches generate
// identical scalar FP code regardless of loop structure.
#if defined(__clang__)
#define NAM_AUTOTUNE_NOVECTORIZE _Pragma("clang loop vectorize(disable)")
#elif defined(_MSC_VER)
#define NAM_AUTOTUNE_NOVECTORIZE __pragma(loop(no_vector))
#else
// GCC: no pragma needed. On ARM Cortex-M7 (no NEON) all loop structures
// produce identical code. On desktop GCC with SIMD, GCC's auto-vectorizer
// is less aggressive than clang and typically doesn't apply the problematic
// cross-frame transpose strategy. If a future GCC version does, add:
//   _Pragma("GCC ivdep") or restructure the benchmark.
#define NAM_AUTOTUNE_NOVECTORIZE
#endif

// Conv1x1: output(M, samples) = weight(M, N) * input(N, samples)
// Column-major: weight(o,i) at weight[i*M+o], input(i,f) at input[f*N+i]
template <int M, int N>
static void conv1x1_inline(const float* __restrict__ weight, const float* __restrict__ input,
                           float* __restrict__ output, int samples, int in_stride)
{
  NAM_AUTOTUNE_NOVECTORIZE
  for (int f = 0; f < samples; f++)
  {
    const float* __restrict__ in_col = input + f * in_stride;
    float* __restrict__ out_col = output + f * M;

    // Initialize with first input channel (=)
    {
      const float in0 = in_col[0];
      for (int o = 0; o < M; o++)
        out_col[o] = weight[o] * in0;
    }
    // Accumulate remaining input channels (+=)
    for (int i = 1; i < N; i++)
    {
      const float in_i = in_col[i];
      const float* __restrict__ w_col = weight + i * M;
      for (int o = 0; o < M; o++)
        out_col[o] += w_col[o] * in_i;
    }
  }
}

// Conv1D per-tap: output(M, samples) += weight(M, N) * input(N, samples)
template <int M, int N>
static void conv1d_pertap_inline(const float* __restrict__ weight, const float* __restrict__ input,
                                 float* __restrict__ output, int samples)
{
  NAM_AUTOTUNE_NOVECTORIZE
  for (int f = 0; f < samples; f++)
  {
    const float* __restrict__ in_col = input + f * N;
    float* __restrict__ out_col = output + f * M;
    for (int i = 0; i < N; i++)
    {
      const float in_i = in_col[i];
      const float* __restrict__ w_col = weight + i * M;
      for (int o = 0; o < M; o++)
        out_col[o] += w_col[o] * in_i;
    }
  }
}

// Conv1D fused: process all kernel_size taps in one pass
template <int M, int N, int KS>
static void conv1d_fused_inline(const float* __restrict__ weights, // KS concatenated M*N weight matrices
                                const float* __restrict__ inputs,  // KS concatenated N*samples input blocks
                                float* __restrict__ output, int samples)
{
  NAM_AUTOTUNE_NOVECTORIZE
  for (int f = 0; f < samples; f++)
  {
    float* __restrict__ out_col = output + f * M;

    // First tap: initialize (=)
    {
      const float* __restrict__ w = weights;
      const float* __restrict__ in_col = inputs + f * N;
      const float in0 = in_col[0];
      for (int o = 0; o < M; o++)
        out_col[o] = w[o] * in0;
      for (int i = 1; i < N; i++)
      {
        const float in_i = in_col[i];
        const float* __restrict__ w_col = w + i * M;
        for (int o = 0; o < M; o++)
          out_col[o] += w_col[o] * in_i;
      }
    }
    // Remaining taps: accumulate (+=)
    for (int k = 1; k < KS; k++)
    {
      const float* __restrict__ w = weights + k * M * N;
      const float* __restrict__ in_col = inputs + k * N * samples + f * N;
      for (int i = 0; i < N; i++)
      {
        const float in_i = in_col[i];
        const float* __restrict__ w_col = w + i * M;
        for (int o = 0; o < M; o++)
          out_col[o] += w_col[o] * in_i;
      }
    }
  }
}

// Depthwise: output(ch, samples) += diag(weight) * input(ch, samples)
template <int CH>
static void conv1d_dw_inline(const float* __restrict__ weight, const float* __restrict__ input,
                             float* __restrict__ output, int samples)
{
  for (int f = 0; f < samples; f++)
  {
    const int off = f * CH;
    for (int c = 0; c < CH; c++)
      output[off + c] += weight[c] * input[off + c];
  }
}

// FiLM: output = input * scale + shift (element-wise)
template <int DIM>
static void film_inline(const float* __restrict__ input, const float* __restrict__ scale,
                        const float* __restrict__ shift, float* __restrict__ output, int samples, int in_stride,
                        int ss_stride)
{
  for (int f = 0; f < samples; f++)
  {
    const float* __restrict__ in_col = input + f * in_stride;
    const float* __restrict__ scale_col = scale + f * ss_stride;
    const float* __restrict__ shift_col = scale_col + DIM;
    float* __restrict__ out_col = output + f * DIM;
    for (int i = 0; i < DIM; i++)
      out_col[i] = in_col[i] * scale_col[i] + shift_col[i];
  }
}

// Generic (runtime) inline GEMM — for the NAM_INLINE_GENERIC_FALLBACK test
static void conv1x1_inline_generic(const float* __restrict__ weight, const float* __restrict__ input,
                                   float* __restrict__ output, int M, int N, int samples, int in_stride)
{
  for (int f = 0; f < samples; f++)
  {
    const float* __restrict__ in_col = input + f * in_stride;
    float* __restrict__ out_col = output + f * M;
    for (int o = 0; o < M; o++)
    {
      float sum = 0.0f;
      for (int i = 0; i < N; i++)
        sum += weight[i * M + o] * in_col[i];
      out_col[o] = sum;
    }
  }
}

// ===========================================================================
// Eigen reference implementations
// ===========================================================================

static void conv1x1_eigen(const Eigen::MatrixXf& weight, const Eigen::MatrixXf& input, Eigen::MatrixXf& output,
                           int samples)
{
  output.leftCols(samples).noalias() = weight * input.leftCols(samples);
}

// ===========================================================================
// Benchmark drivers
// ===========================================================================

template <int M, int N>
static TuneResult tune_conv1x1(int samples, const std::string& macro_name)
{
  Eigen::MatrixXf weight(M, N);
  Eigen::MatrixXf input(N, samples);
  Eigen::MatrixXf output_inline(M, samples);
  Eigen::MatrixXf output_eigen(M, samples);

  fill_random(weight.data(), M * N);
  fill_random(input.data(), N * samples);

  double t_inline = bench_median([&]() {
    conv1x1_inline<M, N>(weight.data(), input.data(), output_inline.data(), samples, N);
    DoNotOptimize(output_inline);
  });

  double t_eigen = bench_median([&]() {
    conv1x1_eigen(weight, input, output_eigen, samples);
    DoNotOptimize(output_eigen);
  });

  return {macro_name, t_inline, t_eigen, t_inline <= t_eigen};
}

template <int M, int N>
static TuneResult tune_conv1d_pertap(int samples, int kernel_size, const std::string& macro_name)
{
  Eigen::MatrixXf weight(M, N);
  Eigen::MatrixXf input(N, samples);
  Eigen::MatrixXf output_inline(M, samples);
  Eigen::MatrixXf output_eigen(M, samples);

  fill_random(weight.data(), M * N);
  fill_random(input.data(), N * samples);

  double t_inline = bench_median([&]() {
    output_inline.setZero();
    for (int k = 0; k < kernel_size; k++)
      conv1d_pertap_inline<M, N>(weight.data(), input.data(), output_inline.data(), samples);
    DoNotOptimize(output_inline);
  });

  double t_eigen = bench_median([&]() {
    output_eigen.setZero();
    for (int k = 0; k < kernel_size; k++)
      output_eigen.leftCols(samples).noalias() += weight * input.leftCols(samples);
    DoNotOptimize(output_eigen);
  });

  return {macro_name, t_inline, t_eigen, t_inline <= t_eigen};
}

template <int M, int N, int KS>
static TuneResult tune_conv1d_fused(int samples, const std::string& macro_name)
{
  // Allocate contiguous blocks for fused kernel
  std::vector<float> weights_flat(KS * M * N);
  std::vector<float> inputs_flat(KS * N * samples);
  fill_random(weights_flat.data(), weights_flat.size());
  fill_random(inputs_flat.data(), inputs_flat.size());

  Eigen::MatrixXf output_inline(M, samples);
  Eigen::MatrixXf output_eigen(M, samples);

  // Also need Eigen matrices for the Eigen path
  std::vector<Eigen::MatrixXf> eigen_weights(KS, Eigen::MatrixXf(M, N));
  std::vector<Eigen::MatrixXf> eigen_inputs(KS, Eigen::MatrixXf(N, samples));
  for (int k = 0; k < KS; k++)
  {
    memcpy(eigen_weights[k].data(), weights_flat.data() + k * M * N, M * N * sizeof(float));
    memcpy(eigen_inputs[k].data(), inputs_flat.data() + k * N * samples, N * samples * sizeof(float));
  }

  double t_inline = bench_median([&]() {
    conv1d_fused_inline<M, N, KS>(weights_flat.data(), inputs_flat.data(), output_inline.data(), samples);
    DoNotOptimize(output_inline);
  });

  double t_eigen = bench_median([&]() {
    output_eigen.setZero();
    for (int k = 0; k < KS; k++)
      output_eigen.leftCols(samples).noalias() += eigen_weights[k] * eigen_inputs[k].leftCols(samples);
    DoNotOptimize(output_eigen);
  });

  return {macro_name, t_inline, t_eigen, t_inline <= t_eigen};
}

template <int CH>
static TuneResult tune_conv1d_dw(int samples, int kernel_size, const std::string& macro_name)
{
  Eigen::VectorXf weight(CH);
  Eigen::MatrixXf input(CH, samples);
  Eigen::MatrixXf output_inline(CH, samples);
  Eigen::MatrixXf output_eigen(CH, samples);

  fill_random(weight.data(), CH);
  fill_random(input.data(), CH * samples);

  double t_inline = bench_median([&]() {
    output_inline.setZero();
    for (int k = 0; k < kernel_size; k++)
      conv1d_dw_inline<CH>(weight.data(), input.data(), output_inline.data(), samples);
    DoNotOptimize(output_inline);
  });

  double t_eigen = bench_median([&]() {
    output_eigen.setZero();
    for (int k = 0; k < kernel_size; k++)
      output_eigen.leftCols(samples).noalias() += weight.asDiagonal() * input.leftCols(samples);
    DoNotOptimize(output_eigen);
  });

  return {macro_name, t_inline, t_eigen, t_inline <= t_eigen};
}

template <int DIM>
static TuneResult tune_film(int samples, const std::string& macro_name)
{
  Eigen::MatrixXf input(DIM, samples);
  // scale_shift has 2*DIM rows (top DIM = scale, bottom DIM = shift)
  Eigen::MatrixXf scale_shift(2 * DIM, samples);
  Eigen::MatrixXf output_inline(DIM, samples);
  Eigen::MatrixXf output_eigen(DIM, samples);

  fill_random(input.data(), DIM * samples);
  fill_random(scale_shift.data(), 2 * DIM * samples);

  double t_inline = bench_median([&]() {
    film_inline<DIM>(input.data(), scale_shift.data(), nullptr, output_inline.data(), samples, DIM, 2 * DIM);
    DoNotOptimize(output_inline);
  });

  double t_eigen = bench_median([&]() {
    const auto scale = scale_shift.topRows(DIM).leftCols(samples);
    const auto shift = scale_shift.bottomRows(DIM).leftCols(samples);
    output_eigen.leftCols(samples).array() =
      input.leftCols(samples).array() * scale.array() + shift.array();
    DoNotOptimize(output_eigen);
  });

  return {macro_name, t_inline, t_eigen, t_inline <= t_eigen};
}

// Generic fallback test (runtime dimensions — NOT unrollable)
static TuneResult tune_generic_fallback(int M, int N, int samples, const std::string& macro_name)
{
  Eigen::MatrixXf weight(M, N);
  Eigen::MatrixXf input(N, samples);
  Eigen::MatrixXf output_inline(M, samples);
  Eigen::MatrixXf output_eigen(M, samples);

  fill_random(weight.data(), M * N);
  fill_random(input.data(), N * samples);

  double t_inline = bench_median([&]() {
    conv1x1_inline_generic(weight.data(), input.data(), output_inline.data(), M, N, samples, N);
    DoNotOptimize(output_inline);
  });

  double t_eigen = bench_median([&]() {
    conv1x1_eigen(weight, input, output_eigen, samples);
    DoNotOptimize(output_eigen);
  });

  return {macro_name, t_inline, t_eigen, t_inline <= t_eigen};
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
  const int samples = 64; // Typical buffer size
  std::vector<TuneResult> results;

  fprintf(stderr, "NAM Auto-Tuner: profiling specializations vs Eigen (buffer_size=%d)\n\n", samples);

  auto print_and_push = [&](TuneResult r) {
    fprintf(stderr, "  %-35s inline: %7.1fns  eigen: %7.1fns  -> %s (%.2fx)\n", r.macro_name.c_str(), r.inline_ns,
            r.eigen_ns, r.use_inline ? "INLINE" : "EIGEN", r.eigen_ns / r.inline_ns);
    results.push_back(r);
  };

  // --- Conv1x1 specializations ---
  fprintf(stderr, "Conv1x1 specializations:\n");
  print_and_push(tune_conv1x1<2, 1>(samples, "NAM_INLINE_CONV1X1_2x1"));
  print_and_push(tune_conv1x1<3, 1>(samples, "NAM_INLINE_CONV1X1_3x1"));
  print_and_push(tune_conv1x1<4, 1>(samples, "NAM_INLINE_CONV1X1_4x1"));
  print_and_push(tune_conv1x1<1, 2>(samples, "NAM_INLINE_CONV1X1_1x2"));
  print_and_push(tune_conv1x1<1, 3>(samples, "NAM_INLINE_CONV1X1_1x3"));
  print_and_push(tune_conv1x1<1, 4>(samples, "NAM_INLINE_CONV1X1_1x4"));
  print_and_push(tune_conv1x1<2, 2>(samples, "NAM_INLINE_CONV1X1_2x2"));
  print_and_push(tune_conv1x1<2, 4>(samples, "NAM_INLINE_CONV1X1_2x4"));
  print_and_push(tune_conv1x1<4, 2>(samples, "NAM_INLINE_CONV1X1_4x2"));
  print_and_push(tune_conv1x1<3, 3>(samples, "NAM_INLINE_CONV1X1_3x3"));
  print_and_push(tune_conv1x1<4, 4>(samples, "NAM_INLINE_CONV1X1_4x4"));
  print_and_push(tune_conv1x1<4, 8>(samples, "NAM_INLINE_CONV1X1_4x8"));
  print_and_push(tune_conv1x1<8, 4>(samples, "NAM_INLINE_CONV1X1_8x4"));
  print_and_push(tune_conv1x1<6, 6>(samples, "NAM_INLINE_CONV1X1_6x6"));
  print_and_push(tune_conv1x1<8, 8>(samples, "NAM_INLINE_CONV1X1_8x8"));

  // --- Conv1D fused kernels ---
  fprintf(stderr, "\nConv1D fused kernels:\n");
  print_and_push(tune_conv1d_fused<4, 4, 3>(samples, "NAM_INLINE_CONV1D_FUSED_3_4x4"));
  print_and_push(tune_conv1d_fused<2, 2, 3>(samples, "NAM_INLINE_CONV1D_FUSED_3_2x2"));
  print_and_push(tune_conv1d_fused<3, 3, 6>(samples, "NAM_INLINE_CONV1D_FUSED_6_3x3"));

  // --- Conv1D per-tap specializations ---
  fprintf(stderr, "\nConv1D per-tap specializations:\n");
  const int typical_ks = 3;
  print_and_push(tune_conv1d_pertap<2, 2>(samples, typical_ks, "NAM_INLINE_CONV1D_2x2"));
  print_and_push(tune_conv1d_pertap<2, 4>(samples, typical_ks, "NAM_INLINE_CONV1D_2x4"));
  print_and_push(tune_conv1d_pertap<4, 1>(samples, typical_ks, "NAM_INLINE_CONV1D_4x1"));
  print_and_push(tune_conv1d_pertap<4, 4>(samples, typical_ks, "NAM_INLINE_CONV1D_4x4"));
  print_and_push(tune_conv1d_pertap<3, 1>(samples, typical_ks, "NAM_INLINE_CONV1D_3x1"));
  print_and_push(tune_conv1d_pertap<3, 3>(samples, typical_ks, "NAM_INLINE_CONV1D_3x3"));
  print_and_push(tune_conv1d_pertap<4, 3>(samples, typical_ks, "NAM_INLINE_CONV1D_4x3"));
  print_and_push(tune_conv1d_pertap<3, 4>(samples, typical_ks, "NAM_INLINE_CONV1D_3x4"));
  print_and_push(tune_conv1d_pertap<6, 1>(samples, typical_ks, "NAM_INLINE_CONV1D_6x1"));
  print_and_push(tune_conv1d_pertap<6, 6>(samples, typical_ks, "NAM_INLINE_CONV1D_6x6"));
  print_and_push(tune_conv1d_pertap<8, 8>(samples, typical_ks, "NAM_INLINE_CONV1D_8x8"));

  // --- Conv1D depthwise ---
  fprintf(stderr, "\nConv1D depthwise:\n");
  print_and_push(tune_conv1d_dw<3>(samples, typical_ks, "NAM_INLINE_CONV1D_DW_3"));
  print_and_push(tune_conv1d_dw<4>(samples, typical_ks, "NAM_INLINE_CONV1D_DW_4"));
  print_and_push(tune_conv1d_dw<8>(samples, typical_ks, "NAM_INLINE_CONV1D_DW_8"));

  // --- FiLM ---
  fprintf(stderr, "\nFiLM:\n");
  print_and_push(tune_film<3>(samples, "NAM_INLINE_FILM_3"));

  // --- Generic fallback (runtime dimensions — tests the generic loop, not unrolled) ---
  fprintf(stderr, "\nGeneric fallback (5x5, runtime dims):\n");
  print_and_push(tune_generic_fallback(5, 5, samples, "NAM_INLINE_GENERIC_FALLBACK"));

  // --- Output header ---
  fprintf(stderr, "\n--- Writing nam_tuning_config.h to stdout ---\n");

  printf("// Auto-generated by nam_autotune — do not edit\n");
  printf("// Platform: %s\n",
#if defined(__aarch64__) || defined(_M_ARM64)
         "ARM64"
#elif defined(__arm__) || defined(_M_ARM)
         "ARM32"
#elif defined(__x86_64__) || defined(_M_X64)
         "x86_64"
#elif defined(__i386__) || defined(_M_IX86)
         "x86"
#elif defined(__EMSCRIPTEN__)
         "WASM"
#else
         "unknown"
#endif
  );
  printf("// Buffer size: %d samples\n", samples);
  printf("#pragma once\n\n");
  printf("#define NAM_TUNING_CONFIG 1\n\n");

  for (const auto& r : results)
  {
    printf("#define %-35s %d  // inline: %.1fns, eigen: %.1fns (%.2fx)\n", r.macro_name.c_str(), r.use_inline ? 1 : 0,
           r.inline_ns, r.eigen_ns, r.eigen_ns / r.inline_ns);
  }
  printf("\n");

  // Summary
  int inline_wins = 0, eigen_wins = 0;
  for (const auto& r : results)
  {
    if (r.use_inline)
      inline_wins++;
    else
      eigen_wins++;
  }
  fprintf(stderr, "\nSummary: %d inline, %d eigen (out of %d total)\n", inline_wins, eigen_wins,
          (int)results.size());

  return 0;
}
