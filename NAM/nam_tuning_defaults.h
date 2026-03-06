// nam_tuning_defaults.h — Default per-operation inline GEMM flags
//
// When NAM_USE_INLINE_GEMM is defined but no tuning config is provided,
// all specializations default to enabled (1).
//
// To auto-tune: build and run tools/autotune (desktop) or daisy/NAMAutotune
// (on-target), which profiles each specialization vs Eigen and outputs a
// nam_tuning_config.h with optimal choices.  Then rebuild with:
//   -DNAM_TUNING_CONFIG -include path/to/nam_tuning_config.h
//
// Each macro controls whether the corresponding inline specialization is used.
// When set to 0, the operation falls through to the generic fallback (if enabled)
// or to Eigen.
#pragma once

// ============================================================================
// Conv1x1 specializations (out_ch x in_ch)
// Used in Conv1x1::process_() — NAM/dsp.cpp
// ============================================================================
#ifndef NAM_INLINE_CONV1X1_2x1
#define NAM_INLINE_CONV1X1_2x1  1
#endif
#ifndef NAM_INLINE_CONV1X1_3x1
#define NAM_INLINE_CONV1X1_3x1  1
#endif
#ifndef NAM_INLINE_CONV1X1_4x1
#define NAM_INLINE_CONV1X1_4x1  1
#endif
#ifndef NAM_INLINE_CONV1X1_1x2
#define NAM_INLINE_CONV1X1_1x2  1
#endif
#ifndef NAM_INLINE_CONV1X1_1x3
#define NAM_INLINE_CONV1X1_1x3  1
#endif
#ifndef NAM_INLINE_CONV1X1_1x4
#define NAM_INLINE_CONV1X1_1x4  1
#endif
#ifndef NAM_INLINE_CONV1X1_2x2
#define NAM_INLINE_CONV1X1_2x2  1
#endif
#ifndef NAM_INLINE_CONV1X1_2x4
#define NAM_INLINE_CONV1X1_2x4  1
#endif
#ifndef NAM_INLINE_CONV1X1_4x2
#define NAM_INLINE_CONV1X1_4x2  1
#endif
#ifndef NAM_INLINE_CONV1X1_3x3
#define NAM_INLINE_CONV1X1_3x3  1
#endif
#ifndef NAM_INLINE_CONV1X1_4x4
#define NAM_INLINE_CONV1X1_4x4  1
#endif
#ifndef NAM_INLINE_CONV1X1_4x8
#define NAM_INLINE_CONV1X1_4x8  1
#endif
#ifndef NAM_INLINE_CONV1X1_8x4
#define NAM_INLINE_CONV1X1_8x4  1
#endif
#ifndef NAM_INLINE_CONV1X1_6x6
#define NAM_INLINE_CONV1X1_6x6  1
#endif
#ifndef NAM_INLINE_CONV1X1_8x8
#define NAM_INLINE_CONV1X1_8x8  1
#endif

// ============================================================================
// Conv1D fused kernels (kernel_size, out_ch x in_ch)
// Used in Conv1D::Process() — NAM/conv1d.cpp
// These fuse all kernel taps into a single pass over frames.
// When disabled, the operation falls through to the per-tap loop.
// ============================================================================
#ifndef NAM_INLINE_CONV1D_FUSED_3_4x4
#define NAM_INLINE_CONV1D_FUSED_3_4x4  1
#endif
#ifndef NAM_INLINE_CONV1D_FUSED_3_2x2
#define NAM_INLINE_CONV1D_FUSED_3_2x2  1
#endif
#ifndef NAM_INLINE_CONV1D_FUSED_6_3x3
#define NAM_INLINE_CONV1D_FUSED_6_3x3  1
#endif

// ============================================================================
// Conv1D per-tap non-depthwise specializations (out_ch x in_ch)
// Used in Conv1D::Process() general path — NAM/conv1d.cpp
// ============================================================================
#ifndef NAM_INLINE_CONV1D_2x2
#define NAM_INLINE_CONV1D_2x2  1
#endif
#ifndef NAM_INLINE_CONV1D_2x4
#define NAM_INLINE_CONV1D_2x4  1
#endif
#ifndef NAM_INLINE_CONV1D_4x1
#define NAM_INLINE_CONV1D_4x1  1
#endif
#ifndef NAM_INLINE_CONV1D_4x4
#define NAM_INLINE_CONV1D_4x4  1
#endif
#ifndef NAM_INLINE_CONV1D_3x1
#define NAM_INLINE_CONV1D_3x1  1
#endif
#ifndef NAM_INLINE_CONV1D_3x3
#define NAM_INLINE_CONV1D_3x3  1
#endif
#ifndef NAM_INLINE_CONV1D_4x3
#define NAM_INLINE_CONV1D_4x3  1
#endif
#ifndef NAM_INLINE_CONV1D_3x4
#define NAM_INLINE_CONV1D_3x4  1
#endif
#ifndef NAM_INLINE_CONV1D_6x1
#define NAM_INLINE_CONV1D_6x1  1
#endif
#ifndef NAM_INLINE_CONV1D_6x6
#define NAM_INLINE_CONV1D_6x6  1
#endif
#ifndef NAM_INLINE_CONV1D_8x8
#define NAM_INLINE_CONV1D_8x8  1
#endif

// ============================================================================
// Conv1D depthwise unrolled specializations (channels)
// Used in Conv1D::Process() depthwise path — NAM/conv1d.cpp
// ============================================================================
#ifndef NAM_INLINE_CONV1D_DW_3
#define NAM_INLINE_CONV1D_DW_3  1
#endif
#ifndef NAM_INLINE_CONV1D_DW_4
#define NAM_INLINE_CONV1D_DW_4  1
#endif
#ifndef NAM_INLINE_CONV1D_DW_8
#define NAM_INLINE_CONV1D_DW_8  1
#endif

// ============================================================================
// FiLM specializations (input_dim)
// Used in FiLM::Process() — NAM/film.h
// ============================================================================
#ifndef NAM_INLINE_FILM_3
#define NAM_INLINE_FILM_3  1
#endif

// ============================================================================
// Generic fallback — inline loop for sizes without a specialization
// When disabled, unmatched sizes fall through to Eigen.
// Applies to Conv1x1, Conv1D (per-tap and depthwise), and FiLM.
// ============================================================================
#ifndef NAM_INLINE_GENERIC_FALLBACK
#define NAM_INLINE_GENERIC_FALLBACK  1
#endif
