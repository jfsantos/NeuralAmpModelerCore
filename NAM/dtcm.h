#pragma once

#include <cstddef>

// DTCM support for ARM Cortex-M7 (STM32H750)
// On Daisy, DTCM is 128KB of single-cycle access memory
//
// This module provides a static weight buffer in DTCM for fastest memory access
// during model inference. On non-ARM platforms, the buffers are placed in regular
// BSS and all code works identically (just without the performance benefit).

#if defined(__ARM_ARCH_7EM__) || defined(ARM_MATH_CM7)
#ifndef DTCM_MEM_SECTION
#define DTCM_MEM_SECTION __attribute__((section(".dtcmram_bss")))
#endif
#define NAM_HAS_DTCM 1
#else
#define DTCM_MEM_SECTION
#define NAM_HAS_DTCM 0
#endif

namespace nam
{
namespace dtcm
{

// Weight buffer: 96KB for weights (leaves ~24KB for stack + scratch)
// 96KB = 24576 floats (each float is 4 bytes)
constexpr size_t WEIGHT_BUFFER_SIZE = 24576;

// Scratch buffer: 16KB for temporary computations
// 16KB = 4096 floats
constexpr size_t SCRATCH_BUFFER_SIZE = 4096;

// Get pointer to DTCM weight buffer (for model weights)
float* get_weight_buffer();
size_t get_weight_buffer_size();

// Get pointer to DTCM scratch buffer (for temporary computations)
float* get_scratch_buffer();
size_t get_scratch_buffer_size();

// Simple bump allocator for weight buffer
// Returns nullptr if not enough space
float* allocate_weights(size_t num_floats);

// Reset weight allocator (call before loading new model)
void reset_weight_allocator();

// Get bytes used/available (in floats)
size_t get_weights_used();
size_t get_weights_available();

} // namespace dtcm
} // namespace nam
