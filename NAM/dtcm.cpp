#include "dtcm.h"

namespace nam
{
namespace dtcm
{

// Static buffers - placed in DTCM on ARM Cortex-M7, regular BSS otherwise
#if NAM_HAS_DTCM
static DTCM_MEM_SECTION float s_weight_buffer[WEIGHT_BUFFER_SIZE];
static DTCM_MEM_SECTION float s_scratch_buffer[SCRATCH_BUFFER_SIZE];
#else
static float s_weight_buffer[WEIGHT_BUFFER_SIZE];
static float s_scratch_buffer[SCRATCH_BUFFER_SIZE];
#endif

// Bump allocator state
static size_t s_weight_offset = 0;

float* get_weight_buffer()
{
  return s_weight_buffer;
}

size_t get_weight_buffer_size()
{
  return WEIGHT_BUFFER_SIZE;
}

float* get_scratch_buffer()
{
  return s_scratch_buffer;
}

size_t get_scratch_buffer_size()
{
  return SCRATCH_BUFFER_SIZE;
}

float* allocate_weights(size_t num_floats)
{
  if (s_weight_offset + num_floats > WEIGHT_BUFFER_SIZE)
    return nullptr;
  float* ptr = s_weight_buffer + s_weight_offset;
  s_weight_offset += num_floats;
  return ptr;
}

void reset_weight_allocator()
{
  s_weight_offset = 0;
}

size_t get_weights_used()
{
  return s_weight_offset;
}

size_t get_weights_available()
{
  return WEIGHT_BUFFER_SIZE - s_weight_offset;
}

} // namespace dtcm
} // namespace nam
