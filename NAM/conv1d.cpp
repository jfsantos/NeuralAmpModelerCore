#include "conv1d.h"
#include <cstring>
#include <stdexcept>

namespace nam
{
// Conv1D =====================================================================

void Conv1D::set_weights_(std::vector<float>::iterator& weights)
{
  if (this->_is_depthwise)
  {
    // Depthwise convolution: one weight per channel per kernel tap
    // Weight layout: for each channel c, for each kernel position k
    const int channels = this->_channels;
    const size_t kernel_size = this->_depthwise_weight.size();
    for (int c = 0; c < channels; c++)
    {
      for (size_t k = 0; k < kernel_size; k++)
      {
        this->_depthwise_weight[k](c) = *(weights++);
      }
    }
  }
  else if (this->_weight.size() > 0)
  {
    const long out_channels = this->_weight[0].rows();
    const long in_channels = this->_weight[0].cols();
    const int numGroups = this->_num_groups;
    const long out_per_group = out_channels / numGroups;
    const long in_per_group = in_channels / numGroups;

    // For grouped convolutions, weights are organized per group
    // Weight layout: for each kernel position k, weights are [group0, group1, ..., groupN-1]
    // Each group's weight matrix is (out_channels/numGroups, in_channels/numGroups)
    // Crazy ordering because that's how it gets flattened.
    for (int g = 0; g < numGroups; g++)
    {
      for (auto i = 0; i < out_per_group; i++)
      {
        for (auto j = 0; j < in_per_group; j++)
        {
          for (size_t k = 0; k < this->_weight.size(); k++)
          {
            this->_weight[k](g * out_per_group + i, g * in_per_group + j) = *(weights++);
          }
        }
      }
    }
  }
  for (long i = 0; i < this->_bias.size(); i++)
    this->_bias(i) = *(weights++);
}

void Conv1D::set_size_(const int in_channels, const int out_channels, const int kernel_size, const bool do_bias,
                       const int _dilation, const int groups)
{
  // Validate that channels divide evenly by groups
  if (in_channels % groups != 0)
  {
    throw std::runtime_error("in_channels (" + std::to_string(in_channels) + ") must be divisible by numGroups ("
                             + std::to_string(groups) + ")");
  }
  if (out_channels % groups != 0)
  {
    throw std::runtime_error("out_channels (" + std::to_string(out_channels) + ") must be divisible by numGroups ("
                             + std::to_string(groups) + ")");
  }

  this->_num_groups = groups;
  this->_dilation = _dilation;

  // Check for depthwise convolution: groups == in_channels == out_channels
  // In this case, each channel is processed independently with a single weight per kernel tap,
  // so we can use efficient element-wise multiplication instead of matrix multiplication.
  this->_is_depthwise = (groups == in_channels && in_channels == out_channels);

  if (this->_is_depthwise)
  {
    // Depthwise: store one weight vector per kernel tap
    this->_channels = in_channels;
    this->_depthwise_weight.resize(kernel_size);
    for (int i = 0; i < kernel_size; i++)
    {
      this->_depthwise_weight[i].resize(in_channels);
      this->_depthwise_weight[i].setZero();
    }
    this->_weight.clear(); // Not used for depthwise
  }
  else
  {
    // Non-depthwise: store full weight matrices (block-diagonal for grouped convolutions)
    this->_weight.resize(kernel_size);
    for (int i = 0; i < kernel_size; i++)
    {
      this->_weight[i].resize(out_channels,
                              in_channels); // y = Ax, input array (C,L)
      this->_weight[i].setZero();
    }
    this->_depthwise_weight.clear(); // Not used for non-depthwise
    this->_channels = 0;
  }

  if (do_bias)
  {
    this->_bias.resize(out_channels);
    this->_bias.setZero();
  }
  else
    this->_bias.resize(0);
}

void Conv1D::set_size_and_weights_(const int in_channels, const int out_channels, const int kernel_size,
                                   const int _dilation, const bool do_bias, const int groups,
                                   std::vector<float>::iterator& weights)
{
  this->set_size_(in_channels, out_channels, kernel_size, do_bias, _dilation, groups);
  this->set_weights_(weights);
}

void Conv1D::SetMaxBufferSize(const int maxBufferSize)
{
  _max_buffer_size = maxBufferSize;

  // Calculate receptive field (maximum lookback needed)
  const long kernel_size = get_kernel_size();
  const long dilation = get_dilation();
  const long receptive_field = kernel_size > 0 ? (kernel_size - 1) * dilation : 0;

  const long in_channels = get_in_channels();

  // Initialize input ring buffer
  // Set max lookback before Reset so that Reset() can use it to calculate storage size
  // Reset() will calculate storage size as: 2 * max_lookback + max_buffer_size
  _input_buffer.SetMaxLookback(receptive_field);
  _input_buffer.Reset(in_channels, maxBufferSize);

  // Pre-allocate output matrix
  const long out_channels = get_out_channels();
  _output.resize(out_channels, maxBufferSize);
  _output.setZero();

#ifdef NAM_USE_CMSIS_DSP
  // Pre-allocate temporary buffer for CMSIS-DSP matrix multiplication
  // Only need temp buffer for accumulation (arm_mat_mult_f32 overwrites dest)
  _cmsis_temp.resize(out_channels, maxBufferSize);
  _cmsis_temp.setZero();
#endif
}


void Conv1D::Process(const Eigen::MatrixXf& input, const int num_frames)
{
  // Write input to ring buffer
  _input_buffer.Write(input, num_frames);

  // Zero output before processing
  _output.leftCols(num_frames).setZero();

  // Process from ring buffer with dilation lookback
  // After Write(), data is at positions [_write_pos, _write_pos+num_frames-1]
  // For kernel tap k with offset, we need to read from _write_pos + offset
  // The offset is negative (looking back), so _write_pos + offset reads from earlier positions

  if (this->_is_depthwise)
  {
    // Depthwise convolution: use efficient element-wise multiplication
    // Each channel is processed independently with a single weight per kernel tap.
    // output[c, t] = sum_k(weight[k, c] * input[c, t - k*dilation])
    const size_t kernel_size = this->_depthwise_weight.size();
#ifdef NAM_USE_INLINE_GEMM
    const int channels = this->_channels;
    float* __restrict__ output_ptr = _output.data();

    for (size_t k = 0; k < kernel_size; k++)
    {
      const long offset = this->_dilation * (k + 1 - (long)kernel_size);
      const long lookback = -offset;
      auto input_block = _input_buffer.Read(num_frames, lookback);
      const float* __restrict__ input_ptr = input_block.data();
      const float* __restrict__ weight_ptr = this->_depthwise_weight[k].data();

      // Specialized paths for common channel counts
      if (channels == 4)
      {
        const float w0 = weight_ptr[0], w1 = weight_ptr[1];
        const float w2 = weight_ptr[2], w3 = weight_ptr[3];
        for (int f = 0; f < num_frames; f++)
        {
          const int off = f * 4;
          output_ptr[off] += w0 * input_ptr[off];
          output_ptr[off + 1] += w1 * input_ptr[off + 1];
          output_ptr[off + 2] += w2 * input_ptr[off + 2];
          output_ptr[off + 3] += w3 * input_ptr[off + 3];
        }
      }
      else if (channels == 8)
      {
        const float w0 = weight_ptr[0], w1 = weight_ptr[1], w2 = weight_ptr[2], w3 = weight_ptr[3];
        const float w4 = weight_ptr[4], w5 = weight_ptr[5], w6 = weight_ptr[6], w7 = weight_ptr[7];
        for (int f = 0; f < num_frames; f++)
        {
          const int off = f * 8;
          output_ptr[off] += w0 * input_ptr[off];
          output_ptr[off + 1] += w1 * input_ptr[off + 1];
          output_ptr[off + 2] += w2 * input_ptr[off + 2];
          output_ptr[off + 3] += w3 * input_ptr[off + 3];
          output_ptr[off + 4] += w4 * input_ptr[off + 4];
          output_ptr[off + 5] += w5 * input_ptr[off + 5];
          output_ptr[off + 6] += w6 * input_ptr[off + 6];
          output_ptr[off + 7] += w7 * input_ptr[off + 7];
        }
      }
      else
      {
        // General depthwise path with loop unrolling
        for (int f = 0; f < num_frames; f++)
        {
          const int off = f * channels;
          int c = 0;
          for (; c + 3 < channels; c += 4)
          {
            output_ptr[off + c] += weight_ptr[c] * input_ptr[off + c];
            output_ptr[off + c + 1] += weight_ptr[c + 1] * input_ptr[off + c + 1];
            output_ptr[off + c + 2] += weight_ptr[c + 2] * input_ptr[off + c + 2];
            output_ptr[off + c + 3] += weight_ptr[c + 3] * input_ptr[off + c + 3];
          }
          for (; c < channels; c++)
          {
            output_ptr[off + c] += weight_ptr[c] * input_ptr[off + c];
          }
        }
      }
    }
#else
    for (size_t k = 0; k < kernel_size; k++)
    {
      const long offset = this->_dilation * (k + 1 - (long)kernel_size);
      const long lookback = -offset;
      auto input_block = _input_buffer.Read(num_frames, lookback);
      // Element-wise multiply: each row of input_block is multiplied by corresponding weight
      _output.leftCols(num_frames).noalias() +=
        this->_depthwise_weight[k].asDiagonal() * input_block.leftCols(num_frames);
    }
#endif
  }
  else
  {
#ifdef NAM_USE_CMSIS_DSP
    // Use CMSIS-DSP optimized matrix multiplication for ARM Cortex-M
    //
    // Eigen uses column-major storage, CMSIS-DSP uses row-major.
    // Column-major A is equivalent to row-major A^T.
    // For C = A * B, we compute C^T = B^T * A^T using CMSIS-DSP.
    //
    // For output = weight * input:
    //   CMSIS computes: temp = input_data * weight_data (row-major interpretation)
    //   Which gives us: output = weight * input (column-major interpretation)

    const uint16_t out_ch = (uint16_t)get_out_channels();
    const uint16_t in_ch = (uint16_t)get_in_channels();
    const uint16_t frames = (uint16_t)num_frames;
    const uint32_t output_size = (uint32_t)(out_ch * frames);
    const size_t kernel_size = this->_weight.size();

    // Pre-initialize matrix descriptors (dimensions are fixed)
    arm_matrix_instance_f32 matInput = {frames, in_ch, nullptr};
    arm_matrix_instance_f32 matWeight = {in_ch, out_ch, nullptr};
    arm_matrix_instance_f32 matTemp = {frames, out_ch, _cmsis_temp.data()};

    float* output_ptr = _output.data();

    for (size_t k = 0; k < kernel_size; k++)
    {
      const long offset = this->_dilation * (k + 1 - (long)kernel_size);
      const long lookback = -offset;

      // Ring buffer block is contiguous (column-major, starting at row 0)
      auto input_block = _input_buffer.Read(num_frames, lookback);

      // Point directly to ring buffer data (no copy needed)
      matInput.pData = const_cast<float*>(input_block.data());
      matWeight.pData = this->_weight[k].data();

      // Compute: temp = input * weight (row-major) = weight * input (col-major)
      arm_mat_mult_f32(&matInput, &matWeight, &matTemp);

      // Accumulate into output
      arm_add_f32(output_ptr, _cmsis_temp.data(), output_ptr, output_size);
    }
#elif defined(NAM_USE_INLINE_GEMM)
    // Hand-optimized inline GEMM for small matrices
    // For output(out_ch, frames) += weight(out_ch, in_ch) * input(in_ch, frames)
    //
    // Column-major layout means:
    //   output(o, f) is at output_ptr[f * out_ch + o]
    //   weight(o, i) is at weight_ptr[i * out_ch + o]
    //   input(i, f) is at input_ptr[f * in_ch + i]
    const int out_ch = (int)get_out_channels();
    const int in_ch = (int)get_in_channels();
    const size_t kernel_size = this->_weight.size();

    for (size_t k = 0; k < kernel_size; k++)
    {
      const long offset = this->_dilation * (k + 1 - (long)kernel_size);
      const long lookback = -offset;
      auto input_block = _input_buffer.Read(num_frames, lookback);

      const float* __restrict__ input_ptr = input_block.data();
      const float* __restrict__ weight_ptr = this->_weight[k].data();
      float* __restrict__ output_ptr = _output.data();

      // Specialized fully-unrolled paths for common small channel counts
      // These avoid all loop overhead for the tiny matrices in NAM models
      if (out_ch == 2 && in_ch == 2)
      {
        // 2x2 fully unrolled
        const float w00 = weight_ptr[0], w10 = weight_ptr[1];
        const float w01 = weight_ptr[2], w11 = weight_ptr[3];
        for (int f = 0; f < num_frames; f++)
        {
          const float i0 = input_ptr[f * 2];
          const float i1 = input_ptr[f * 2 + 1];
          output_ptr[f * 2] += w00 * i0 + w01 * i1;
          output_ptr[f * 2 + 1] += w10 * i0 + w11 * i1;
        }
      }
      else if (out_ch == 2 && in_ch == 4)
      {
        // 2x4 fully unrolled
        const float w00 = weight_ptr[0], w10 = weight_ptr[1];
        const float w01 = weight_ptr[2], w11 = weight_ptr[3];
        const float w02 = weight_ptr[4], w12 = weight_ptr[5];
        const float w03 = weight_ptr[6], w13 = weight_ptr[7];
        for (int f = 0; f < num_frames; f++)
        {
          const float i0 = input_ptr[f * 4];
          const float i1 = input_ptr[f * 4 + 1];
          const float i2 = input_ptr[f * 4 + 2];
          const float i3 = input_ptr[f * 4 + 3];
          output_ptr[f * 2] += w00 * i0 + w01 * i1 + w02 * i2 + w03 * i3;
          output_ptr[f * 2 + 1] += w10 * i0 + w11 * i1 + w12 * i2 + w13 * i3;
        }
      }
      else if (out_ch == 4 && in_ch == 1)
      {
        // 4x1 fully unrolled
        const float w0 = weight_ptr[0], w1 = weight_ptr[1];
        const float w2 = weight_ptr[2], w3 = weight_ptr[3];
        for (int f = 0; f < num_frames; f++)
        {
          const float in_val = input_ptr[f];
          output_ptr[f * 4] += w0 * in_val;
          output_ptr[f * 4 + 1] += w1 * in_val;
          output_ptr[f * 4 + 2] += w2 * in_val;
          output_ptr[f * 4 + 3] += w3 * in_val;
        }
      }
      else if (out_ch == 4 && in_ch == 4)
      {
        // 4x4 fully unrolled - cache weights in registers
        const float w00 = weight_ptr[0], w10 = weight_ptr[1], w20 = weight_ptr[2], w30 = weight_ptr[3];
        const float w01 = weight_ptr[4], w11 = weight_ptr[5], w21 = weight_ptr[6], w31 = weight_ptr[7];
        const float w02 = weight_ptr[8], w12 = weight_ptr[9], w22 = weight_ptr[10], w32 = weight_ptr[11];
        const float w03 = weight_ptr[12], w13 = weight_ptr[13], w23 = weight_ptr[14], w33 = weight_ptr[15];
        for (int f = 0; f < num_frames; f++)
        {
          const int in_off = f * 4;
          const int out_off = f * 4;
          const float i0 = input_ptr[in_off];
          const float i1 = input_ptr[in_off + 1];
          const float i2 = input_ptr[in_off + 2];
          const float i3 = input_ptr[in_off + 3];
          output_ptr[out_off] += w00 * i0 + w01 * i1 + w02 * i2 + w03 * i3;
          output_ptr[out_off + 1] += w10 * i0 + w11 * i1 + w12 * i2 + w13 * i3;
          output_ptr[out_off + 2] += w20 * i0 + w21 * i1 + w22 * i2 + w23 * i3;
          output_ptr[out_off + 3] += w30 * i0 + w31 * i1 + w32 * i2 + w33 * i3;
        }
      }
      else if (out_ch == 3 && in_ch == 1)
      {
        // 3x1 fully unrolled
        const float w0 = weight_ptr[0], w1 = weight_ptr[1], w2 = weight_ptr[2];
        for (int f = 0; f < num_frames; f++)
        {
          const float in_val = input_ptr[f];
          output_ptr[f * 3] += w0 * in_val;
          output_ptr[f * 3 + 1] += w1 * in_val;
          output_ptr[f * 3 + 2] += w2 * in_val;
        }
      }
      else if (out_ch == 3 && in_ch == 3)
      {
        // 3x3 fully unrolled
        const float w00 = weight_ptr[0], w10 = weight_ptr[1], w20 = weight_ptr[2];
        const float w01 = weight_ptr[3], w11 = weight_ptr[4], w21 = weight_ptr[5];
        const float w02 = weight_ptr[6], w12 = weight_ptr[7], w22 = weight_ptr[8];
        for (int f = 0; f < num_frames; f++)
        {
          const int off = f * 3;
          const float i0 = input_ptr[off];
          const float i1 = input_ptr[off + 1];
          const float i2 = input_ptr[off + 2];
          output_ptr[off] += w00 * i0 + w01 * i1 + w02 * i2;
          output_ptr[off + 1] += w10 * i0 + w11 * i1 + w12 * i2;
          output_ptr[off + 2] += w20 * i0 + w21 * i1 + w22 * i2;
        }
      }
      else if (out_ch == 4 && in_ch == 3)
      {
        // 4x3 fully unrolled
        const float w00 = weight_ptr[0], w10 = weight_ptr[1], w20 = weight_ptr[2], w30 = weight_ptr[3];
        const float w01 = weight_ptr[4], w11 = weight_ptr[5], w21 = weight_ptr[6], w31 = weight_ptr[7];
        const float w02 = weight_ptr[8], w12 = weight_ptr[9], w22 = weight_ptr[10], w32 = weight_ptr[11];
        for (int f = 0; f < num_frames; f++)
        {
          const float i0 = input_ptr[f * 3];
          const float i1 = input_ptr[f * 3 + 1];
          const float i2 = input_ptr[f * 3 + 2];
          output_ptr[f * 4] += w00 * i0 + w01 * i1 + w02 * i2;
          output_ptr[f * 4 + 1] += w10 * i0 + w11 * i1 + w12 * i2;
          output_ptr[f * 4 + 2] += w20 * i0 + w21 * i1 + w22 * i2;
          output_ptr[f * 4 + 3] += w30 * i0 + w31 * i1 + w32 * i2;
        }
      }
      else if (out_ch == 3 && in_ch == 4)
      {
        // 3x4 fully unrolled
        const float w00 = weight_ptr[0], w10 = weight_ptr[1], w20 = weight_ptr[2];
        const float w01 = weight_ptr[3], w11 = weight_ptr[4], w21 = weight_ptr[5];
        const float w02 = weight_ptr[6], w12 = weight_ptr[7], w22 = weight_ptr[8];
        const float w03 = weight_ptr[9], w13 = weight_ptr[10], w23 = weight_ptr[11];
        for (int f = 0; f < num_frames; f++)
        {
          const float i0 = input_ptr[f * 4];
          const float i1 = input_ptr[f * 4 + 1];
          const float i2 = input_ptr[f * 4 + 2];
          const float i3 = input_ptr[f * 4 + 3];
          output_ptr[f * 3] += w00 * i0 + w01 * i1 + w02 * i2 + w03 * i3;
          output_ptr[f * 3 + 1] += w10 * i0 + w11 * i1 + w12 * i2 + w13 * i3;
          output_ptr[f * 3 + 2] += w20 * i0 + w21 * i1 + w22 * i2 + w23 * i3;
        }
      }
      else if (out_ch == 6 && in_ch == 1)
      {
        // 6x1 fully unrolled
        const float w0 = weight_ptr[0], w1 = weight_ptr[1], w2 = weight_ptr[2];
        const float w3 = weight_ptr[3], w4 = weight_ptr[4], w5 = weight_ptr[5];
        for (int f = 0; f < num_frames; f++)
        {
          const float in_val = input_ptr[f];
          const int off = f * 6;
          output_ptr[off] += w0 * in_val;
          output_ptr[off + 1] += w1 * in_val;
          output_ptr[off + 2] += w2 * in_val;
          output_ptr[off + 3] += w3 * in_val;
          output_ptr[off + 4] += w4 * in_val;
          output_ptr[off + 5] += w5 * in_val;
        }
      }
      else if (out_ch == 6 && in_ch == 6)
      {
        // 6x6 - unroll weights, loop over frames
        for (int f = 0; f < num_frames; f++)
        {
          const float* __restrict__ in_col = input_ptr + f * 6;
          float* __restrict__ out_col = output_ptr + f * 6;
          const float i0 = in_col[0], i1 = in_col[1], i2 = in_col[2];
          const float i3 = in_col[3], i4 = in_col[4], i5 = in_col[5];
          for (int o = 0; o < 6; o++)
          {
            out_col[o] += weight_ptr[o] * i0 + weight_ptr[6 + o] * i1 + weight_ptr[12 + o] * i2
                          + weight_ptr[18 + o] * i3 + weight_ptr[24 + o] * i4 + weight_ptr[30 + o] * i5;
          }
        }
      }
      else if (out_ch == 8 && in_ch == 8)
      {
        // 8x8 - unroll weights, loop over frames
        for (int f = 0; f < num_frames; f++)
        {
          const float* __restrict__ in_col = input_ptr + f * 8;
          float* __restrict__ out_col = output_ptr + f * 8;
          const float i0 = in_col[0], i1 = in_col[1], i2 = in_col[2], i3 = in_col[3];
          const float i4 = in_col[4], i5 = in_col[5], i6 = in_col[6], i7 = in_col[7];
          for (int o = 0; o < 8; o++)
          {
            out_col[o] += weight_ptr[o] * i0 + weight_ptr[8 + o] * i1 + weight_ptr[16 + o] * i2
                          + weight_ptr[24 + o] * i3 + weight_ptr[32 + o] * i4 + weight_ptr[40 + o] * i5
                          + weight_ptr[48 + o] * i6 + weight_ptr[56 + o] * i7;
          }
        }
      }
      else
      {
        // Fall back to Eigen for larger matrices where it's more efficient
        _output.leftCols(num_frames).noalias() += this->_weight[k] * input_block;
      }
    }
#else
    // Eigen fallback for non-ARM platforms
    // Grouped convolution note: The weight matrices are block-diagonal (zeros off-diagonal),
    // so we can use a single GEMM for all cases. A more advanced implementation could store
    // compact per-group weight matrices and loop over groups, but at typical model sizes
    // (e.g. 8 channels, 4 groups, 64 samples), the GEMM call overhead tends to dominate
    // and the single sparse GEMM approach is faster.
    for (size_t k = 0; k < this->_weight.size(); k++)
    {
      const long offset = this->_dilation * (k + 1 - (long)this->_weight.size());
      const long lookback = -offset;
      auto input_block = _input_buffer.Read(num_frames, lookback);
      _output.leftCols(num_frames).noalias() += this->_weight[k] * input_block;
    }
#endif
  }

  // Add bias if present
  if (this->_bias.size() > 0)
  {
#if defined(NAM_USE_CMSIS_DSP)
    // Use CMSIS-DSP for bias addition (broadcast across frames)
    const long out_channels = get_out_channels();
    float* out_ptr = _output.data();
    // CMSIS-DSP takes non-const pointers, but bias won't be modified
    float* bias_ptr = const_cast<float*>(this->_bias.data());
    for (int frame = 0; frame < num_frames; frame++)
    {
      arm_add_f32(out_ptr, bias_ptr, out_ptr, (uint32_t)out_channels);
      out_ptr += out_channels;
    }
#elif defined(NAM_USE_INLINE_GEMM)
    // Inline bias addition for small channel counts
    const int out_ch = (int)get_out_channels();
    float* __restrict__ output_ptr = _output.data();
    const float* __restrict__ bias_ptr = this->_bias.data();

    if (out_ch == 2)
    {
      const float b0 = bias_ptr[0], b1 = bias_ptr[1];
      for (int f = 0; f < num_frames; f++)
      {
        output_ptr[f * 2] += b0;
        output_ptr[f * 2 + 1] += b1;
      }
    }
    else if (out_ch == 3)
    {
      const float b0 = bias_ptr[0], b1 = bias_ptr[1], b2 = bias_ptr[2];
      for (int f = 0; f < num_frames; f++)
      {
        output_ptr[f * 3] += b0;
        output_ptr[f * 3 + 1] += b1;
        output_ptr[f * 3 + 2] += b2;
      }
    }
    else if (out_ch == 4)
    {
      const float b0 = bias_ptr[0], b1 = bias_ptr[1];
      const float b2 = bias_ptr[2], b3 = bias_ptr[3];
      for (int f = 0; f < num_frames; f++)
      {
        output_ptr[f * 4] += b0;
        output_ptr[f * 4 + 1] += b1;
        output_ptr[f * 4 + 2] += b2;
        output_ptr[f * 4 + 3] += b3;
      }
    }
    else if (out_ch == 6)
    {
      const float b0 = bias_ptr[0], b1 = bias_ptr[1], b2 = bias_ptr[2];
      const float b3 = bias_ptr[3], b4 = bias_ptr[4], b5 = bias_ptr[5];
      for (int f = 0; f < num_frames; f++)
      {
        const int off = f * 6;
        output_ptr[off] += b0;
        output_ptr[off + 1] += b1;
        output_ptr[off + 2] += b2;
        output_ptr[off + 3] += b3;
        output_ptr[off + 4] += b4;
        output_ptr[off + 5] += b5;
      }
    }
    else if (out_ch == 8)
    {
      const float b0 = bias_ptr[0], b1 = bias_ptr[1], b2 = bias_ptr[2], b3 = bias_ptr[3];
      const float b4 = bias_ptr[4], b5 = bias_ptr[5], b6 = bias_ptr[6], b7 = bias_ptr[7];
      for (int f = 0; f < num_frames; f++)
      {
        const int off = f * 8;
        output_ptr[off] += b0;
        output_ptr[off + 1] += b1;
        output_ptr[off + 2] += b2;
        output_ptr[off + 3] += b3;
        output_ptr[off + 4] += b4;
        output_ptr[off + 5] += b5;
        output_ptr[off + 6] += b6;
        output_ptr[off + 7] += b7;
      }
    }
    else
    {
      for (int f = 0; f < num_frames; f++)
      {
        for (int o = 0; o < out_ch; o++)
        {
          output_ptr[f * out_ch + o] += bias_ptr[o];
        }
      }
    }
#else
    _output.leftCols(num_frames).colwise() += this->_bias;
#endif
  }

  // Advance ring buffer write pointer after processing
  _input_buffer.Advance(num_frames);
}

void Conv1D::process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start, const long ncols,
                      const long j_start) const
{
  if (this->_is_depthwise)
  {
    // Depthwise convolution: use efficient element-wise multiplication
    const size_t kernel_size = this->_depthwise_weight.size();
    for (size_t k = 0; k < kernel_size; k++)
    {
      const long offset = this->_dilation * (k + 1 - (long)kernel_size);
      if (k == 0)
        output.middleCols(j_start, ncols).noalias() =
          this->_depthwise_weight[k].asDiagonal() * input.middleCols(i_start + offset, ncols);
      else
        output.middleCols(j_start, ncols).noalias() +=
          this->_depthwise_weight[k].asDiagonal() * input.middleCols(i_start + offset, ncols);
    }
  }
  else
  {
    // Grouped convolution note: The weight matrices are block-diagonal (zeros off-diagonal),
    // so we can use a single GEMM for all cases. A more advanced implementation could store
    // compact per-group weight matrices and loop over groups, but at typical model sizes
    // (e.g. 8 channels, 4 groups, 64 samples), the GEMM call overhead tends to dominate
    // and the single sparse GEMM approach is faster.
    for (size_t k = 0; k < this->_weight.size(); k++)
    {
      const long offset = this->_dilation * (k + 1 - this->_weight.size());
      if (k == 0)
        output.middleCols(j_start, ncols).noalias() = this->_weight[k] * input.middleCols(i_start + offset, ncols);
      else
        output.middleCols(j_start, ncols).noalias() += this->_weight[k] * input.middleCols(i_start + offset, ncols);
    }
  }
  if (this->_bias.size() > 0)
  {
    output.middleCols(j_start, ncols).colwise() += this->_bias;
  }
}

long Conv1D::get_in_channels() const
{
  if (this->_is_depthwise)
    return this->_channels;
  return this->_weight.size() > 0 ? this->_weight[0].cols() : 0;
}

long Conv1D::get_out_channels() const
{
  if (this->_is_depthwise)
    return this->_channels;
  return this->_weight.size() > 0 ? this->_weight[0].rows() : 0;
}

long Conv1D::get_kernel_size() const
{
  if (this->_is_depthwise)
    return this->_depthwise_weight.size();
  return this->_weight.size();
}

long Conv1D::get_num_weights() const
{
  long num_weights = this->_bias.size();
  if (this->_is_depthwise)
  {
    // Depthwise: one weight per channel per kernel tap
    num_weights += this->_channels * this->_depthwise_weight.size();
  }
  else if (this->_weight.size() > 0)
  {
    const long out_channels = this->_weight[0].rows();
    const long in_channels = this->_weight[0].cols();
    // For grouped convolutions, the number of weights is reduced by numGroups
    num_weights += (out_channels * in_channels * this->_weight.size()) / this->_num_groups;
  }
  return num_weights;
}

size_t Conv1D::copy_weights_to_buffer(float* buffer, size_t buffer_size) const
{
  size_t total_size = (size_t)get_num_weights();
  if (total_size > buffer_size)
    return 0;

  size_t offset = 0;

  // Copy weights
  if (this->_is_depthwise)
  {
    for (const auto& w : this->_depthwise_weight)
    {
      std::memcpy(buffer + offset, w.data(), w.size() * sizeof(float));
      offset += w.size();
    }
  }
  else
  {
    for (const auto& w : this->_weight)
    {
      std::memcpy(buffer + offset, w.data(), w.size() * sizeof(float));
      offset += w.size();
    }
  }

  // Copy bias
  if (this->_bias.size() > 0)
  {
    std::memcpy(buffer + offset, this->_bias.data(), this->_bias.size() * sizeof(float));
    offset += this->_bias.size();
  }

  return offset;
}
} // namespace nam
