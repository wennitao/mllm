#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Element-wise dtype casts. One work-item per logical element.
__kernel void cast_f32_to_f16(__global const float *input, __global half *output) {
  const int i = get_global_id(0);
  output[i] = (half)input[i];
}

__kernel void cast_f16_to_f32(__global const half *input, __global float *output) {
  const int i = get_global_id(0);
  output[i] = (float)input[i];
}
