#include "a_opencl_source_map.hpp" 
namespace mllm::opencl { 
const char* cast_type = 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"// Element-wise dtype casts. One work-item per logical element.\n"
"__kernel void cast_f32_to_f16(__global const float *input,__global half *output) {\n"
" const int i=get_global_id(0);\n"
" output[i]=(half)input[i];\n"
"}\n"
"__kernel void cast_f16_to_f32(__global const half *input,__global float *output) {\n"
" const int i=get_global_id(0);\n"
" output[i]=(float)input[i];\n"
"}\n"
;
} // namespace mllm::opencl
