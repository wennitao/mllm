#include <map> 
#include <string> 
namespace mllm::opencl { 
extern const char* sub;
extern const char* silu;
extern const char* embedding;
extern const char* softmax;
extern const char* fill;
extern const char* matmul;
extern const char* add;
extern const char* flash_attention;
extern const char* rmsnorm;
extern const char* div;
extern const char* rope;
extern const char* cast_type;
extern const char* matmul_transb_bias;
extern const char* causal_mask;
extern const char* mul;
extern const char* transpose;
const std::map<std::string, const char*> OpenCLProgramMap = 
 { 
  { "sub", sub },
  { "silu", silu },
  { "embedding", embedding },
  { "softmax", softmax },
  { "fill", fill },
  { "matmul", matmul },
  { "add", add },
  { "flash_attention", flash_attention },
  { "rmsnorm", rmsnorm },
  { "div", div },
  { "rope", rope },
  { "cast_type", cast_type },
  { "matmul_transb_bias", matmul_transb_bias },
  { "causal_mask", causal_mask },
  { "mul", mul },
  { "transpose", transpose },
};
} // namespace mllm::opencl
const std::map<std::string, std::string> OpenCLProgramMd5Map = 
 { 
  { "sub", "e85bb10e1ddad3d0e38f09e6b9fec7c3" },
  { "silu", "c9c2197b4b426d12cc652296738c24e1" },
  { "embedding", "2f53599ee23db57480589c629c0d2069" },
  { "softmax", "ed7b176870d74194ae683712b71bc2fc" },
  { "fill", "0378c49b52d12aee7d7fb0d8495945c7" },
  { "matmul", "3575e473588e7034a12913e30f861ee0" },
  { "add", "a1ed4a6207f790f5dc436cf841744cf3" },
  { "flash_attention", "3727637dc183e8320876ee02aea30ee9" },
  { "rmsnorm", "c80a3d46b5760cddc6cca0205a7b6e85" },
  { "div", "5936825bc33ffe218c11ade20fd68203" },
  { "rope", "a0fb1d3c9e5f3cfbb6384471d8a40591" },
  { "cast_type", "995c97bd81887ea6082f8c1839d738c2" },
  { "matmul_transb_bias", "9fd958568df8525ae1b6d620ff0bf5e5" },
  { "causal_mask", "91d58f2fc38e1011d07dee99abefe12b" },
  { "mul", "1241c7141443ad9e18dadd14e150516a" },
  { "transpose", "87caf150f16b16711c85f250ef3b4078" },
};
