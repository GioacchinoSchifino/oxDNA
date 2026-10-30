#ifndef OXDNA_MOCK_CUDA_RUNTIME_API_H_
#define OXDNA_MOCK_CUDA_RUNTIME_API_H_
// Test-only mock, never used by production builds.
typedef enum { cudaSuccess=0, cudaErrorInvalidValue=11 } cudaError_t;
extern "C" cudaError_t cudaGetLastError(void);
extern "C" const char* cudaGetErrorString(cudaError_t);
#endif
