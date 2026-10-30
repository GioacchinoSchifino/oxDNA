#ifndef OXDNA_CUDA_ERROR_CHECK_H_
#define OXDNA_CUDA_ERROR_CHECK_H_

// Minimal independent replacement for the historical CUDA samples error helper.
// Checks and clears the pending CUDA runtime error via cudaGetLastError().
// This is host-side diagnostics; it does not modify simulation interactions.
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>

inline void oxdna_cuda_check_last_error(const char *message, const char *file, int line) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s(%d) : CUT_CHECK_ERROR() CUDA error : %s : (%d) %s.\n",
                     file, line, message, static_cast<int>(error), cudaGetErrorString(error));
        std::fflush(stderr);
        std::exit(EXIT_FAILURE);
    }
}

#endif // OXDNA_CUDA_ERROR_CHECK_H_
