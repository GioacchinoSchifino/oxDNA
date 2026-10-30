#include <cstring>
#include "OxDNACudaErrorCheck.h"
static cudaError_t pending = cudaSuccess;
extern "C" cudaError_t cudaGetLastError(void) {
    cudaError_t current = pending;
    pending = cudaSuccess;
    return current;
}
extern "C" const char* cudaGetErrorString(cudaError_t v) {
    return v == cudaSuccess ? "success" : "invalid value (mock)";
}
int main(int argc, char **argv) {
    if (argc != 2) return 3;
    if (std::strcmp(argv[1], "success") == 0) {
        pending = cudaSuccess;
        oxdna_cuda_check_last_error("mock success", __FILE__, __LINE__);
        return pending == cudaSuccess ? 0 : 4;
    }
    if (std::strcmp(argv[1], "failure") == 0) {
        pending = cudaErrorInvalidValue;
        oxdna_cuda_check_last_error("mock failure", __FILE__, __LINE__);
        return 5; // Must never reach: fail-fast expected.
    }
    return 6;
}
