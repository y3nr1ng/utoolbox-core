#include <cstdint>

extern "C" {

__global__ 
void ushort_to_float(
    float *dst, const uint16_t *src, const int nelem
) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    for (; i < nelem; i += blockDim.x) {
        dst[i] = (float)src[i]
    }
}

__global__
void float_to_ushort(
    uint16_t *dst, const float *src, const int nelem
) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    for (; i < nelem; i += blockDim.x) {
        dst[i] = (uint16_t)src[i]
    }
}

}