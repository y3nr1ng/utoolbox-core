
extern "C" {

__global__
void shear_kernel(
    float* dst,
    cudaTextureObject_t src,
    int nx, int ny, int nu,
    float shift
) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if ((ix >= nx) || (iy >= ny)) {
        return;
    }

    float v = iy;
    float u = ix - shift * (v-ny/2) - (nx-nu)/2; // integer division, floor
    
    dst[iy * nx + ix] = tex2D<float>(src, u, v);
}

__global__
void rotate_kernel(
    float *dst,
    cudaTextureObject_t src,
    int nx, int ny
) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if ((ix >= nx) || (iy >= ny)) {
        return;
    }
}

}