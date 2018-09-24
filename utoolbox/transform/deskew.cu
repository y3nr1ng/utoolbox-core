__constant__ float px_shift;
__constant__ float vsin;
__constant__ float vcos;
__constant__ int iw_ori;

texture<float, cudaTextureType3D, cudaReadModeElementType> ref_vol;

__global__
void deskew_kernel(
    float *dst,
    const int iv,
    const int nu, const int nw, // output size
    const int nx, const int nz // input size
) {
    int iu = blockIdx.x*blockDim.x + threadIdx.x;
    int iw = blockIdx.y*blockDim.y + threadIdx.y;
    if ((iu >= nu) || (iw >= nw)) {
        return;
    }

    // center of the image
    const float cx = nx/2.f;
    const float cz = nz/2.f;
    //NOTE int for intermediate coordinates to avoid premature interpolations
    float ix0 = roundf((iu-cx)*vcos - (iw-cz+iw_ori)*vsin + cx);
    float iz0 = roundf((iu-cx)*vsin + (iw-cz+iw_ori)*vcos + cz);

    // skewed
    float ix = (ix0 - px_shift*iz0) + 0.5f;
    float iy = iv + 0.5f;
    float iz = iz0 + 0.5f;

    const int i = iw*nu + iu;
    dst[i] = tex3D(ref_vol, ix, iy, iz);
}
