__constant__ float px_shift;
__constant__ float vsin, vcos;

texture<float, cudaTextureType3D, cudaReadModeElementType> ref_vol;

__global__
void deskew_kernel(
    {{ dst_type }} *dst,
    const int iv,
    const int nu, const int nw, // output size
    const int nx, const int nz // input size
) {
    int iu = blockIdx.x*blockDim.x + threadIdx.x;
    int iw = blockIdx.y*blockDim.y + threadIdx.y;
    if ((iu >= nu) || (iw >= nw)) {
        return;
    }

    const float cu = nu/2, cw = nw/2;
    const float cx = nx/2, cz = nz/2;

    // rotate
    float x = (iu-cu)*vcos - (iw-cw)*vsin + cx;
    float z = (iu-cu)*vsin + (iw-cw)*vcos + cz;

    // round off to avoid over using interpolation
    x = roundf(x); z = roundf(z);

    // shear
    x -= px_shift*(z-cz);

    float ix = x + .5f;
    float iy = iv + .5f;
    float iz = z + .5f;

    const int i = iw*nu + iu;
    dst[i] = ({{ dst_type }})tex3D(ref_vol, ix, iy, iz);
}
