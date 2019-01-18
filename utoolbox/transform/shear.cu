texture<float, cudaTextureType2D, cudaReadModeElementType> shear2_tex;

__global__
void shear2_kernel(
    float *dst,
    const float dx, const float dy,               // unit shifts
    const unsigned int nu, const unsigned int nv, // output size
    const float sx, const float sy,               // scale 
    const unsigned int nx, const unsigned int ny  // input size
) {
    unsigned int iu = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int iv = blockIdx.y*blockDim.y+threadIdx.y;
    if ((iu >= nu) || (iv >= nv)) {
        return;
    }

    // move origin to center
    float u0 = iu - nu/2.;
    float v0 = iv - nv/2.;

    // shear
    float x0 = u0 - dx*v0;
    float y0 = v0 - dy*u0;

    // rescale
    x0 /= sx; y0 /= sy;

    // move origin to corner
    float x = x0 + nx/2.;
    float y = y0 + ny/2.;

    // write back
    const unsigned int i = iv*nu + iu;
    dst[i] = tex2D(shear2_tex, x+.5f, y+.5f);
}