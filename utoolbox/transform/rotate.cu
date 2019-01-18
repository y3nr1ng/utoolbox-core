texture<float, cudaTextureType2D, cudaReadModeElementType> rot2_tex;

__global__
void rot2_kernel(
    float *dst,
    const float theta,
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

    // rotate
    float x0 =  u0*cosf(theta) + v0*sinf(theta);
    float y0 = -u0*sinf(theta) + v0*cosf(theta);

    // rescale
    x0 /= sx; y0 /= sy;

    // move origin to corner
    float x = x0 + nx/2.;
    float y = y0 + ny/2.;

    // write back
    const unsigned int i = iv*nu + iu;
    dst[i] = tex2D(rot2_tex, x+.5f, y+.5f);
}