#include <pycuda-helpers.hpp>

texture<float, cudaTextureType2D, cudaReadModeElementType> rot2_tex;

__global__
void rot2_kernel(
    float *dst,
    const float theta,
    const unsigned int nu, const unsigned int nv, // input size
    const float sx, const float sy,
    const unsigned int nx, const unsigned int ny  // output size
) {
    unsigned int iu = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int iv = blockIdx.y*blockDim.y+threadIdx.y;
    if ((iu >= nu) || (iv >= nv)) {
        return;
    }

    // move origin to center
    float u0 = (float)iu - (float)nu/2;
    float v0 = (float)iv - (float)nv/2;

    // rescale
    u0 /= sx; v0 /= sy;

    // rotate
    float x0 =  u0*cosf(theta) + v0*sinf(theta);
    float y0 = -u0*sinf(theta) + v0*cosf(theta);

    // move origin to corner
    float x = x0 + (float)nx/2;
    float y = y0 + (float)ny/2;

    // write back
    const unsigned int i = iv*nv + iu;
    dst[i] = (float)42; //tex2D(rot2_tex, x+.5f, y+.5f);
}