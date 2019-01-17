#include <pycuda-helpers.hpp>

texture<float, cudaTextureType2D, cudaReadModeElementType> rot2_tex;

__global__
void rot2_kernel(
    float *dst,
    const float theta,
    const unsigned int nu, const unsigned int nv, // output size
    const float rx, const float ry
) {
    unsigned int iu = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int iv = blockIdx.y*blockDim.y+threadIdx.y;
    if ((iu >= nu) || (iv >= nv)) {
        return;
    }

    // normalized texture coordinate
    float u = iu/(float)nu - .5f;
    float v = iv/(float)nv - .5f;

    // reverse rotate
    float x =  u*cosf(theta) + v*sinf(theta) + .5f;
    float y = -u*sinf(theta) + v*cosf(theta) + .5f;

    // rescale
    x *= rx; y *= ry;

    // write back
    const unsigned int i = iv*nv + iu;
    dst[i] = tex2D(rot2_tex, x, y);
}