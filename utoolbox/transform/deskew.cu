texture<float, cudaTextureType2DLayered, cudaReadModeElementType> shear_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> rotate_tex;

__global__
void shear_kernel(
    float *dst,
    const float shift,                            // unit shifts
    const unsigned int nu, const unsigned int nv, // output size
    const float ratio,                            // scale 
    const unsigned int nx, const unsigned int ny, // input size
    const unsigned int nz                         // layers
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
    float x0 = u0 - shift*v0;
    float y0 = v0;

    // rescale
    y0 /= ratio;

    // move origin to corner
    float x = x0 + nx/2.;
    float y = y0 + ny/2.;

    // write back
    for (unsigned int iz = 0; iz < nz; iz++) {
        unsigned int i = iz*nv*nu + iv*nu + iu;
        dst[i] = tex2DLayered(shear_tex, x+.5f, y+.5f, iz);
    }
}

__global__
void rotate_kernel(
    float *dst,
    const float vsin, const float vcos,           // rotation matrix
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
    float x0 =  u0*vcos + v0*vsin;
    float y0 = -u0*vsin + v0*vcos;

    // rescale
    x0 /= sx; y0 /= sy;

    // move origin to corner
    float x = x0 + nx/2.;
    float y = y0 + ny/2.;

    // write back
    unsigned int i = iv*nu + iu;
    dst[i] = tex2D(rotate_tex, x+.5f, y+.5f);
}