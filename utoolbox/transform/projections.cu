extern "C" {

__global__
void z_proj_kernel(
    unsigned short *dst,
    unsigned short *src,
    const int nx, const int ny, const int nz
) {
    int ix = blockDim.x*blockIdx.x + threadIdx.x;
    int iy = blockDim.y*blockIdx.y + threadIdx.y;
    if ((ix >= nx) || (iy >= ny)) {
        return;
    }

    const int i = nx*iy + ix;
    const int stride = nx*ny;
    unsigned short value = 0;
    for (int iz = 0; iz < nz; iz++) {
        value = max(value, src[i + stride*iz]);
    }
    dst[i] = value;
}

__global__
void y_proj_kernel(
    unsigned short *dst,
    unsigned short *src,
    const int nx, const int ny, const int nz
) {
    int ix = blockDim.x*blockIdx.x + threadIdx.x;
    int iz = blockDim.y*blockIdx.y + threadIdx.y;
    if ((ix >= nx) || (iz >= nz)) {
        return;
    }

    const int i_in = (nx*ny)*iz + ix;
    const int i_out = nx*iz + ix;
    unsigned short value = 0; 
    for (int iy = 0; iy < ny; iy++) {
        value = max(value, src[i_in + nx*iy]);
    }
    dst[i_out] = value;
}

__global__
void x_proj_kernel(
    unsigned short *dst,
    unsigned short *src,
    const int nx, const int ny, const int nz
) {
    int iy = blockDim.x*blockIdx.x + threadIdx.x;
    int iz = blockDim.y*blockIdx.y + threadIdx.y;
    if ((iy >= ny) || (iz >= nz)) {
        return;
    }

    const int i_in = (nx*ny)*iz + nx*iy;
    const int i_out = ny*iz + iy;
    unsigned short value = 0; 
    for (int ix = 0; ix < nx; ix++) {
        value = max(value, src[i_in + ix]);
    }
    dst[i_out] = value;
}

}