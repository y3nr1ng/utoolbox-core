__global__
void z_proj_kernel(
    {{ src_type }} *dst,
    {{ src_type }} *src,
    const int nx, const int ny, const int nz
) {
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if ((ix >= nx) || (iy >= ny)) {
        return;
    }

    const int i = iy*nx + ix;
    const int stride = nx*ny;
    for (int iz = 0; iz < nz; iz++) {
        dst[i] = max(dst[i], src[i + stride*iz]);
    }
}
