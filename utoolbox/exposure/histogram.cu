extern "C"
{

__global__
void hist_atom_kernel(
    unsigned int *p_hist,   // partial histogram
    const unsigned short *in,
    const int n_bins, const float scale, 
    const int nx, const int ny, const int nz
) {
    extern __shared__ unsigned int smem[];

    // pixel coordinate
    int ix = blockDim.x*blockIdx.x + threadIdx.x;
    int iy = blockDim.y*blockIdx.y + threadIdx.y;

    // grid dimension 
    int ngx = blockDim.x*gridDim.x;
    int ngy = blockDim.y*gridDim.y;
    // linear block index in a grid
    int ig = gridDim.x*blockIdx.y + blockIdx.x;

    // total threads
    int nt = blockDim.x*blockDim.y;
    // linear thread index in a block 
    int it = blockDim.x*threadIdx.y + threadIdx.x;

    // clear shared memory
    for (int i = it; i < n_bins; i += nt) {
        smem[i] = 0;
    }
    __syncthreads();

    // update partial histogram in shared memory
    for (int iz = 0; iz < nz; iz++) {
        for (int y = iy; y < ny; y += ngy) {
            for (int x = ix; x < nx; x += ngx) {
                float p = (float)in[(nx*ny)*iz + nx*y + x];
                unsigned int ih = (unsigned int)(p/scale);
                atomicAdd(&smem[ih], 1);
            }
        }
    }
    __syncthreads();

    // move to current slice in global memory
    p_hist += n_bins*ig;
    // move shared memory to global memory
    for (int i = it; i < n_bins; i += nt) {
        p_hist[i] = smem[i];
    }
}

__global__
void hist_accu_kernel(
    unsigned int *hist,
    const unsigned int *p_hist,
    const int n_bins, const int n_partials
) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i > n_bins) {
        return;
    }

    unsigned int sum = 0;
    for (int ip = 0; ip < n_partials; ip++) {
        sum += p_hist[i + n_bins*ip];
    }
    hist[i] = sum;
}

}