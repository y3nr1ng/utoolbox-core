extern "C" {

#define TILE_WIDTH      ${tile_width}
#define KERNEL_WIDTH    3
#define KERNEL_RADIUS   1
#define CACHE_WIDTH     (TILE_WIDTH+KERNEL_WIDTH-1)  

__global__
void modified_laplacian_kernel(
    float *dst,
    const float *src,
    const int nx, const int ny
) {
    __shared__ 
    float cache[CACHE_WIDTH*CACHE_WIDTH];

    // cache linear index
    int ic = TILE_WIDTH*threadIdx.y + threadIdx.x;
    // load padded data to cache
    for (int i = ic; i < CACHE_WIDTH*CACHE_WIDTH; i += TILE_WIDTH*TILE_WIDTH) {
        // cache index
        int cx = i%CACHE_WIDTH;
        int cy = i/CACHE_WIDTH;

        // padded global index
        int pgx = TILE_WIDTH*blockIdx.x + cx - KERNEL_RADIUS;
        int pgy = TILE_WIDTH*blockIdx.y + cy - KERNEL_RADIUS;
        
        // mirror padding
        if (pgx < 0) {
            pgx = -pgx;
        } else if (pgx > (nx-1)) {
            pgx = 2*(nx-1)-pgx;
        }
        if (pgy < 0) {
            pgy = -pgy;
        } else if(pgy > (ny-1)) {
            pgy = 2*(ny-1)-pgy;
        }

        cache[i] = src[nx*pgy+pgx];
    }

    __syncthreads();

    // global index
    int gx = TILE_WIDTH*blockIdx.x + threadIdx.x;
    int gy = TILE_WIDTH*blockIdx.y + threadIdx.y;
    if ((gx >= nx) || (gy >= ny)) {
        return;
    }

    // local index
    int lx = threadIdx.x+KERNEL_RADIUS;
    int ly = threadIdx.y+KERNEL_RADIUS;

    // data grid
    //   x u x 
    //   l c r
    //   x d x
    float pc = cache[CACHE_WIDTH*ly + lx];
    float pu = cache[CACHE_WIDTH*(ly+1) + lx];
    float pd = cache[CACHE_WIDTH*(ly-1) + lx];
    float pr = cache[CACHE_WIDTH*ly + (lx+1)];
    float pl = cache[CACHE_WIDTH*ly + (lx-1)]; 
    
    // global linear index
    dst[nx*gy+gx] = abs(2*pc-pu-pd) + abs(2*pc-pl-pr);
}

__global__
void sml_kernel(
    float *dst,
    const float *src,
    const int nx, const int ny,
    const float T
) {
    __shared__ 
    float cache[CACHE_WIDTH*CACHE_WIDTH];

    // cache linear index
    int ic = TILE_WIDTH*threadIdx.y + threadIdx.x;
    // load padded data to cache
    for (int i = ic; i < CACHE_WIDTH*CACHE_WIDTH; i += TILE_WIDTH*TILE_WIDTH) {
        // cache index
        int cx = i%CACHE_WIDTH;
        int cy = i/CACHE_WIDTH;

        // padded global index
        int pgx = TILE_WIDTH*blockIdx.x + cx - KERNEL_RADIUS;
        int pgy = TILE_WIDTH*blockIdx.y + cy - KERNEL_RADIUS;
        
        // clamping
        if ((pgx < 0) || (pgx >= nx) || (pgy < 0) || (pgy >= ny)) {
            cache[i] = 0.f;
        } else {
            cache[i] = src[nx*pgy+pgx];
        }
    }

    __syncthreads();

    // global index
    int gx = TILE_WIDTH*blockIdx.x + threadIdx.x;
    int gy = TILE_WIDTH*blockIdx.y + threadIdx.y;
    if ((gx >= nx) || (gy >= ny)) {
        return;
    }

    // local index
    int lx = threadIdx.x+KERNEL_RADIUS;
    int ly = threadIdx.y+KERNEL_RADIUS;

    // data grid
    //   x u x 
    //   l c r
    //   x d x
    float sum = 0;
    for (int ky=ly-KERNEL_RADIUS; ky<=ly+KERNEL_RADIUS; ky++) {
        for (int kx=lx-KERNEL_RADIUS; kx<=lx+KERNEL_RADIUS; kx++) {
            float v = cache[CACHE_WIDTH*ky + kx];
            if (v >= T) {
                sum += v;
            }
        }
    }
    dst[nx*gy+gx] = sum;
}

__global__
void keep_max_kernel(
    int *M, float *V,
    const float *S,
    const int nelem,
    const int level
) {
    // global linear index
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }

    if (abs(S[i]) > V[i]) {
        M[i] = level;
        V[i] = S[i];
    }
}

}