extern "C" {

#define TILE_WIDTH      ${tile_width}
#define KERNEL_WIDTH    3
#define KERNEL_RADIUS   1
#define CACHE_WIDTH     (TILE_WIDTH+KERNEL_WIDTH-1)  

__device__
float quadric(float norm, float thre) {
    return 1.f / (1.f + norm*norm / (thre*thre));
}

__device__
float exponential(float norm, float thre) {
    return exp(-norm*norm / (thre*thre));
}

__global__
void perona_malik_2d_kernel(
    float *dst,
    const float *src,
    const float thre, const float lambda,
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
    float pc = cache[CACHE_WIDTH*ly + lx];
    float pu = cache[CACHE_WIDTH*(ly+1) + lx];
    float pd = cache[CACHE_WIDTH*(ly-1) + lx];
    float pr = cache[CACHE_WIDTH*ly + (lx+1)];
    float pl = cache[CACHE_WIDTH*ly + (lx-1)]; 

    // delta
    float du = pu-pc;
    float dd = pd-pc;
    float dr = pr-pc;
    float dl = pl-pc;

    // apply function
    // TODO use function pointer, assume quadric for now
    float cu = exponential(abs(du), thre);
    float cd = exponential(abs(dd), thre);
    float cr = exponential(abs(dr), thre);
    float cl = exponential(abs(dl), thre);

    // global linear index
    dst[nx*gy+gx] = pc + lambda * (cu*du + cd*dd + cr*dr + cl*dl);
}

}