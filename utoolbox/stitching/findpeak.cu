extern "C" {

    #define TILE_WIDTH      ${tile_width}
    #define KERNEL_WIDTH    3
    #define KERNEL_RADIUS   1
    #define CACHE_WIDTH     (TILE_WIDTH+KERNEL_WIDTH-1)  
    
    __global__
    void find_peak_3d_kernel(
        unsigned short *dst,
        const unsigned short *src,
        const unsigned short thre, 
        const int nx, const int ny, const int nz
    ) {
        __shared__
        unsigned short cache[CACHE_WIDTH*CACHE_WIDTH*CACHE_WIDTH];
    
        // cache linear index
        int ic = TILE_WIDTH*(TILE_WIDTH*threadIdx.z + threadIdx.y) + threadIdx.x;
        // load padded data to cache
        for (
            int i = ic; 
            i < CACHE_WIDTH*CACHE_WIDTH*CACHE_WIDTH; 
            i+= TILE_WIDTH*TILE_WIDTH*TILE_WIDTH
        ) {
            // cache index
            int cx = i%CACHE_WIDTH;
            int cy = (i/CACHE_WIDTH)%CACHE_WIDTH;
            int cz = (i/CACHE_WIDTH)/CACHE_WIDTH;
    
            // padded global index
            int pgx = TILE_WIDTH*blockIdx.x + cx - KERNEL_RADIUS;
            int pgy = TILE_WIDTH*blockIdx.y + cy - KERNEL_RADIUS;
            int pgz = TILE_WIDTH*blockIdx.z + cz - KERNEL_RADIUS;
    
            // mirror padding
            if (
                (pgx < 0) || (pgx >= nx) 
                || (pgy < 0) || (pgy >= ny) 
                || (pgz < 0) || (pgz >= nz)
            ) {
                cache[i] = 0;
            } else {
                unsigned short v = src[nx*ny*pgz + nx*pgy + pgx];
                cache[i] = (v < thre) ? 0 : v;
            }
        }
    
        __syncthreads();
    
        // global index
        int gx = TILE_WIDTH*blockIdx.x + threadIdx.x;
        int gy = TILE_WIDTH*blockIdx.y + threadIdx.y;
        int gz = TILE_WIDTH*blockIdx.z + threadIdx.z;
        if ((gx >= nx) || (gy >= ny) || (gz >= nz)) {
            return;
        }
    
        // local index
        int lx = threadIdx.x+KERNEL_RADIUS;
        int ly = threadIdx.y+KERNEL_RADIUS;
        int lz = threadIdx.z+KERNEL_RADIUS;
    
        // data grid
        //   x x x   x u x   x x x
        //   x t x   l c r   x b x
        //   x x x   x d x   x x x
        unsigned short pc = cache[CACHE_WIDTH*(CACHE_WIDTH*lz + ly) + lx];
        unsigned short pu = cache[CACHE_WIDTH*(CACHE_WIDTH*lz + (ly+1)) + lx];
        unsigned short pd = cache[CACHE_WIDTH*(CACHE_WIDTH*lz + (ly-1)) + lx];
        unsigned short pr = cache[CACHE_WIDTH*(CACHE_WIDTH*lz + ly) + (lx+1)];
        unsigned short pl = cache[CACHE_WIDTH*(CACHE_WIDTH*lz + ly) + (lx-1)]; 
        unsigned short pt = cache[CACHE_WIDTH*(CACHE_WIDTH*(lz+1) + ly) + lx];
        unsigned short pb = cache[CACHE_WIDTH*(CACHE_WIDTH*(lz-1) + ly) + lx];
        
        // global linear index
        if (pc > pu && pc > pd && pc > pr && pc > pl && pc > pt && pc > pb) {
            dst[nx*ny*gz+nx*gy+gx] = 1;
        } else {
            dst[nx*ny*gz+nx*gy+gx] = 0;
        }
    }
        
}