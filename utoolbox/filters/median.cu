extern "C" {

#define TILE_WIDTH ${tile_width}
#define KERNEL_RADIUS ${kernel_radius}
#define KERNEL_WIDTH (2 * KERNEL_RADIUS + 1)
#define CACHE_WIDTH (TILE_WIDTH + KERNEL_WIDTH - 1)

__global__ void median_3d_kernel(
  float *dst, const float *src, 
  const int nx, const int ny, const int nz) 
{
  __shared__ float cache[CACHE_WIDTH * CACHE_WIDTH * CACHE_WIDTH];

  // cache linear index
  int ic = TILE_WIDTH * (TILE_WIDTH * threadIdx.z + threadIdx.y) + threadIdx.x;
  // load padded data to cache
  for (int i = ic; i < CACHE_WIDTH * CACHE_WIDTH * CACHE_WIDTH;
       i += TILE_WIDTH * TILE_WIDTH * TILE_WIDTH) {
    // cache index
    int cx = i % CACHE_WIDTH;
    int cy = (i / CACHE_WIDTH) % CACHE_WIDTH;
    int cz = (i / CACHE_WIDTH) / CACHE_WIDTH;

    // padded global index
    int pgx = TILE_WIDTH * blockIdx.x + cx - KERNEL_RADIUS;
    int pgy = TILE_WIDTH * blockIdx.y + cy - KERNEL_RADIUS;
    int pgz = TILE_WIDTH * blockIdx.z + cz - KERNEL_RADIUS;

    // mirror padding
    if ((pgx < 0) || (pgx >= nx) || (pgy < 0) || (pgy >= ny) || (pgz < 0) ||
        (pgz >= nz)) {
      cache[i] = 0.f;
    } else {
      cache[i] = src[nx * ny * pgz + nx * pgy + pgx];
    }
  }

  __syncthreads();

  // global index
  int gx = TILE_WIDTH * blockIdx.x + threadIdx.x;
  int gy = TILE_WIDTH * blockIdx.y + threadIdx.y;
  int gz = TILE_WIDTH * blockIdx.z + threadIdx.z;
  if ((gx >= nx) || (gy >= ny) || (gz >= nz)) {
    return;
  }

  // local index
  int lx = threadIdx.x + KERNEL_RADIUS;
  int ly = threadIdx.y + KERNEL_RADIUS;
  int lz = threadIdx.z + KERNEL_RADIUS;

  // data grid
  //   x x x   x u x   x x x
  //   x t x   l c r   x b x
  //   x x x   x d x   x x x
  float v[KERNEL_WIDTH * KERNEL_WIDTH * KERNEL_WIDTH] = {0};
  for (int k = -KERNEL_RADIUS, t = 0; k <= KERNEL_RADIUS; k++) {
    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
      for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++, t++) {
        v[t] =
            cache[CACHE_WIDTH * (CACHE_WIDTH * (lz + k) + (ly + j)) + (lx + i)];
      }
    }
  }

  // bubble sort
  int nelem = KERNEL_WIDTH * KERNEL_WIDTH * KERNEL_WIDTH;
  int t_half = nelem / 2;

  for (int i = 0; i < t_half + 1; i++) {
    for (int j = i + 1; j < nelem; j++) {
      if (v[i] > v[j]) { // swap?
        float tmp = v[i];
        v[i] = v[j];
        v[j] = tmp;
      }
    }
  }

  // pick the middle one
  dst[nx * ny * gz + nx * gy + gx] = v[t_half];
}
}