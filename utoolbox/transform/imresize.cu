extern "C" {

__global__
void imresize_bilinear_kernel(
    float *dst,
    const int nu, const int nv,
    const float *src,
    const int nx, const int ny
) { 
    int iu = blockIdx.x*blockDim.x + threadIdx.x;
    int iv = blockIdx.y*blockDim.y + threadIdx.y;

    if ((iu >= nu) || (iv >= nv)) {
        return;
    }

    // 0   t   (nu-1)
    // | --|-----|
    // 0   i   (nx-1)
    //
    // t/(nu-1) = i/(nx-1), i = t*(nx-1)/(nu-1)
    float x = iu * (nx-1) / (nu-1);
    float y = iv * (ny-1) / (nv-1);
    // TODO lienar interpolate coordinates around this position

    dst[nu*iv+iu] = 42.f;
} 


}
