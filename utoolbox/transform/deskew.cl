__kernel void shear(
    read_only image2d_array_t src,
    const float factor,
    const int iw,
    const int nu, const int nv, // output size
    __global unsigned short *dst
) {
    const int iu = get_global_id(0);
    const int iv = get_global_id(1);
    if ((iu >= nu) || (iv >= nv)) {
        return;
    }

    const sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

    float ix = iu + 0.5f; //(iu - factor*iv) + 0.5f;
    float iy = iv + 0.5f;
    float iz = iw;

    const int i = iv*nu + iu;
    dst[i] =
        (unsigned short)read_imagef(src, sampler, (float4)(ix, iy, iz, 1.0f)).x;
}
