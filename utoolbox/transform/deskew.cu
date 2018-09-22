__kernel void shear(
    read_only image2d_array_t src,
    const float factor,
    const int iw,
    const int nu, const int nv, // output size
    __global unsigned short *dst
) {
    const int iu = get_global_id(1);
    const int iv = get_global_id(0);
    if ((iu >= nu) || (iv >= nv)) {
        return;
    }

    const sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

    float ix = (iu - factor*iv) + 0.5f;
    float iy = iv + 0.5f;
    float iz = iw;

    const int i = iv*nu + iu;
    dst[i] =
        (unsigned short)read_imagef(src, sampler, (float4)(ix, iy, iz, 1.0f)).x;
}

__kernel void shear_and_rotate(
    read_only image2d_array_t src,
    const float vsin, const float vcos,
    const float factor,
    const int iw,
    const int nu, const int nv, // output size
    const int ov,
    __global unsigned short *dst
) {
    const int iu = get_global_id(1);
    const int iv = get_global_id(0);
    if ((iu >= nu) || (iv >= nv)) {
        return;
    }

    const sampler_t sampler =
        CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;

    // center of the image
    const float uc = nu/2.0f;
    const float vc = nv/2.0f;
    //NOTE int or float for intermediate coordinate?
    //const float iu0 = (iu-uc)*vcos - (iv+ov-vc)*vsin + uc;
    //const float iv0 = (iu-uc)*vsin + (iv+ov-vc)*vcos + vc;
    const int iu0 = (int)((iu-uc)*vcos - (iv+ov-vc)*vsin + uc);
    const int iv0 = (int)((iu-uc)*vsin + (iv+ov-vc)*vcos + vc);

    // skewed
    float ix = (iu0 - factor*iv0) + 0.5f;
    float iy = (iv0) + 0.5f;
    float iz = iw;

    const int i = iv*nu + iu;
    dst[i] =
        (unsigned short)read_imagef(src, sampler, (float4)(ix, iy, iz, 1.0f)).x;
}
