<<<<<<< Updated upstream
<<<<<<< Updated upstream

texture<float, cudaTextureType2DLayered, cudaReadModeElementType> shear_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> rotate_tex;

=======
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
__constant__ float px_shift;
__constant__ float vsin, vcos;

__global__
void shear_kernel(
    float *dst,
    const float shift,                            // unit shifts
    const unsigned int nu, const unsigned int nv, // output size
    const float ratio,                            // scale 
    const unsigned int nx, const unsigned int ny, // input size
    const unsigned int nz                         // layers
) {
    unsigned int iu = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int iv = blockIdx.y*blockDim.y+threadIdx.y;
    if ((iu >= nu) || (iv >= nv)) {
        return;
    }

    // move origin to center
    float u0 = iu - nu/2.;
    float v0 = iv - nv/2.;

    // shear
    float x0 = u0 - shift*v0;
    float y0 = v0;

    // rescale
    y0 /= ratio;

    // move origin to corner
    float x = x0 + nx/2.;
    float y = y0 + ny/2.;

    // write back
    for (unsigned int iz = 0; iz < nz; iz++) {
        unsigned int i = iz*nv*nu + iv*nu + iu;
        dst[i] = tex2DLayered(shear_tex, x+.5f, y+.5f, iz);
    }
}

__global__
void rotate_kernel(
    float *dst,
    const float vsin, const float vcos,           // rotation matrix
    const unsigned int nu, const unsigned int nv, // output size
    const float sx, const float sy,               // scale 
    const unsigned int nx, const unsigned int ny  // input size
) {
    unsigned int iu = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int iv = blockIdx.y*blockDim.y+threadIdx.y;
    if ((iu >= nu) || (iv >= nv)) {
        return;
    }

    // move origin to center
    float u0 = iu - nu/2.;
    float v0 = iv - nv/2.;

    // rotate
    float x0 =  u0*vcos + v0*vsin;
    float y0 = -u0*vsin + v0*vcos;

    // rescale
    x0 /= sx; y0 /= sy;

    // move origin to corner
    float x = x0 + nx/2.;
    float y = y0 + ny/2.;

    const int i = iw*nu + iu;
    dst[i] = ({{ dst_type }})tex3D(ref_vol, ix, iy, iz);
}
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes
=======
=======
>>>>>>> Stashed changes
#define TILE_SIZE 16

__global__
void copy_kernel(
    float       *out,
    const float *in,
    int nx, int ny, int nz
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if ((ix >= nx) || (iy >= ny)) {
        return;
    }

    for (int iz = 0; iz < nz; iz++) {
        int i = ix + iy * nx + iz * nx*ny;
        out[i] = in[i];
    }
}

__global__
void transpose_xzy_outofplace( 
    float*       out,
    const float* in,
    int np0, int np1, int np2
) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

	int x, y, z,
	    ind_in,
	    ind_out;

	int lx = threadIdx.x,
	    ly = threadIdx.y,
	    bx = blockIdx.x,
	    by = blockIdx.y;

	x = lx + TILE_SIZE * bx;
	y = ly + TILE_SIZE * by;
	z = blockIdx.z;

	ind_in = x + (y + z * np1) * np0;  // x + np0 * y + np0 * np1 * z
	ind_out = x + (z + y * np2) * np0; // x + np0 * z + np0 * np2 * y

	if( x < np0 && y < np1 )
	{
		tile[lx][ly] = in[ind_in];
		if( y + 4 < np1 )
		{
			tile[lx][ly +  4] = in[ind_in +  4*np0];
			if( y + 8 < np1 )
			{
				tile[lx][ly +  8] = in[ind_in +  8*np0];
				if( y + 12 < np1 )
				{
					tile[lx][ly +  12] = in[ind_in +  12*np0];
				}
			}
		}
	}

	__syncthreads();

	if( x < np0 && y < np1 )
	{
		out[ind_out] = tile[lx][ly];
		if( y + 4 < np1 )
		{
			out[ind_out +  4*np0*np2] = tile[lx][ly + 4];
			if( y + 8 < np1 )
			{
				out[ind_out +  8*np0*np2] = tile[lx][ly + 8];
				if( y + 12 < np1 )
				{
					out[ind_out +  12*np0*np2] = tile[lx][ly + 12];
				}
			}
		}
	}
}
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
>>>>>>> Stashed changes
