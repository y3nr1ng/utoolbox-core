import logging
from math import radians, sin, cos, ceil, hypot
import os

from jinja2 import Template
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.tools import dtype_to_ctype
import tqdm

from utoolbox.container import AttrDict, Resolution

logger = logging.getLogger(__name__)

class DeskewTransform(object):
    """
    Perform standard deskew operation on sample scanned data.

    Note
    ----
    Shearing **always** happened along the X-axis.
    """
    def __init__(self, resolution, angle, rotate=True):
        """
        Parameters
        ----------
        resolution : utoolbox.container.Resolution
            Resolution object that specifies the lateral and axial resolution.
        angle : float
            Angle between coverslip and detection objective.
        rotate : bool
            Rotate the result to perpendicular to coverslip, default to True.
        """
        if type(resolution) is not Resolution:
            try:
                resolution = Resolution._make(resolution)
            except TypeError:
                raise TypeError("invalid resolution input")
        angle = radians(angle)

        self._in_res = resolution
        self._angle = angle
        self._rotate = rotate

        self._px_shift = resolution.dz * cos(angle) / resolution.dxy
        self._out_res = Resolution(resolution.dxy, resolution.dz * sin(angle))
        logger.debug(
            "dyx={:.3f}um, dz={:.3f}um, angle={:.2f}rad, shift={:.2f}px".format(
                resolution.dxy, resolution.dz, angle, self.px_shift
            )
        )

        self._in_shape, self._out_shape = None, None
        self._iw_origin = 0
        self._upload_texture = None

    def __call__(self, data, copy=False):
        """
        Parameters
        ----------
        data : np.ndarray
            SPIM data.
        copy : bool
            Determine whether output should be a duplicate of internal buffer,
            default to False.
        """
        if (data.shape != self._in_shape) or (data.dtype != self._dtype):
            logger.info("resizing workspace")
            if self._in_shape is not None:
                self.destroy_workspace()
            self._in_shape, self._dtype = data.shape, data.dtype

            try:
                self._load_kernel()
            except cuda.CompileError:
                logger.error("compile error")
                raise

            cuda.memcpy_htod(self._d_px_shift, np.float32(self.px_shift))

            self.create_workspace()

        # upload reference volume
        ref_vol = self._upload_ref_vol(data.astype(np.float32))

        # determine grid & block size
        n_blocks = (32, 32, 1)
        nw, nv, nu = self._out_shape
        n_grids = (ceil(float(nu)/n_blocks[0]), ceil(float(nw)/n_blocks[1]), 1)

        nz, _, nx = self._in_shape
        dxy, dz = self._in_res
        duv, dw = self._out_res

        # execute
        for iv in tqdm.trange(nv):
            self._kernel.prepared_call(
                n_grids, n_blocks,
                self._d_slice.gpudata,
                np.int32(iv),
                np.int32(nu), np.int32(nw),
                np.int32(nx), np.int32(nz)
            )
            h_slice = self._d_slice.get()
            self._h_vol[:, iv, :] = h_slice

        # free resource
        ref_vol.free()

        result = self._h_vol if not copy else self._h_vol.copy()
        return result

    @property
    def angle(self):
        return self._angle

    @property
    def out_res(self):
        return self._out_res

    @property
    def px_shift(self):
        return self._px_shift

    @property
    def rotate(self):
        return self._rotate

    def create_workspace(self):
        nz, ny, nx = self._in_shape # [px]
        total_shift = self.px_shift * (nz-1) # [px]

        if self.rotate:
            h = hypot(total_shift, nz) # [px]
            vsin, vcos = nz/h, total_shift/h # from [px]

            nw, nv, nu = ceil(nx*vsin), ny, ceil(h + nx*vcos)

            # update output resolution
            dxy, dz = self._in_res # [um]
            self._out_res = Resolution(
                hypot(total_shift*dxy, nz*dz) / nu,
                nx*dxy / nw
            )
            #TODO need to findout whether resample is required
        else:
            vsin, vcos = 0., 1.
            nw, nv, nu = nz, ny, ceil(nx+total_shift)
        self._out_shape = (nw, nv, nu)
        logger.debug("duv={:.3f}um, dw={:.3f}um".format(self._out_res.dxy, self._out_res.dz))

        cuda.memcpy_htod(self._d_vsin, np.float32(vsin))
        cuda.memcpy_htod(self._d_vcos, np.float32(vcos))
        logger.debug("vsin={:.4f}, vcos={:.4f}".format(vsin, vcos))

        self._h_vol = np.empty(self._out_shape, dtype=self._dtype)
        self._d_slice = gpuarray.empty((nw, nu), dtype=self._dtype, order='C')
        logger.info("workspace allocated, {}, {}".format(self._out_shape, self._dtype))

    def destroy_workspace(self):
        pass

    def _load_kernel(self):
        path = os.path.join(os.path.dirname(__file__), "deskew.cu")
        with open(path, 'r') as fd:
            tpl = Template(fd.read())
            source = tpl.render(dst_type=dtype_to_ctype(self._dtype))
            module = SourceModule(source)

        self._kernel = module.get_function("deskew_kernel")

        self._d_px_shift, _ = module.get_global('px_shift')
        self._d_vsin, _ = module.get_global('vsin')
        self._d_vcos, _ = module.get_global('vcos')

        self._texture = module.get_texref("ref_vol")
        self._texture.set_address_mode(0, cuda.address_mode.BORDER)
        self._texture.set_address_mode(1, cuda.address_mode.BORDER)
        self._texture.set_address_mode(2, cuda.address_mode.BORDER)
        self._texture.set_filter_mode(cuda.filter_mode.LINEAR)

        self._kernel.prepare('Piiiii',texrefs=[self._texture])

    def _upload_ref_vol(self, data):
        """Upload the reference volume into texture memory."""
        assert data.dtype == np.float32, "np.float32 is required"
        ref_vol = cuda.np_to_array(data, 'C')
        self._texture.set_array(ref_vol)
        return ref_vol
