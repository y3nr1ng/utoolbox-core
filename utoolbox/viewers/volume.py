from vispy.gloo import Texture3D, TextureEmulated3D, VertexBuffer, IndexBuffer
from vispy.visuals import Visual
from vispy.visuals.shaders import Function
from vispy.color import get_colormap
from vispy.scene.visuals import create_visual_node

import numpy as np

from .multi_volume_shaders import get_shaders
from .callback_list import CallbackList

class VolumeRenderVisual(Visual):
    """
    This is a refactored version of MultiVolumeVisual class [1], which is
    derived from the original VolumeVisual class in vispy.visuals.volume

    Parameters
    ----------
    TBA

    Reference
    ---------
    [1] https://github.com/astrofrog/vispy-multivol
    """
    def __init__(self, volumes, clim=None, threshold=None,
                 relative_step_size=0.8, cmap='grays',
                 emulate_texture=False, n_volume_max=3):
        # We store the data and colormaps in a CallbackList which can warn us
        # when it is modified.
        self.volumes = CallbackList()
        self.volumes.on_size_change = self._update_all_volumes
        self.volumes.on_item_change = self._update_volume

        self._shape = None
        self._need_vertex_update = True

        # create OpenGL program
        vert_shader, frag_shader = get_shaders(n_volume_max)
        super(VolumeRenderVisual, self).__init__(vcode=vert_shader, fcode=frag_shader)

        # create gloo objects
        self._vertices = VertexBuffer()
        self._texcoord = VertexBuffer(
            np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ], dtype=np.float32))

        # Set up textures
        self.textures = []
        tex_cls = TextureEmulated3D if emulate_texture else Texture3D
        for i in range(n_volume_max):
            self.textures.append(tex_cls((10, 10, 10), interpolation='linear',
                                         wrapping='clamp_to_edge'))
            self.shared_program['u_volumetex{0}'.format(i)] = self.textures[i]
            self.shared_program.frag['cmap{0:d}'.format(i)] = Function(get_colormap(cmap).glsl_map)

        self.shared_program['a_position'] = self._vertices
        self.shared_program['a_texcoord'] = self._texcoord
        self._draw_mode = 'triangle_strip'
        self._index_buffer = IndexBuffer()

        self.shared_program.frag['sampler_type'] = self.textures[0].glsl_sampler_type
        self.shared_program.frag['sample'] = self.textures[0].glsl_sample

        # Only show back faces of cuboid. This is required because if we are
        # inside the volume, then the front faces are outside of the clipping
        # box and will not be drawn.
        self.set_gl_state('translucent', cull_face=False)

        self.relative_step_size = relative_step_size
        self.freeze()

        # Add supplied volumes
        self.volumes.extend(volumes)

    def _update_all_volumes(self, volumes):
        """Update the number of simultaneous textures.
        Parameters
        ----------
        n_textures : int
            The number of textures to use
        """
        if len(self.volumes) > len(self.textures):
            raise ValueError("Number of volumes ({0}) exceeds number of textures ({1})".format(len(self.volumes), len(self.textures)))
        for index in range(len(self.volumes)):
            self._update_volume(volumes, index)

    def _update_volume(self, volumes, index):
        data, clim, cmap = volumes[index]

        cmap = get_colormap(cmap)

        if clim is None:
            clim = data.min(), data.max()

        data = data.astype(np.float32)
        if clim[1] == clim[0]:
            if clim[0] != 0.:
                data *= 1.0 / clim[0]
        else:
            data -= clim[0]
            data /= clim[1] - clim[0]

        self.shared_program['u_volumetex{0:d}'.format(index)].set_data(data)
        self.shared_program.frag['cmap{0:d}'.format(index)] = Function(cmap.glsl_map)

        print(self.shared_program.frag)

        if self._shape is None:
            self.shared_program['u_shape'] = data.shape[::-1]
            self._shape = data.shape
        elif data.shape != self._shape:
            raise ValueError("Shape of arrays should be {0} instead of {1}".format(self._shape, data.shape))

        self.shared_program['u_n_tex'] = len(self.volumes)


    @property
    def relative_step_size(self):
        """Step size during ray casting.

        Larger values yield higher performance at reduced quality. If set > 2.0
        the ray skips entire voxels. Recommended values are between 0.5 and 1.5.
        The amount of quality degredation depends on the render method.
        """
        return self._relative_step_size

    @relative_step_size.setter
    def relative_step_size(self, value):
        value = float(value)
        if value < 0.1:
            raise ValueError('Step size cannot be smaller than 0.1')
        self._relative_step_size = value
        self.shared_program['u_relative_step_size'] = value

    def _create_vertex_data(self):
        """Create and set positions and texture coords from the given shape.

        Having 6 faces with 2 triangles each, this results in 6*2*3=36 verticies
        in total.
        """
        shape = self._shape

        # get corner coordinates
        #NOTE The -0.5 offset is to center pixels/voxels.
        x0, x1 = -0.5, shape[2] - 0.5
        y0, y1 = -0.5, shape[1] - 0.5
        z0, z1 = -0.5, shape[0] - 0.5

        pos = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y1, z0],
            [x1, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
        ], dtype=np.float32)

        """
          6-------7
         /|      /|
        4-------5 |
        | |     | |
        | 2-----|-3
        |/      |/
        0-------1
        """
        # Order is chosen such that normals face outward; front faces will be
        # culled.
        indices = np.array([2, 6, 0, 4, 5, 6, 7, 2, 3, 0, 1, 5, 3, 7],
                           dtype=np.uint32)

        self._vertices.set_data(pos)
        self._index_buffer.set_data(indices)

    def _compute_bounds(self, axis, view):
        return 0, self._shape[axis]

    def _prepare_transforms(self, view):
        trs = view.transforms
        view.view_program.vert['transform'] = trs.get_transform()

        view_tr_f = trs.get_transform('visual', 'document')
        view_tr_i = view_tr_f.inverse
        view.view_program.vert['viewtransformf'] = view_tr_f
        view.view_program.vert['viewtransformi'] = view_tr_i

    def _prepare_draw(self, view):
        if self._need_vertex_update:
            self._create_vertex_data()
            self._need_vertex_update = False

VolumeRender = create_visual_node(VolumeRenderVisual)
