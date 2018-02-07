"""Multi-volume visual node.

This file implements the MultiVolumeVisual class modified from [1] that can be
used to show multiple volumes simultaneously. It is the direct derivative from
the original VolumeVisual class in vispy.visuals.volume, where both of them are
released under a BSD license. Please see their respective LICENSE for more info.

Note
----
    https://github.com/astrofrog/vispy-multivol
"""
from itertools import count

from vispy.gloo import Texture3D, TextureEmulated3D, VertexBuffer, IndexBuffer
from vispy.visuals import Visual
from vispy.visuals.shaders import Function
from vispy.color import get_colormap
from vispy.scene.visuals import create_visual_node

import numpy as np

from utoolbox.utils.defaults import DefaultFormat

# Vertex shader
VERT_SHADER = """
attribute vec3 a_position;
// attribute vec3 a_texcoord;
uniform vec3 u_shape;

// varying vec3 v_texcoord;
varying vec3 v_position;
varying vec4 v_nearpos;
varying vec4 v_farpos;

void main() {
    // v_texcoord = a_texcoord;
    v_position = a_position;

    // Project local vertex coordinate to camera position. Then do a step
    // backward (in cam coords) and project back. Voila, we get our ray vector.
    vec4 pos_in_cam = $viewtransformf(vec4(v_position, 1));

    // intersection of ray and near clipping plane (z = -1 in clip coords)
    pos_in_cam.z = -pos_in_cam.w;
    v_nearpos = $viewtransformi(pos_in_cam);

    // intersection of ray and far clipping plane (z = +1 in clip coords)
    pos_in_cam.z = pos_in_cam.w;
    v_farpos = $viewtransformi(pos_in_cam);

    gl_Position = $transform(vec4(v_position, 1.0));
}
"""

# Fragment shader
FRAG_SHADER = """
// uniforms
uniform int u_n_tex;
%(texture_declaration)s
uniform vec3 u_shape;
uniform float u_threshold;
uniform float u_relative_step_size;

//varyings
// varying vec3 v_texcoord;
varying vec3 v_position;
varying vec4 v_nearpos;
varying vec4 v_farpos;

// uniforms for lighting. Hard coded until we figure out how to do lights
const vec4 u_ambient = vec4(0.2, 0.4, 0.2, 1.0);
const vec4 u_diffuse = vec4(0.8, 0.2, 0.2, 1.0);
const vec4 u_specular = vec4(1.0, 1.0, 1.0, 1.0);
const float u_shininess = 40.0;

//varying vec3 lightDirs[1];

// global holding view direction in local coordinates
vec3 view_ray;

float rand(vec2 co)
{{
    // Create a pseudo-random number between 0 and 1.
    return fract(sin(dot(co.xy ,vec2(12.9898, 78.233))) * 43758.5453);
}}

float colorToVal(vec4 color)
{{
    return color.g;
}}

vec4 fromTexture(int index, vec3 loc)
{{
%(from_texture)s
}}

vec4 fromColormap(int index, float val)
{{
%(from_colormap)s
}}

vec4 calculateColor(int index, vec4 betterColor, vec3 loc, vec3 step)
{{
    // Calculate color by incorporating lighting
    vec4 color1;
    vec4 color2;

    // View direction
    vec3 V = normalize(view_ray);

    // calculate normal vector from gradient
    vec3 N; // normal
    color1 = fromTexture( index, loc+vec3(-step[0],0.0,0.0) );
    color2 = fromTexture( index, loc+vec3(step[0],0.0,0.0) );
    N[0] = colorToVal(color1) - colorToVal(color2);
    betterColor = max(max(color1, color2),betterColor);
    color1 = fromTexture( index, loc+vec3(0.0,-step[1],0.0) );
    color2 = fromTexture( index, loc+vec3(0.0,step[1],0.0) );
    N[1] = colorToVal(color1) - colorToVal(color2);
    betterColor = max(max(color1, color2),betterColor);
    color1 = fromTexture( index, loc+vec3(0.0,0.0,-step[2]) );
    color2 = fromTexture( index, loc+vec3(0.0,0.0,step[2]) );
    N[2] = colorToVal(color1) - colorToVal(color2);
    betterColor = max(max(color1, color2),betterColor);
    float gm = length(N); // gradient magnitude
    N = normalize(N);

    // Flip normal so it points towards viewer
    float Nselect = float(dot(N,V) > 0.0);
    N = (2.0*Nselect - 1.0) * N;  // ==  Nselect * N - (1.0-Nselect)*N;

    // Get color of the texture (albeido)
    color1 = betterColor;
    color2 = color1;
    // todo: parametrise color1_to_color2

    // Init colors
    vec4 ambient_color = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 diffuse_color = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 specular_color = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 final_color;

    // todo: allow multiple light, define lights on viewvox or subscene
    int nlights = 1;
    for (int i=0; i<nlights; i++)
    {{
        // Get light direction (make sure to prevent zero devision)
        vec3 L = normalize(view_ray);  //lightDirs[i];
        float lightEnabled = float( length(L) > 0.0 );
        L = normalize(L+(1.0-lightEnabled));

        // Calculate lighting properties
        float lambertTerm = clamp( dot(N,L), 0.0, 1.0 );
        vec3 H = normalize(L+V); // Halfway vector
        float specularTerm = pow( max(dot(H,N),0.0), u_shininess);

        // Calculate mask
        float mask1 = lightEnabled;

        // Calculate colors
        ambient_color +=  mask1 * u_ambient;  // * gl_LightSource[i].ambient;
        diffuse_color +=  mask1 * lambertTerm;
        specular_color += mask1 * specularTerm * u_specular;
    }}

    // Calculate final color by componing different components
    final_color = color2 * ( ambient_color + diffuse_color) + specular_color;
    final_color.a = color2.a;

    // Done
    return final_color;
}}

// for some reason, this has to be the last function in order for the
// filters to be inserted in the correct place...

void main() {{
    vec3 farpos = v_farpos.xyz / v_farpos.w;
    vec3 nearpos = v_nearpos.xyz / v_nearpos.w;

    // Calculate unit vector pointing in the view direction through this
    // fragment.
    view_ray = normalize(farpos.xyz - nearpos.xyz);

    // Compute the distance to the front surface or near clipping plane
    float distance = dot(nearpos-v_position, view_ray);
    distance = max(distance, min((-0.5 - v_position.x) / view_ray.x,
                            (u_shape.x - 0.5 - v_position.x) / view_ray.x));
    distance = max(distance, min((-0.5 - v_position.y) / view_ray.y,
                            (u_shape.y - 0.5 - v_position.y) / view_ray.y));
    distance = max(distance, min((-0.5 - v_position.z) / view_ray.z,
                            (u_shape.z - 0.5 - v_position.z) / view_ray.z));

    // Now we have the starting position on the front surface
    vec3 front = v_position + view_ray * distance;

    // Decide how many steps to take
    int nsteps = int(-distance / u_relative_step_size + 0.5);
    if( nsteps < 1 )
        discard;

    // Get starting location and step vector in texture coordinates
    vec3 step = ((v_position - front) / u_shape) / nsteps;
    vec3 start_loc = front / u_shape;

    // For testing: show the number of steps. This helps to establish
    // whether the rays are correctly oriented
    //gl_FragColor = vec4(0.0, nsteps / 3.0 / u_shape.x, 1.0, 1.0);
    //return;

    {before_loop}

    // This outer loop seems necessary on some systems for large
    // datasets. Ugly, but it works ...
    vec3 loc = start_loc;
    int iter = 0;
    while (iter < nsteps) {{
        for (iter=iter; iter<nsteps; iter++)
        {{
            for (int i_tex = 0; i_tex < u_n_tex; i_tex++)
            {{
                // Get sample color
                vec4 color = fromTexture(i_tex, loc);
                float val = color.g;

                {in_loop}
            }}

            {after_sampling}

            // Advance location deeper into the volume
            loc += step;
        }}
    }}

    {after_loop}

    /* Set depth value - from visvis TODO
    int iter_depth = int(maxi);
    // Calculate end position in world coordinates
    vec4 position2 = vertexPosition;
    position2.xyz += ray*shape*float(iter_depth);
    // Project to device coordinates and set fragment depth
    vec4 iproj = gl_ModelViewProjectionMatrix * position2;
    iproj.z /= iproj.w;
    gl_FragDepth = (iproj.z+1.0)/2.0;
    */
}}


"""


MIP_SNIPPETS = dict(
    before_loop="""
        float maxval = -99999.0; // The maximum encountered value
        int maxi = 0;  // Where the maximum value was encountered
        """,
    in_loop="""
        if( val > maxval ) {
            maxval = val;
            maxi = iter;
        }
        """,
    after_loop="""
        // Refine search for max value
        loc = start_loc + step * (float(maxi) - 0.5);
        for (int i=0; i<10; i++) {
            maxval = max(maxval, colorToVal(fromTexture(i_tex, loc)));
            loc += step * 0.1;
        }
        gl_FragColor += fromColormap(i_tex, maxval);
        """,
    after_sampling="""
    gl_FragColor *= gl_FragColor / u_n_tex;
        """,
)
MIP_FRAG_SHADER = FRAG_SHADER.format_map(DefaultFormat(**MIP_SNIPPETS))


TRANSLUCENT_SNIPPETS = dict(
    before_loop="""
        vec4 integrated_color = vec4(0., 0., 0., 0.);
        """,
    in_loop="""
            color = fromColormap(i_tex, val);
            float a1 = integrated_color.a;
            float a2 = color.a * (1 - a1);
            float alpha = max(a1 + a2, 0.001);

            // Doesn't work.. GLSL optimizer bug?
            //integrated_color = (integrated_color * a1 / alpha) +
            //                   (color * a2 / alpha);
            // This should be identical but does work correctly:
            integrated_color *= a1 / alpha;
            integrated_color += color * a2 / alpha;

            integrated_color.a = alpha;

            if( alpha > 0.99 ){
                // stop integrating if the fragment becomes opaque
                iter = nsteps;
            }

        """,
    after_loop="""
        gl_FragColor += integrated_color / u_n_tex;
        """,
    after_sampling="""
        """,
)
TRANSLUCENT_FRAG_SHADER = FRAG_SHADER.format_map(DefaultFormat(**TRANSLUCENT_SNIPPETS))


ADDITIVE_SNIPPETS = dict(
    before_loop="""
        vec4 integrated_color = vec4(0., 0., 0., 0.);
        """,
    in_loop="""
                color = fromColormap(i_tex, val);

                integrated_color = 1.0 - (1.0 - integrated_color) * (1.0 - color);
        """,
    after_loop="""
        gl_FragColor += integrated_color / u_n_tex;
        """,
    after_sampling="""
        """,
)
ADDITIVE_FRAG_SHADER = FRAG_SHADER.format_map(DefaultFormat(**ADDITIVE_SNIPPETS))


ISO_SNIPPETS = dict(
    before_loop="""
        vec4 color3 = vec4(0.0);  // final color
        vec3 dstep = 1.5 / u_shape;  // step to sample derivative
        gl_FragColor = vec4(0.0);
    """,
    in_loop="""
        if (val > u_threshold-0.2) {
            // Take the last interval in smaller steps
            vec3 iloc = loc - step;
            for (int i=0; i<10; i++) {
                val = colorToVal(fromTexture(i_tex, iloc));
                if (val > u_threshold) {
                    color = fromColormap(i_tex, val);
                    gl_FragColor += calculateColor(i_tex, color, iloc, dstep);
                    iter = nsteps;
                    break;
                }
                iloc += step * 0.1;
            }
        }
        """,
    after_loop="""
        """,
    after_sampling="""
    gl_FragColor *= gl_FragColor / u_n_tex;
        """,
)

ISO_FRAG_SHADER = FRAG_SHADER.format_map(DefaultFormat(**ISO_SNIPPETS))

frag_dict = {
    'mip': MIP_FRAG_SHADER,
    'iso': ISO_FRAG_SHADER,
    'translucent': TRANSLUCENT_FRAG_SHADER,
    'additive': ADDITIVE_FRAG_SHADER,
}


class MultiVolumeVisual(Visual):
    """ Displays a 3D Volume

    Parameters
    ----------
    vols : ndarray
        Volumes to display. Must be ndim==4, where number of volumes is placed
        at the first dimension.
    clims : list of tuple of two floats | None
        The contrast limits. The values in the volume are mapped to black and
        white corresponding to these values. Default maps between min and max of
        each volume.
    method : {'mip', 'translucent', 'additive', 'iso'}
        The render method to use. See corresponding docs for details.
        Default 'mip'.
    threshold : float
        The threshold to use for the isosurface render method. By default
        the mean of the given volume is used.
    relative_step_size : float
        The relative step size to step through the volume. Default 0.8.
        Increase to e.g. 1.5 to increase performance, at the cost of
        quality.
    cmaps : list of str
        Colormap to use for each volume.
    emulate_texture : bool
        Use 2D textures to emulate a 3D texture. OpenGL ES 2.0 compatible,
        but has lower performance on desktop platforms.
    """

    def __init__(self, vols, clims=None, cmaps=None, method='mip', max_vol=4,
                 threshold=None, relative_step_size=0.8, emulate_texture=False):
        tex_cls = TextureEmulated3D if emulate_texture else Texture3D

        # Storage of information of volume
        self._vol_shape = ()
        self._clims = [None for _ in range(len(vols))]
        self._need_vertex_update = True

        # Create gloo objects
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
        self._texes = [tex_cls((10, 10, 10), interpolation='linear',
                               wrapping='clamp_to_edge')
                       for _ in range(max_vol)]
        # Create colormaps
        self._cmaps = [None for _ in range(max_vol)]

        # Generate fragment shader program
        tex_declare = ""
        fromtex = ""
        fromcmap = ""
        for i in range(max_vol):
            tex = "u_volumetex{}".format(i)
            tex_declare += "uniform $sampler_type {:s};\n".format(tex)
            condition = "if (index == {:d})".format(i)
            fromtex += "{:s} return $sample({:s}, loc);\n".format(condition, tex)
            fromcmap += "{:s} return $cmap{:d}(val);\n".format(condition, i)
        for key, value in frag_dict.items():
            frag_dict[key] = value % {
                'texture_declaration': tex_declare,
                'from_texture': fromtex,
                'from_colormap': fromcmap
            }

        # Create program
        super().__init__(vcode=VERT_SHADER, fcode="")
        for i in range(len(vols)):
            self.shared_program['u_volumetex{}'.format(i)] = self._texes[i]
        self.shared_program['a_position'] = self._vertices
        self.shared_program['a_texcoord'] = self._texcoord
        self._draw_mode = 'triangle_strip'
        self._index_buffer = IndexBuffer()

        # Only show back faces of cuboid. This is required because if we are
        # inside the volume, then the front faces are outside of the clipping
        # box and will not be drawn.
        self.set_gl_state('translucent', cull_face=False)

        # Set data
        self.set_data(vols, clims)

        # Set params
        self.method = method
        self.cmaps = cmaps
        self.relative_step_size = relative_step_size
        #TODO use threshold for each volume
        self.threshold = threshold if (threshold is not None) else 0
        self.freeze()

    def set_data(self, vols, clims=None, resize=False):
        """ Set all the volume data.

        Parameters
        ----------
        vols : ndarray
            3D volumes.
        clims : lists of tuples | None
            Colormap limits to use. None will use the min and max values of each
            volume.
        """
        if len(vols) > len(self._texes):
            raise ValueError("Number of volumes ({n_vol}) exceeds number of textures ({n_tex})." \
                             .format(n_vol=len(vols), n_tex=len(self._texes)))
        if clims is None:
            clims = [None for _ in range(len(vols))]
        elif (len(vols) != len(clims)):
            raise ValueError("Number of clims does not match number of volumes.")

        for index, vol, clim in zip(count(), vols, clims):
            self.set_idata(index, vol, clim, resize)
        self.shared_program['u_n_tex'] = len(vols)

    def set_idata(self, index, vol, clim=None, resize=False):
        """ Set the volume data.
        Parameters
        ----------
        index : int
            The volume to update.
        vol : ndarray
            The 3D volume.
        clim : tuple | None
            Colormap limits to use. None will use the min and max values.
        resize : bool
            Resize underlying texture size.
        """
        if not isinstance(vol, np.ndarray):
            raise ValueError("Multi-volume visual needs a numpy array.")
        if not (vol.ndim == 3):
            raise ValueError("Multi-volume visual needs a 3D image for each volume.")

        if clim is not None:
            clim = np.array(clim, float)
            if not (clim.ndim == 1 and clim.size == 2):
                raise ValueError('clim must be a 2-element array-like')
            self._clims[index] = tuple(clim)
        if self._clims[index] is None:
            self._clims[index] = vol.min(), vol.max()

        # apply clim to data
        vol = np.array(vol, dtype='float32', copy=False)
        if self._clims[index][1] == self._clims[index][0]:
            if self._clims[index][0] != 0.:
                vol *= 1.0 / self._clim[0]
        else:
            vol -= self._clims[index][0]
            vol /= self._clims[index][1] - self._clims[index][0]

        #NOTE _tex.set_data() is efficient if vol is of same shape
        self._texes[index].set_data(vol)
        if self._vol_shape is None or self._vol_shape != vol.shape:
            self.shared_program['u_shape'] = tuple(vol.shape)
            self._vol_shape = vol.shape
            self._need_vertex_update = True

    @property
    def clims(self):
        """ The contrast limits that were applied to the volumes.
        Settable via set_data().
        """
        return self._clims

    @property
    def cmaps(self):
        return self._cmaps

    @cmaps.setter
    def cmaps(self, cmaps):
        if len(cmaps) > len(self._cmaps):
            raise ValueError("Provided colormaps ({n_cmap}) exceeds number of storage ({n_cs})." \
                             .format(n_cmap=len(cmaps), n_cs=len(self._cmaps)))
        for index, cmap in enumerate(cmaps):
            self._cmaps[index] = get_colormap(cmap)
            self.shared_program.frag['cmap{}'.format(index)] = Function(self._cmaps[index].glsl_map)
        self.update()

    @property
    def method(self):
        """The render method to use

        Current options are:

            * translucent: voxel colors are blended along the view ray until the
              result is opaque.
            * mip: maxiumum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * additive: voxel colors are added along the view ray until the
              result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are performed
              to give the visual appearance of a surface.
        """
        return self._method

    @method.setter
    def method(self, method):
        # Check and save
        known_methods = list(frag_dict.keys())
        if method not in known_methods:
            raise ValueError('Volume render method should be in %r, not %r' %
                             (known_methods, method))
        self._method = method
        # Get rid of specific variables - they may become invalid
        if 'u_threshold' in self.shared_program:
            self.shared_program['u_threshold'] = None

        self.shared_program.frag = frag_dict[method]
        self.shared_program.frag['sampler_type'] = self._texes[0].glsl_sampler_type
        self.shared_program.frag['sample'] = self._texes[0].glsl_sample
        self.update()

    @property
    def threshold(self):
        """ The threshold value to apply for the isosurface render method.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = float(value)
        if 'u_threshold' in self.shared_program:
            self.shared_program['u_threshold'] = self._threshold
        self.update()

    @property
    def relative_step_size(self):
        """ The relative step size used during raycasting.

        Larger values yield higher performance at reduced quality. If
        set > 2.0 the ray skips entire voxels. Recommended values are
        between 0.5 and 1.5. The amount of quality degredation depends
        on the render method.
        """
        return self._relative_step_size

    @relative_step_size.setter
    def relative_step_size(self, value):
        value = float(value)
        if value < 0.1:
            raise ValueError('relative_step_size cannot be smaller than 0.1')
        self._relative_step_size = value
        self.shared_program['u_relative_step_size'] = value

    def _create_vertex_data(self):
        """ Create and set positions and texture coords from the given shape

        We have six faces with 1 quad (2 triangles) each, resulting in
        6*2*3 = 36 vertices in total.
        """
        shape = self._vol_shape

        # Get corner coordinates. The -0.5 offset is to center
        # pixels/voxels. This works correctly for anisotropic data.
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

        # Apply
        self._vertices.set_data(pos)
        self._index_buffer.set_data(indices)

    def _compute_bounds(self, axis, view):
        return 0, self._vol_shape[axis]

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

MultiVolume = create_visual_node(MultiVolumeVisual)
