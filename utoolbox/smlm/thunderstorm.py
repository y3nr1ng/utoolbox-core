import logging
import os
import subprocess as sp
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import NamedTuple

from mako.template import Template

from utoolbox.imagej import run_macro_file

__all__ = [
    'ThunderSTORM'
]

logger = logging.getLogger(__name__)


class ThunderSTORM(object):
    class DefaultParameters(NamedTuple):
        # camera pixel size [nm]
        px_size: int = 102
        # quantum efficiency
        qe: float = .85
        # background offset
        offset: int = 100

        # wavelet S.D. level
        level: float = 1.

        # export path
        path: str = ''
        # output floating precision
        precision: int = 1

    def __init__(self, ndim, ij_root=None, cal_file=None, tmp_dir=None):
        self._ndim, self._cal_file = ndim, cal_file
        self._parameters = ThunderSTORM.DefaultParameters()._asdict()

        self._tmp_dir = tmp_dir

    def __call__(self, src, dst_dir):
        """Convenient function for run()."""
        self.run(src, dst_dir)

    @property
    def ndim(self):
        return self._ndim

    def set_camera_options(self, px_size=102, qe=0.85, offset=100.0):
        self._parameters['px_size'] = int(px_size)
        self._parameters['qe'] = qe
        self._parameters['offset'] = int(offset)

    def set_analysis_options(self, level):
        self._parameters['level'] = float(level)

    def set_export_options(self, precision=1):
        self._parameters['precision'] = int(precision)

    def run(self, src, dst_dir):
        if isinstance(src, str):
            if os.path.isdir(src):
                file_list = [os.path.join(src, fn) for fn in os.listdir(src)]
            else:
                # file
                file_list = [src]
        elif isinstance(src, list):
            file_list = src
        else:
            raise ValueError("unknown source")

        # NOTE delayed configuration
        #self._parameters['path'] = dst_dir
        # TODO use temporary folder
        self._parameters['path'] = '\" + path + \"'
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
            logger.debug('"{}" created'.format(dst_dir))

        with TemporaryDirectory(dir=self._tmp_dir) as workspace:
            # create file list
            file_list_path = os.path.join(workspace, 'files.txt')
            with open(file_list_path, 'w') as fd:
                for file in file_list:
                    fd.write('{}\n'.format(os.path.abspath(file)))
            print(file_list)

            # build macro
            cwd = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.join(cwd, 'template.ijm')
            template = Template(filename=template_path)
            macro = template.render(
                file_list=file_list_path,
                camera_setup=self._build_camera_setup(),
                run_analysis=self._build_run_analysis(),
                export_results=self._build_export_results(),
                dst_dir=dst_dir
            )
            macro_path = os.path.join(workspace, 'macro.ijm')
            with open(macro_path, 'w') as fd:
                fd.write(macro)
            print(macro)

            cwd = os.path.dirname(__file__)
            plugins_dir = os.path.join(cwd, 'ij_plugins')

            # NOTE ThunderSTORM cannot run under headless mode
            run_macro_file(macro_path, plugins_dir=plugins_dir, headless=False)

            # TODO merge the result using DataFrame
        logger.warning("workspace wiped")

    def _build_camera_setup(self):
        return self._build_command_str(
            'Camera setup',
            {
                'readoutnoise': 1.64,
                'offset': self._parameters['offset'],
                'quantumefficiency': self._parameters['qe'],
                'isemgain': 'false',
                'photons2adu': 0.47,
                'pixelsize': self._parameters['px_size']
            }
        )

    def _build_run_analysis(self):
        filter_parameters = {
            'filter': '[Wavelet filter (B-Spline)]',
            'scale': 2.0,
            'order': 3
        }

        threshold = '{:.1f}*std(Wave.F1)'.format(self._parameters['level'])
        detector_parameters = {
            'detector': '[Local maximum]',
            'connectivity': '8-neighbourhood',
            'threshold': threshold
        }

        if self.ndim == 2:
            estimator = '[PSF: Integrated Gaussian]'
        else:
            estimator = '[PSF: Elliptical Gaussian (3D astigmatism)]'
        estimator_parameters = {
            'estimator': estimator,
            'sigma': 1.6,
            'fitradius': 4
        }

        method_parameters = {
            'method': '[Maximum likelihood]',
            'full_image_fitting': 'false',
            'mfaenabled': 'false'
        }
        if self.ndim == 3:
            if self._cal_file is None:
                raise ValueError("invalid calibration file path")
            method_parameters = {
                'calibrationpath': '[{}]'.format(self._cal_file),
                **method_parameters
            }

        # NOTE these options are stubs
        renderer_parameters = {
            'renderer': '[Averaged shifted histograms]',
            'magnification': 1.0,
            'colorize': 'false',
            'threed': 'false',
            'shifts': 2,
            'repaint': 100
        }

        return self._build_command_str(
            'Run analysis',
            {
                **filter_parameters,
                **detector_parameters,
                **estimator_parameters,
                **method_parameters,
                **renderer_parameters
            }
        )

    def _build_export_results(self):
        common_parameters = {
            'filepath': self._parameters['path'],
            'fileformat': '[CSV (comma separated)]',
            'floatprecision': self._parameters['precision'],
            'saveprotocol': 'false',
            'id': 'false',
            'frame': 'true',
            'x': 'true',
            'y': 'true',
            'intensity': 'true',
            'uncertainty_xy': 'true',
            'bkgstd': 'true',
            'offset': 'true'
        }

        # append dimension dependend parameters
        if self.ndim == 2:
            parameters = {
                'sigma': 'true',
                **common_parameters
            }
        else:
            parameters = {
                'z': 'true',
                'uncertainty_z': 'true',
                'sigma1': 'true',
                'sigma2': 'true',
                **common_parameters
            }

        return self._build_command_str('Export results', parameters)

    def _build_command_str(self, command, parameters):
        merged_parameters = [
            "{}={}".format(k, v) for k, v in parameters.items()
        ]
        merged_parameters = ' '.join(merged_parameters)
        return 'run("{}", "{}");'.format(command, merged_parameters)
