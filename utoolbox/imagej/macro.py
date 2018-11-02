from collections import OrderedDict
import logging
import os

from mako.template import Template
from mako.lookup import TemplateLookup

__all__ = [
    'run_macro',
    'run_macro_file',
    'Macro'
]

logger = logging.getLogger(__name__)


def run_macro(macro, *args, ij_path=None, plugins_dir=None):
    import jnius_config
    if not ij_path:
        logger.info("using built-in ImageJ distribution")
        cwd = os.path.dirname(__file__)
        ij_path = os.path.join(cwd, 'ImageJ', 'ij.jar')
        if not os.path.exists(ij_path):
            raise RuntimeError("unable to locate built-in ImageJ distribution")

    if not plugins_dir:
        ij_root = os.path.dirname(ij_path)
        plugins_dir = os.path.join(ij_root, 'plugins')
    jnius_config.set_options('-Dplugins.dir={}'.format(plugins_dir))

    jnius_config.set_classpath(ij_path)

    # NOTE use jnius_config before import jnius
    import jnius

    String = jnius.autoclass('java.lang.String')
    macro = String(macro)

    # wrap input arguments
    if args:
        args = ','.join([str(arg) for arg in args])
    else:
        args = ''
    args = String(args)

    # run it
    MacroRunner = jnius.autoclass('ij.macro.MacroRunner')
    macro = MacroRunner(macro, args)
    macro.run()

def run_macro_file(path, *args, **kwargs):
    with open(macro_path, 'r') as fd:
        macro = fd.read()
        logger.info("{} bytes read".format(len(macro)))
    run_macro(macro, *args, **kwargs)

class Macro(object):
    """
    Describes an ImageJ macro file as an object.
    """
    def __init__(self):
        cwd = os.path.dirname(os.path.abspath(__file__))
        tpl_dir = os.path.join(cwd, 'macro_template')
        self._templates = TemplateLookup(directories=[tpl_dir])

        self._design = {
            'prologue': '',
            'body': {},
            'epilogue': ''
        }

        self.batch_mode = False

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        self._args = tuple(str(arg) for arg in args)

    @property
    def batch_mode(self):
        return 'batch_mode' in self._design['body']

    @batch_mode.setter
    def batch_mode(self, mode):
        if mode and (not self.batch_mode):
            self._design['body']['batch_mode'] = \
                self._templates.get_template('batch_mode.mako')
        elif (not mode) and self.batch_mode:
            self._design['body'].pop('batch_mode', '')

    @property
    def loop_files(self):
        return 'loop_files' in self._design['body']

    @loop_files.setter
    def loop_files(self, mode):
        if mode and (not self.loop_files):
            self._design['body']['loop_files'] = \
                self._templates.get_template('loop_files.mako')
        elif (not mode) and self.loop_files:
            self._design['body'].pop('loop_files', '')

    @property
    def main(self):
        return self._design['body']['main']

    @main.setter
    def main(self, text):
        self._design['body']['main'] = Template(text)

    def render(self):
        if 'main' not in self._design['body']:
            raise RuntimeError("main text not found")

        # sort in encapsulation order
        order = ('batch_mode', 'loop_files', 'main')
        tbody = OrderedDict(sorted(
            self._design['body'].items(),
            key=lambda k: order.index(k[0]), reverse=True
        ))

        rbody = ''
        for body in tbody.values():
            rbody = body.render(body=rbody)
        return rbody

        #TODO merge design

        return self._parse_args()

    def _parse_args(self):
        if not self._args:
            return ''
        template = self._templates.get_template('parse_args.mako')
        return template.render(arg_list=self._args)
