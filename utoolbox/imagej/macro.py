import logging
import os

__all__ = [
    'run_macro'
]

logger = logging.getLogger(__name__)

def run_macro(macro_path, *args, ij_path=None):
    import jnius_config
    if not ij_path:
        logger.info("using built-in ImageJ distribution")
        cwd = os.path.dirname(__file__)
        ij_path = os.path.join(cwd, 'ImageJ', 'ij.jar')
        if not os.path.exists(ij_path):
            raise RuntimeError("unable to locate built-in ImageJ distribution")
    jnius_config.set_classpath(ij_path)

    # NOTE use jnius_config before import jnius
    import jnius

    with open(macro_path, 'r') as fd:
        macro = fd.read()
        logger.info("{} bytes read".format(len(macro)))
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
