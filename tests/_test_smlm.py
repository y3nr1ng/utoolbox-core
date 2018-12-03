import logging
import os

import coloredlogs
import imageio

from utoolbox.smlm.thunderstorm import ThunderSTORM

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(module)s[%(process)d] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


##### FETCH DATA #####
src = [
    '09142018_HalotagEGFPc3Nup153+PmeI/clone1_zp5um_100ms_interval_20s_partial/clone_ch1_stack0003_638nm_0060056msec_0001984925msecAbs.tif',
    '09142018_HalotagEGFPc3Nup153+PmeI/clone1_zp5um_100ms_interval_20s_partial/clone_ch1_stack0000_638nm_0000000msec_0001924869msecAbs.tif',
    '09142018_HalotagEGFPc3Nup153+PmeI/clone1_zp5um_100ms_interval_20s_partial/clone_ch1_stack0002_638nm_0040037msec_0001964906msecAbs.tif'
]

##### EXCEUTE THUNDERSTORM #####
# ThunderSTORM(ndim, cal_file=None, tmp_dir=None)
#worker = ThunderSTORM(2, tmp_dir='/home2/scratch')
worker = ThunderSTORM(2)
worker.run(src, 'dst_dir')

##### RESULT #####
