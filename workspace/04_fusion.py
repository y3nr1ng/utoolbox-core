
import coloredlogs
import imageio 

from utoolbox.container.datastore import (
    ImageDatastore
)
from utoolbox.stitching import Sandbox

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s  %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

ds = ImageDatastore('data/xy', imageio.imread)

sb = Sandbox(ds)