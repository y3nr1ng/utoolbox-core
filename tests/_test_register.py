
import coloredlogs
coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s  %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

from utoolbox.feature import DftRegister



with DftRegister() as reg:
    pass