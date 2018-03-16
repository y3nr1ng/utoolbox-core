import os

import utoolbox.utils.files as fileutils

source_folder = os.path.join(*["data", "RFiSHp2aLFCYC", "decon", "488"])

ef = fileutils.ExtensionFilter('tif')
sf = fileutils.SPIMFilter(channel=0)
file_list = fileutils.list_files(source_folder, name_filters=[ef, sf])

print(len(file_list))
