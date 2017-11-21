from utoolbox.io import imopen
from timeit import default_timer as timer

from pprint import pprint

#file_path = 'data/RAWtan1_3_3DSIMb_ch1_stack0001_561nm_0019400msec_0000215229msecAbs.tif'
file_path = '/Users/Andy/Downloads/min_data.tif'

start = timer()

with imopen(file_path, 'r') as imfile:
    for page in imfile:
        pprint(page.tags)
        print(page.rasters)

end = timer()
print('image scanned in {:.3f}s'.format(end - start))
