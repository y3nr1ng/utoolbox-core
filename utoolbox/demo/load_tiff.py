from utoolbox.io import imopen
from utoolbox.io.codecs.tiff.tags import Tags
from timeit import default_timer as timer

from pprint import pprint

file_path = 'data/RAWtan1_3_3DSIMb_ch1_stack0004_561nm_0077599msec_0000273428msecAbs.tif'
#file_path = '/Users/Andy/Downloads/min_data.tif'

start = timer()

with imopen(file_path, 'r') as imfile:
    for page in imfile:
        for tag_id, tag_info in page.tags.items():
            print('{}: {}'.format(Tags(tag_id).name, tag_info))
        print(page.rasters)
        print()

        # only test the first page
        break

end = timer()
print('image scanned in {:.3f}s'.format(end - start))

input('Press Enter to continue...')
