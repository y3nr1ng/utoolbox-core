from utoolbox.io import imopen
from utoolbox.io.codecs.tiff.tags import TagName
from timeit import default_timer as timer

from pprint import pprint

from utoolbox.io.codecs.tiff.constants import TagType
from random import randint

start = timer()
n = 10000
for _ in range(n):
    x = TagType(randint(1, 13))
end = timer()
dt = end-start
print('elapsed {:.3f}s, {:.3f}us per iteration'.format(dt, dt/n*1e6))
print()

raise RuntimeError('PAUSE')

file_path = 'data/RAWtan1_3_3DSIMb_ch1_stack0004_561nm_0077599msec_0000273428msecAbs.tif'
#file_path = '/Users/Andy/Downloads/min_data.tif'

start = timer()

with imopen(file_path, 'r') as imfile:
    for page in imfile:
        for tag_id, tag_info in page.tags.items():
            print('{}: {}'.format(TagName[tag_id].name, tag_info))
        print(page.rasters)
        print()

end = timer()
print('image scanned in {:.3f}s'.format(end - start))

input('Press Enter to continue...')
