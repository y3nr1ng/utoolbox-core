from timeit import default_timer as timer
from pprint import pprint

import matplotlib.pyplot as plt

from utoolbox.io import imopen
from utoolbox.io.codecs.tiff.tags import Tags

file_path = 'data/cell4_ch0_stack0000_488nm_0000000msec_0007934731msecAbs_decon.tif'

start = timer()

with imopen(file_path, 'r') as imfile:
    h = None
    for page in imfile:
        #for tag, tag_info in page.tags.items():
        #    print('{} ({}): {}'.format(tag, tag.value, tag_info))
        if h is None:
            h = plt.imshow(page.raster)
            plt.ion()
            plt.show()
        else:
            h.set_data(page.raster)
            plt.draw()
        plt.pause(0.001)

        # only test the first page
        #break

end = timer()
print('image scanned in {:.3f}s'.format(end - start))

input('Press Enter to continue...')
