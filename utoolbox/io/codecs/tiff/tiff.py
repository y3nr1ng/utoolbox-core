from ..template import FileIO
from .tags import Tags
from mmap import mmap, ACCESS_READ, ACCESS_WRITE
import os
from struct import unpack, iter_unpack

class Tiff(FileIO):
    def __init__(self, path, mode):
        self._mode = mode
        self._path = path

        self._subfiles = []
        self._current_page = 0

    def __enter__(self):
        (o_mode, m_mode) = Tiff._interpret_mode(self._mode)

        # open and map the file
        self._fd = open(self._path, o_mode)
        self._mm = mmap(self._fd.fileno(), 0, access=m_mode)

        first_ifd_offset = self._parse_header()
        self._enumerate_ifds(first_ifd_offset)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._mm.close()
        self._fd.close()

    @staticmethod
    def _interpret_mode(mode):
        """
        Reinterpret module mode into syscall mode.

        Parameter
        ---------
        mode: str
            Mode for the Tiff module.

        Return
        ------
        Mode for (open, mmap) syscall.
        """
        if mode == 'r' or mode == 'w':
            o_mode = mode + 'b'
            m_mode = ACCESS_WRITE if mode == 'w' else ACCESS_READ
            return (o_mode, m_mode)
        else:
            raise ValueError('Invalid mode')

    def _parse_header(self):
        """
        Analyze TIFF header.
        """
        (identifier, magic) = unpack('HH', self._mm.read(4))
        if magic != 42:
            raise TypeError('Not a TIFF file')
        self._set_byte_order(identifier)

        # extract offset of the 1st IFD
        (first_ifd_offset, ) = unpack(self._byte_order+'I', self._mm.read(4))
        return first_ifd_offset

    def _set_byte_order(self, order):
        """
        Determine byte order by identifier string.
        """
        if order == 0x4949:
            self._byte_order = '<'
        elif order == 0x4D4D:
            self._byte_order = '>'
        else:
            raise ValueError('Invalid byte order identifier')

    def _enumerate_ifds(self, first_ifd_offset):
        next_ifd_offset = first_ifd_offset
        while next_ifd_offset != 0:
            # move to next IFD
            self._mm.seek(next_ifd_offset, os.SEEK_SET)

            # create IFD object
            ifd = IFD(self._path, self._mm.tell())
            ifd.parse_tags(self._mm, self._byte_order)
            self._subfiles.append(ifd)

            # update the offset
            (next_ifd_offset, ) = unpack(self._byte_order+'I', self._mm.read(4))

        print('{} subfiles'.format(len(self)))

    def __len__(self):
        """
        Return number of subfiles.
        """
        return len(self._subfiles)

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_page == len(self):
            self._current_page = 0
            raise StopIteration
        else:
            self._current_page += 1
            return self[self._current_page-1]

    def __getitem__(self, index):
        return self._subfiles[index]

    def __setitem__(self, index, data):
        raise NotImplementedError

class IFD(object):
    def __init__(self, path, offset):
        self._path = path
        self._offset = offset

        # lazy load the raster data
        #TODO use lazy property
        self._raster = None

    def parse_tags(self, mm, byte_order):
        """
        Parse the containing tags in specified IFD.
        """
        (n_tags, ) = unpack(byte_order+'H', mm.read(2))

        tag_fmt = byte_order + 'HHII'
        raw_tags = mm.read(n_tags*12)
        #TODO replace Tag list instead of dict
        self.tags = {
            tag_id: (dtype, count, offset)
            for (tag_id, dtype, count, offset) in iter_unpack(tag_fmt, raw_tags)
        }

    @property
    def rasters(self):
        if self._raster is None:
            self._determine_image_type()

        address = format(self._offset, '02x')
        n_tags = len(self.tags)
        return 'address @ 0x{}, {} tags'.format(address, n_tags)

    def _determine_image_type(self):
        """
        Using hard-coded logic to determine type of the raster from tags
        contained in the IFD.
        """
        tags = set(self.tags.keys())
        if tags < BilevelImage.REQUIRED_TAGS:
            raise TypeError('Insufficient tag information')
        elif tags < GrayscaleImage.REQUIRED_TAGS:
            print('-> bilevel')
        else:
            if tags > PaletteImage.REQUIRED_TAGS:
                print('-> palette')
            elif tags > RGBImage.REQUIRED_TAGS:
                print('-> rgb')
            else:
                print('-> grayscale')

class Tags(object):
    def __init__(self, tag_id, dtype, count, offset):
        pass

    def __str__(self):
        pass

class BilevelImage(IFD):
    REQUIRED_TAGS = {
        256, 257, 259, 262, 273, 278, 279, 282, 283, 296
    }
    pass

class GrayscaleImage(BilevelImage):
    REQUIRED_TAGS = {
        256, 257, 258, 259, 262, 273, 278, 279, 282, 283,
        296
    }
    pass

class PaletteImage(GrayscaleImage):
    REQUIRED_TAGS = {
        256, 257, 258, 259, 262, 273, 278, 279, 282, 283,
        296, 320
    }
    pass

class RGBImage(GrayscaleImage):
    REQUIRED_TAGS = {
        256, 257, 258, 259, 262, 273, 277, 278, 279, 282,
        283, 296
    }
    pass
