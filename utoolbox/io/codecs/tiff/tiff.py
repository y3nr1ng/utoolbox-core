from ..template import FileIO
from .tags import Tags, TagType
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
            ifd = IFD(self._path, self._mm, self._byte_order, self._mm.tell())
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
        #NOTE delayed tag interpretation?
        self._subfiles[index].interpret_tags()
        return self._subfiles[index]

    def __setitem__(self, index, data):
        raise NotImplementedError

class IFD(object):
    def __init__(self, path, mm, byte_order, offset):
        self._path = path
        self._mm = mm
        self._byte_order = byte_order
        self._offset = offset

        self._parse_tags()

        # lazy load the raster data
        #TODO use lazy property
        self._raster = None

    def _parse_tags(self):
        """
        Parse the containing tags in specified IFD.
        """
        (n_tags, ) = unpack(self._byte_order+'H', self._mm.read(2))

        tag_fmt = self._byte_order + 'HHII'
        raw_tags = self._mm.read(n_tags*12)

        self.tags = {
            Tags(tag_id): (TagType(dtype), count, offset)
            for (tag_id, dtype, count, offset) in iter_unpack(tag_fmt, raw_tags)
        }

    def interpret_tags(self):
        #TODO remove the wrapper?
        for tag, (ttype, count, offset) in self.tags.items():
            # skip uknown tags per specification
            if tag == Tags.Unknown:
                continue
            self.tags[tag] = self._interpret_tag(ttype, count, offset)

    def _interpret_tag(self, ttype, count, offset):
        """
        Extract the actual information of specified field.
        """
        return {
            TagType.Byte:      self._interpret_numeric_tag,
            TagType.ASCII:     self._interpret_ascii_tag,
            TagType.Short:     self._interpret_numeric_tag,
            TagType.Long:      self._interpret_numeric_tag,
            TagType.Rational:  self._interpret_rational_tag,
            TagType.SByte:     self._interpret_numeric_tag,
            TagType.Undefined: self._interpret_undefined_tag,
            TagType.SShort:    self._interpret_numeric_tag,
            TagType.SLong:     self._interpret_numeric_tag,
            TagType.SRational: self._interpret_rational_tag,
            TagType.Float:     self._interpret_numeric_tag,
            TagType.Double:    self._interpret_numeric_tag
        }[ttype](ttype, count, offset)

    def _interpret_numeric_tag(self, ttype, count, offset):
        if count == 1:
            return offset
        self._mm.seek(offset, os.SEEK_SET)
        fmt = self._byte_order + str(count) + ttype.format
        buf = self._mm.read(ttype.size * count)
        val = unpack(fmt, buf)
        return val

    def _interpret_rational_tag(self, ttype, count, offset):
        self._mm.seek(offset, os.SEEK_SET)
        fmt = self._byte_order + ttype.format
        if count == 1:
            buf = self._mm.read(ttype.size)
            (nom, den) = unpack(fmt, buf)
            val = nom/den
        else:
            raw_rationals = self._mm.read(ttype.size * count)
            val = {
                nom/den for (nom, den) in iter_unpack(fmt, raw_rationals)
            }
        return val

    def _interpret_ascii_tag(self, ttype, count, offset):
        self._mm.seek(offset, os.SEEK_SET)
        fmt = self._byte_order + str(count) + ttype.format
        buf = self._mm.read(count)
        (val, ) = unpack(fmt, buf)
        #TODO split strings by \0
        return val

    def _interpret_undefined_tag(self, ttype, count, offset):
        return hex(offset)

    @property
    def rasters(self):
        if self._raster is None:
            self._determine_image_type()

        return 'address @ 0x{}'.format(hex(self._offset))

    def _determine_image_type(self):
        """
        Using hard-coded logic to determine type of the raster from tags
        contained in the IFD.
        """
        #TODO use photometic field to determine the type
        tags = set(self.tags.keys())
        if tags < BilevelImage.REQUIRED_TAGS:
            raise TypeError('Insufficient tag information')
        elif tags < GrayscaleImage.REQUIRED_TAGS:
            print('<bilevel>')
        else:
            if tags > PaletteImage.REQUIRED_TAGS:
                print('<palette>')
            elif tags > RGBImage.REQUIRED_TAGS:
                print('<rgb>')
            else:
                print('<grayscale>')

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
