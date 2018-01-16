import os
from mmap import mmap, ACCESS_READ, ACCESS_WRITE
from struct import unpack, iter_unpack
from operator import add

import numpy as np

from ..template import FileIO
from .tags import Tags, TagType, SampleFormatOptions
from utoolbox.utils.decorators import run_once

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
            raise ValueError("Invalid mode.")

    def _parse_header(self):
        """Analyze TIFF header."""
        (identifier, magic) = unpack('HH', self._mm.read(4))
        if magic != 42:
            raise TypeError("Not a TIFF file.")
        self._set_byte_order(identifier)

        # extract offset of the 1st IFD
        (first_ifd_offset, ) = unpack(self._byte_order+'I', self._mm.read(4))
        return first_ifd_offset

    def _set_byte_order(self, order):
        """Determine byte order by identifier string."""
        if order == 0x4949:
            self._byte_order = '<'
        elif order == 0x4D4D:
            self._byte_order = '>'
        else:
            raise ValueError("Invalid byte order identifier.")

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
        """Return number of subfiles."""
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
        self._subfiles[index].interpret_tags()
        return self._subfiles[index]

    def __setitem__(self, index, data):
        raise NotImplementedError

    @property
    def shape(self):
        shapes = [
            (page.tags[Tags.ImageWidth], page.tags[Tags.ImageLength])
            for page in self
        ]
        width, length = shapes[0]
        depth = len(self)
        return (width, length, depth) if shapes[1:] == shapes[:-1] else shapes

    @property
    def dtype(self):
        dtypes = [page.dtype for page in self._subfiles]
        return dtypes[0] if dtypes[1:] == dtypes[:-1] else dtypes

class IFD(object):
    def __init__(self, path, mm, byte_order, offset):
        self._path = path
        self._mm = mm
        self._byte_order = byte_order
        self._offset = offset

        self._parse_tags()

        # lazy load the raster data
        self._raster = None

    def _parse_tags(self):
        """Parse the containing tags in specified IFD."""
        (n_tags, ) = unpack(self._byte_order+'H', self._mm.read(2))

        tag_fmt = self._byte_order + 'HHII'
        raw_tags = self._mm.read(n_tags*12)

        self.tags = {
            Tags(tag_id): (TagType(dtype), count, offset)
            for (tag_id, dtype, count, offset) in iter_unpack(tag_fmt, raw_tags)
        }

    @run_once
    def interpret_tags(self):
        """
        Interpret the raw tag definitions and degrade into no-op.
        """
        for tag, (ttype, count, offset) in self.tags.items():
            # skip uknown tags per specification
            if tag == Tags.Unknown:
                continue
            self.tags[tag] = self._interpret_tag(ttype, count, offset)

    def _interpret_tag(self, ttype, count, offset):
        """
        Extract the actual information of specified field.

        Args:
            ttype:
                TBA
            count: integer
                TBA
            offset: integer
                TBA
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
        # NOTE only planned to support utf-8
        val = val.decode('utf-8', 'backslashreplace')
        return val

    def _interpret_undefined_tag(self, ttype, count, offset):
        return hex(offset)

    @property
    def raster(self):
        if self._raster is None:
            self._determine_image_type()
            self._raster = self._parse_raster()
        return self._raster

    def _determine_image_type(self):
        """
        Using hard-coded logic to determine type of the raster from tags
        contained in the IFD.
        """
        #TODO use photometic field to determine the type
        tags = set(self.tags.keys())
        if tags < BilevelImage.REQUIRED_TAGS:
            raise TypeError("Insufficient tag information.")
        elif tags < GrayscaleImage.REQUIRED_TAGS:
            self.__class__ = BilevelImage
        else:
            if tags > PaletteImage.REQUIRED_TAGS:
                self.__class__ = PaletteImage
            elif tags > RGBImage.REQUIRED_TAGS:
                self.__class__ = RGBImage
            else:
                self.__class__ = GrayscaleImage

    def _parse_raster(self):
        raise NotImplementedError("Raster parser is not specified.")

    def _stripes_summary(self):
        """Summarize stripe records for raster extraction."""
        return zip(self.tags[Tags.StripOffsets],
                   self.tags[Tags.StripByteCounts])

    @property
    def continuous(self):
        """
        Determine whether stripes are placed in continuous block of memory.
        """
        offset = self.tags[Tags.StripOffsets]
        offset_shift = list(map(add, offset, self.tags[Tags.StripByteCounts]))
        return all(i == j for i, j in zip(offset[1:], offset_shift[:-1]))

    @property
    def dtype(self):
        return {
            # unsigned integer
            (SampleFormatOptions.UInt, 16):     np.uint16,
            (SampleFormatOptions.UInt, 32):     np.uint32,
            (SampleFormatOptions.UInt, 64):     np.uint64,

            # integer
            (SampleFormatOptions.Int, 16):      np.int16,
            (SampleFormatOptions.Int, 32):      np.int32,
            (SampleFormatOptions.Int, 64):      np.int64,

            # floating point
            (SampleFormatOptions.IEEEFP, 16):   np.float16,
            (SampleFormatOptions.IEEEFP, 32):   np.float32,
            (SampleFormatOptions.IEEEFP, 64):   np.float64
        }[(SampleFormatOptions(self.tags[Tags.SampleFormat]), self.tags[Tags.BitsPerSample])]

class BilevelImage(IFD):
    REQUIRED_TAGS = {
        256, 257, 259, 262, 273, 278, 279, 282, 283, 296
    }

    def _parse_raster(self):
        raise NotImplementedError('<bilevel>')

    def __repr__(self):
        return 'Bilevel Image'

class GrayscaleImage(BilevelImage):
    REQUIRED_TAGS = {
        256, 257, 258, 259, 262, 273, 278, 279, 282, 283,
        296
    }

    def _parse_raster(self):
        shape = (self.tags[Tags.ImageWidth], self.tags[Tags.ImageLength])
        if self.continuous:
            data = np.memmap(self._path,
                             dtype=self.dtype,
                             shape=shape,
                             offset=self.tags[Tags.StripOffsets][0])
        else:
            raise NotImplementedError('<grayscale> sequential')
        return data

    def __repr__(self):
        return 'Grayscale Image'

class PaletteImage(GrayscaleImage):
    REQUIRED_TAGS = {
        256, 257, 258, 259, 262, 273, 278, 279, 282, 283,
        296, 320
    }

    def _parse_raster(self):
        raise NotImplementedError('<palette>')

    def __repr__(self):
        return 'Palette Image'

class RGBImage(GrayscaleImage):
    REQUIRED_TAGS = {
        256, 257, 258, 259, 262, 273, 277, 278, 279, 282,
        283, 296
    }

    def _parse_raster(self):
        raise NotImplementedError('<rgb>')

    def __repr__(self):
        return 'RGB Image'
