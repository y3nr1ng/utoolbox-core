from ..template import FileIO
from .tags import Tags
from mmap import mmap, ACCESS_READ, ACCESS_WRITE
import os
from struct import unpack

import matplotlib.pyplot as plt

#DEBUG
from timeit import default_timer as timer

class Tiff(FileIO):
    def __init__(self, path, mode):
        print('in Tiff.__init__')

        self._mode = mode
        self._path = path

        start = timer()

        (o_mode, m_mode) = Tiff._interpret_mode(mode)
        with open(path, o_mode) as fd:
            with mmap(fd.fileno(), 0, access=m_mode) as mm:
                self._parse_header(mm)
                self._enumerate_ifd_offsets(mm)

        end = timer()
        print('image scanned in {:.3f}s'.format(end - start))

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

    def _parse_header(self, mm):
        """
        Analyze TIFF header.
        """
        (identifier, magic) = unpack('HH', mm.read(4))
        if magic != 42:
            raise TypeError('Not a TIFF file')
        self._set_byte_order(identifier)

        # extract offset of the 1st IFD
        (offset, ) = unpack(self._byte_order+'I', mm.read(4))
        self._ifd_offsets = [offset]

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

    def _enumerate_ifd_offsets(self, mm):
        next_ifd_offset = self._ifd_offsets[0]
        while next_ifd_offset != 0:
            # move to next IFD
            mm.seek(next_ifd_offset, os.SEEK_SET)

            (n_tags, ) = unpack(self._byte_order+'H', mm.read(2))
            # store (current offset, number of tags)
            self._ifd_offsets.append((mm.tell(), n_tags))

            # each tag entry contains 12 bytes, skip them
            mm.seek(n_tags*12, os.SEEK_CUR)
            # update the offset
            (next_ifd_offset, ) = unpack(self._byte_order+'I', mm.read(4))

        print('{} IFDs found'.format(len(self._ifd_offsets)))
