from enum import Enum
from collections import namedtuple as ntuple

class TagType(Enum):
    __slots__ = ()

    Byte      = ('Byte',             1,  'B',    1)
    ASCII     = ('ASCII',            2,  's', None)
    Short     = ('Short',            3,  'H',    2)
    Long      = ('Long',             4,  'I',    4)
    Rational  = ('Rational',         5, 'II',    8)
    SByte     = ('Signed Byte',      6,  'b',    1)
    Undefined = ('Undefined',        7,  'p', None)
    SShort    = ('Signed Short',     8,  'h',    2)
    SLong     = ('Signed Long',      9,  'i',    4)
    SRational = ('Signed Rational', 10, 'ii',    8)
    Float     = ('Float',           11,  'f',    4)
    Double    = ('Double',          12,  'd',    8)

    def __new__(cls, name, value, fmt, size):
        """
        Parse pre-defined internal table.

        Parameters
        ----------
        cls: TagType
            Type of current object used in the initialization.
        name: str
            String representation for print out.
        value: int
            Binary index in the TIFF specification.
        fmt: str
            Format string used in pack/unpack method.
        size: int
            Size to read in bytes.
        """
        obj = object.__new__(cls)
        obj._name = name
        obj._value_ = value
        obj._format = fmt
        obj._size = size
        return obj

    def __str__(self):
        return self._name

    @property
    def size(self):
        """
        Size of bytes to read.
        """
        return self._size

    @property
    def format(self):
        """
        Format string used in pack/unpack method.
        """
        return self._format

    def _missing_(value):
        """
        Return the default value.
        """
        return TagType.Undefined

class NewSubfileOptions(Enum):
    """
    A general indication of the kind of data contained in this subfile,  mainly
    useful when there are multiple subfiles in a single TIFF file.
    """
    __slots__ = ()

    Unspecified = ('Unspecified',        0)
    Reduced     = ('Reduced-resolution', 1)
    Page        = ('Multi-page',         2)
    Mask        = ('Transparency Mask',  4)

    def __new__(cls, name, value):
        """
        Parse pre-defined internal table.

        Parameters
        ----------
        cls: TagType
            Type of current object used in the initialization.
        name: str
            String representation for print out.
        value: int
            Binary index in the TIFF specification.
        """
        obj = object.__new__(cls)
        obj._name = name
        obj._value_ = value
        return obj

    def __str__(self):
        return self._name

class CompressionOptions(Enum):
    """
    Compression scheme used on the image data.
    """
    __slots__ = ()

    Uncompressed = ('Uncompressed', 1)
    CCITT        = ('CCITT 1D',     2)
    LZW          = ('LZW',          5)
    PackBits     = ('PackBits', 32773)

    def __new__(cls, name, value):
        """
        Parse pre-defined internal table.

        Parameters
        ----------
        cls: TagType
            Type of current object used in the initialization.
        name: str
            String representation for print out.
        value: int
            Binary index in the TIFF specification.
        """
        obj = object.__new__(cls)
        obj._name = name
        obj._value_ = value
        return obj

    def __str__(self):
        return self._name

class PhotometricOptions(Enum):
    """
    The color space of the image data.
    """
    __slots__ = ()

    WhiteIsZero = ('White is Zero',    0)
    BlackIsZero = ('Black is Zero',    1)
    RGB         = ('RGB',              2)
    Palette     = ('Palette RGB',      3)
    Mask        = ('Transpareny Mask', 4)

    def __new__(cls, name, value):
        """
        Parse pre-defined internal table.

        Parameters
        ----------
        cls: TagType
            Type of current object used in the initialization.
        name: str
            String representation for print out.
        value: int
            Binary index in the TIFF specification.
        """
        obj = object.__new__(cls)
        obj._name = name
        obj._value_ = value
        return obj

    def __str__(self):
        return self._name

class FillOrderOptions(Enum):
    """
    How the components of each pixel are stored.
    """
    __slots__ = ()

    MSB2LSB = 1
    LSB2MSB = 2

    def __str__(self):
        return self._name_

class OrientationOptions(Enum):
    """
    The orientation of the image with respect to the rows and columns.
    """
    __slots__ = ()

    TopLeft     = 1
    TopRight    = 2
    BottomRight = 3
    BottomLeft  = 4
    LeftTop     = 5
    RightTop    = 6
    RightBottom = 7
    LeftBottom  = 8

    def __str__(self):
        return self._name_

class PlanarConfigOptions(Enum):
    """
    How the components of each pixel are stored.
    """
    __slots__ = ()

    Chunky = 1
    Planar = 2

    def __str__(self):
        return self._name_

class ResolutionUnitOptions(Enum):
    """
    The unit of measurement for XResolution and YResolution.
    """
    __slots__ = ()

    NoUnit     = ('None',       1)
    Inch       = ('Inch',       2)
    Centimeter = ('Centimeter', 3)

    def __new__(cls, name, value):
        """
        Parse pre-defined internal table.

        Parameters
        ----------
        cls: TagType
            Type of current object used in the initialization.
        name: str
            String representation for print out.
        value: int
            Binary index in the TIFF specification.
        """
        obj = object.__new__(cls)
        obj._name = name
        obj._value_ = value
        return obj

    def __str__(self):
        return self._name_

class SampleFormatOptions(Enum):
    """
    Specifies how to interpret each data sample in a pixel.
    """
    __slots__ = ()

    UInt      = ('Unsigned Integer',    1)
    Int       = ('Signed Integer',      2)
    IEEEFP    = ('IEEE Floating Point', 3)
    Undefined = ('Undefined',           4)
    CplxInt   = ('Complex Integer',     5)
    CplxFP    = ('Complex IEEE FP',     6)

    def __new__(cls, name, value):
        """
        Parse pre-defined internal table.

        Parameters
        ----------
        cls: TagType
            Type of current object used in the initialization.
        name: str
            String representation for print out.
        value: int
            Binary index in the TIFF specification.
        """
        obj = object.__new__(cls)
        obj._name = name
        obj._value_ = value
        return obj

    def __str__(self):
        return self._name_

class Tags(Enum):
    """
    Reference extracted from the TIFF specification, containing baseline tags
    and extension tags.

    Baseline tags are those tags that are listed as part of the core of TIFF,
    the essentials that all mainstream TIFF developers should support in their
    products.

    Extension tags are those tags listed as part of TIFF features that may not
    be supported by all TIFF readers.
    """
    __slots__ = ()

    NewSubfileType   = ('New Subfile Type',   254,      TagType.Long, NewSubfileOptions.Unspecified)
    ImageWidth       = ('Image Width',        256,     TagType.Short, None)
    ImageLength      = ('Image Length',       257,     TagType.Short, None)
    BitsPerSample    = ('Bits per Sample',    258,     TagType.Short, 1)
    Compression      = ('Compression',        259,     TagType.Short, CompressionOptions.Uncompressed)
    Photometric      = ('Photometric Interpretation', 262, TagType.Short, PhotometricOptions.WhiteIsZero)
    FillOrder        = ('Fill Order',         266,     TagType.Short, FillOrderOptions.MSB2LSB)
    ImageDescription = ('Image Description',  270,     TagType.ASCII, None)
    StripOffsets     = ('Strip Offsets',      273,     TagType.Short, None)
    Orientation      = ('Orientation',        274,     TagType.Short, OrientationOptions.TopLeft)
    SamplesPerPixel  = ('Samples per Pixel',  277,     TagType.Short, 1)
    RowsPerStrip     = ('Rows per Strip',     278,     TagType.Short, 2**16-1)
    StripByteCounts  = ('Strip Byte Counts',  279,      TagType.Long, None)
    XResolution      = ('X Resolution',       282,  TagType.Rational, None)
    YResolution      = ('Y Resolution',       283,  TagType.Rational, None)
    PlanarConfig     = ('Planar Configuration', 284,   TagType.Short, PlanarConfigOptions.Chunky)
    PageName         = ('Page Name',          285,     TagType.ASCII, None)
    ResolutionUnit   = ('Resolution Unit',    296,     TagType.Short, ResolutionUnitOptions.Inch)
    PageNumber       = ('Page Number',        297,     TagType.Short, None)
    Software         = ('Software',           305,     TagType.ASCII, None)
    ColorMap         = ('Color Map',          320,     TagType.Short, None)
    ExtraSamples     = ('Extra Samples',      338,     TagType.Short, None)
    SampleFormat     = ('Sample Format',      339,     TagType.Short, SampleFormatOptions.UInt)
    Unknown          = ('<Unknown>',            0, TagType.Undefined, None)

    def __new__(cls, name, value, dtype, default):
        """
        Parse pre-defined internal table.

        Parameters
        ----------
        cls: TagType
            Type of current object used in the initialization.
        name: str
            String representation for print out.
        value: int
            Binary index in the TIFF specification.
        dtype: TagType
            Type of the field.
        default: (arbitrary)
            Default value, `None` if not specified.
        """
        obj = object.__new__(cls)
        obj._name = name
        obj._value_ = value
        obj._type = dtype
        obj._default = default
        return obj

    def __str__(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def default(self):
        return self._default

    def _missing_(value):
        """
        Return the default value.
        """
        return Tags.Unknown
