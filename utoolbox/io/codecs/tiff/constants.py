from enum import Enum
from collections import namedtuple as ntuple

class TagType(Enum):
    __slots__ = ()

    Byte      = ('Byte',             1,  'B')
    ASCII     = ('ASCII',            2,  's')
    Short     = ('Short',            3,  'H')
    Long      = ('Long',             4,  'I')
    Rational  = ('Rational',         5, 'II')
    SByte     = ('Signed Byte',      6,  'b')
    Undefined = ('Undefined',        7,  'p')
    SShort    = ('Signed Short',     8,  'h')
    SLong     = ('Signed Long',      9,  'i')
    SRational = ('Signed Rational', 10, 'ii')
    Float     = ('Float',           11,  'f')
    Double    = ('Double',          12,  'd')

    def __new__(cls, name, value, fmt):
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
        """
        obj = object.__new__(cls)
        obj._name = name
        obj._value_ = value
        obj._format = fmt
        return obj

    def __str__(self):
        return self._name

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

class NewSubfileOption(Enum):
    """
    A general indication of the kind of data contained in this subfile,  mainly
    useful when there are multiple subfiles in a single TIFF file.
    """
    __slots__ = ()

    Unspecified = ('Unspecified',    0)
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

class CompressionType(object):
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

    WhiteIsZero = ('WhiteIsZero',     0)
    BlackIsZero = ('BlackIsZero',     1)
    RGB         = ('RGB',             2)
    Palette     = ('Palette',         3)
    Mask        = ('TransparenyMask', 4)

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

class Tags(object):
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

    NewSubfileType   = ('NewSubfileType',   254,     TagType.Long, NewSubfileOption.Unspecified)
    ImageWidth       = ('ImageWidth',       256,    TagType.Short, None)
    ImageLength      = ('ImageLength',      257,    TagType.Short, None)
    BitsPerSample    = ('BitsPerSample',    258,    TagType.Short, 1)
    Compression      = ('Compression',      259,    TagType.Short, CompressionOptions.Uncompressed)
    Photometric      = ('PhotometricInterpretation', 262, TagType.Short, PhotometricOptions.WhiteIsZero)
    ImageDescription = ('ImageDescription', 270,    TagType.ASCII, None)
    StripOffsets     = ('StripOffsets',     273,    TagType.Short, None)
    Orientation      = ('Orientation',      274,    TagType.Short, OrientationOptions.TopLeft)
    SamplesPerPixel  = ('SamplesPerPixel',  277,    TagType.Short, 1)
    RowsPerStrip     = ('RowsPerStrip',     278,    TagType.Short, 2**16-1)
    StripByteCounts  = ('StripByteCounts',  279,     TagType.Long, None)
    XResolution      = ('XResolution',      282, TagType.Rational, None)
    YResolution      = ('YResolution',      283, TagType.Rational, None)
    PlanarConfig     = ('PlanarConfiguration', 284, TagType.Short, PlanarConfigOptions.Chunky)
    PageName         = ('PageName',         285,    TagType.ASCII, None)
    ResolutionUnit   = ('ResolutionUnit',   296,    TagType.Short, ResolutionUnitOptions.Inch)
    PageNumber       = ('PageNumber',       297,    TagType.Short, None)
    Software         = ('Software',         305,    TagType.ASCII, None)
    ColorMap         = ('ColorMap',         320,    TagType.Short, None)
    ExtraSamples     = ('ExtraSamples',     338,    TagType.Short, None)
    SampleFormat     = ('SampleFormat',     339,    TagType.Short, SampleFormatOptions.UInt)

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
