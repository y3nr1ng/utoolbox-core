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

    NewSubfileType   = ('NewSubfileType',   254,     TagType.Long, 1, None)
    SubfileType      = ('SubfileType',      255,    TagType.Short, 1, None)
    ImageWidth       = ('ImageWidth',       256,    TagType.Short, 1, None)
    ImageLength      = ('ImageLength',      257,    TagType.Short, 1, None)
    BitsPerSample    = ('BitsPerSample',    258,    TagType.Short, '<SamplesPerPixel>', 1)
    Compression      = ('Compression',      259,    TagType.Short, CompressionOptions.Uncompressed)
    Photometric      = ('PhotometricInterpretation', 262, TagType.Short, PhotometricOptions.WhiteIsZero)
    ImageDescription = ('ImageDescription', 270,    TagType.ASCII, None)
    Make             = ('Make',             271,    TagType.ASCII, None)
    Model            = ('Model',            272,    TagType.ASCII, None)
    StripOffsets     = ('StripOffsets',     273,    TagType.Short, '<StripsPerImage>')
    Orientation      = ('Orientation',      274,    TagType.Short, 1)
    SamplesPerPixel  = ('SamplesPerPixel',  277,    TagType.Short, 1)
    RowsPerStrip     = ('RowsPerStrip',     278,    TagType.Short, 1)
    StripByteCounts  = ('StripByteCounts',  279,     TagType.Long,  1)
    XResolution      = ('XResolution',      282, TagType.Rational, )
    YResolution      = ('YResolution',      283, TagType.Rational, )
    PlanarConfig     = ('PlanarConfiguration', 284, TagType.Short, '<Contig>')
    PageName         = ('PageName',         285,    TagType.ASCII, )
    XPosition        = ('XPosition',        286, TagType.Rational, )
    YPosition        = ('YPosition',        287, TagType.Rational, )
    ResolutionUnit   = ('ResolutionUnit',   296,    TagType.Short, '<None>')
    PageNumber       = ('PageNumber',       297,    TagType.Short, )
    Software         = ('Software',         305,    TagType.ASCII, None)
    DateTime         = ('DateTime',         306,    TagType.ASCII, None)
    Artist           = ('Artist',           315,    TagType.ASCII, None)
    HostComputer     = ('HostComputer',     316,    TagType.ASCII, None)
    ColorMap         = ('ColorMap',         320,    TagType.Short, None)
    ExtraSamples     = ('ExtraSamples',     338,    TagType.Short, None)
    SampleFormat     = ('SampleFormat',     339,    TagType.Short, '<unsigned integer>')

    def __new__(cls, name, value, dtype, count, default):
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
        count: int
            Default number of field to interpret.
        default: (arbitrary)
            Default value, `None` if not specified.
        """
        obj = object.__new__(cls)
        obj._name = name
        obj._value_ = value
        obj._type = dtype
        obj._count = count
        obj._default = default
        return obj

    def __str__(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def count(self):
        return self._count

    @property
    def default(self):
        return self._default
