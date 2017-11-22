from enum import Enum

class TagName(Enum):
    NewSubfileType = 254
    SubfileType = 255
    ImageWidth = 256
    ImageLength = 257
    BitsPerSample = 258
    Compression = 259
    PhotometricInterpretation = 262
    Thresholding = 263
    CellWidth = 264
    CellLength = 265
    FillOrder = 266
    DocumentName = 269
    ImageDescription = 270
    Make = 271
    Model = 272
    StripOffsets = 273
    Orientation = 274
    SamplesPerPixel = 277
    RowsPerStrip = 278
    StripByteCounts = 279
    MinSampleValue = 280
    MaxSampleValue = 281
    XResolution = 282
    YResolution = 283
    PlanarConfiguration = 284
    PageName = 285
    XPosition = 286
    YPosition =  287
    FreeOffsets = 288
    FreeByteCounts = 289
    GrayResponseUnit = 290
    GrayResponseCurve = 291
    T4Options = 292
    T6Options = 293
    ResolutionUnit = 296
    PageNumber = 297
    TransferFunction = 301
    Software = 305
    DateTime = 306
    Artist = 315
    HostComputer = 316
    Predictor = 317
    WhitePoint = 318
    PrimaryChromaticities = 319
    lorMap = 320
    HalftoneHints = 321
    TileWidth = 322
    TileLength = 323
    TileOffsets = 324
    TileByteCounts = 325
    InkSet = 332
    InkNames = 333
    NumberOfInks = 334
    DotRange = 336
    TargetPrinter = 337
    ExtraSamples = 338
    SampleFormat = 339
    SMinSampleValue = 340
    SMaxSampleValue = 341
    TransferRange = 342
    JPEGProc = 512
    JPEGInterchangeFormat = 513
    JPEGInterchangeFormatLngth = 514
    JPEGRestartInterval = 515
    JPEGLosslessPredictors = 517
    JPEGPointTransforms = 518
    JPEGQTables = 519
    JPEGDCTables = 520
    JPEGACTables = 521
    YCbCrCoefficients = 529
    YCbCrSubSampling = 530
    YCbCrPositioning = 531
    ReferenceBlackWhite = 532
    Copyright = 33432

class CompressionOptions(Enum):
    Uncompressed = 1
    CCITT = 2
    Group3 = 3
    Group4 = 4
    LZW = 5
    JPEG = 6
    PackBits = 32773

class PhotometricOptions(Enum):
    WhiteIsZero = 0
    BlackIsZero = 1
    RGB = 2
    Palette = 3
    TransparencyMask = 4
    CMYK = 5
    YCbCr = 6
    CIELab = 8

class Tags(object):
    def __init__(self):
        pass
