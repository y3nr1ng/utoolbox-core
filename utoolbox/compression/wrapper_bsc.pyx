import os

with open(os.path.join('libbsc', 'VERSION'), 'r') as fd:
    LIBBSC_VERSION = fd.read()

cdef extern from "libbsc/libbsc.h":
    cdef int LIBBSC_NO_ERROR               
    cdef int LIBBSC_BAD_PARAMETER          
    cdef int LIBBSC_NOT_ENOUGH_MEMORY      
    cdef int LIBBSC_NOT_COMPRESSIBLE       
    cdef int LIBBSC_NOT_SUPPORTED          
    cdef int LIBBSC_UNEXPECTED_EOB         
    cdef int LIBBSC_DATA_CORRUPT   

    cdef int LIBBSC_GPU_ERROR              
    cdef int LIBBSC_GPU_NOT_SUPPORTED      
    cdef int LIBBSC_GPU_NOT_ENOUGH_MEMORY  

    cdef int LIBBSC_BLOCKSORTER_NONE       
    cdef int LIBBSC_BLOCKSORTER_BWT        

    cdef int LIBBSC_CODER_NONE              
    cdef int LIBBSC_CODER_QLFC_STATIC       
    cdef int LIBBSC_CODER_QLFC_ADAPTIVE    

    cdef int LIBBSC_FEATURE_NONE            
    cdef int LIBBSC_FEATURE_FASTMODE        
    cdef int LIBBSC_FEATURE_MULTITHREADING  
    cdef int LIBBSC_FEATURE_LARGEPAGES      
    cdef int LIBBSC_FEATURE_CUDA           

    cdef int LIBBSC_DEFAULT_LZPHASHSIZE     
    cdef int LIBBSC_DEFAULT_LZPMINLEN       
    cdef int LIBBSC_DEFAULT_BLOCKSORTER   
    cdef int LIBBSC_DEFAULT_CODER         
    cdef int LIBBSC_DEFAULT_FEATURES        

    cdef int LIBBSC_HEADER_SIZE             

    int bsc_init(int features)

    int bsc_init_full(
        int features, 
        void * (*malloc)(size_t size),
        void * (*zero_malloc)(size_t size),
        void (*free)(void *address)
    )

    int bsc_compress(
        const unsigned char *input_, 
        unsigned char *output, 
        int n, 
        int lzpHashSize, 
        int lzpMinLen, 
        int blockSorter, 
        int coder, 
        int features
    )

    int bsc_store(
        const unsigned char *input, 
        unsigned char *output, 
        int n, 
        int features
    )

    int bsc_block_info(
        const unsigned char *blockHeader, 
        int headerSize, 
        int *pBlockSize, 
        int *pDataSize, 
        int features
    )

    int bsc_decompress(
        const unsigned char *input, 
        int inputSize, 
        unsigned char *output, 
        int outputSize, 
        int features
    )

class BSCError(Exception):
    """Exception raised on compression and decompression errors."""

class BadParameterError(BSCError):
    """Bad input parameters."""

class NotEnoughMemoryError(BSCError):
    """Not enough host memory."""

class NotCompressibleError(BSCError):
    """Input data is not compressible."""

class NotSupportedError(BSCError):
    """Unsupported feature."""

class UnexpectedEOBError(BSCError):
    """Unexpected EOB."""

class DataCorruptError(BSCError):
    """Data corrupted."""

class GPUError(BSCError):
    """GPU error."""

class GPUNotSupportedError(BSCError):
    """GPU is not supported in provided library."""

class GPUNotEnoughMemoryError(BSCError):
    """Not enough GPU memory."""

ERROR_CODE_TO_EXCEPTION = {
    LIBBSC_BAD_PARAMETER: BadParameterError,
    LIBBSC_NOT_ENOUGH_MEMORY: NotEnoughMemoryError,
    LIBBSC_NOT_COMPRESSIBLE: NotCompressibleError,
    LIBBSC_NOT_SUPPORTED: NotSupportedError, 
    LIBBSC_UNEXPECTED_EOB: UnexpectedEOBError,
    LIBBSC_DATA_CORRUPT: DataCorruptError,
    LIBBSC_GPU_ERROR: GPUError,
    LIBBSC_GPU_NOT_SUPPORTED: GPUNotSupportedError,
    LIBBSC_GPU_NOT_ENOUGH_MEMORY: GPUNotEnoughMemoryError
}

ctypedef unsigned char uchar
ctypedef long long int64

cdef class Compress:
    cdef:
        int _block_size
        int _hash_size
        int _min_len
        int _sorter
        int _coder

        bytes _buffer
        uchar _cbuffer
    
    def __init__(
        self,
        int block_size=25*(2**20),
        int hash_size=16,
        int min_len=128, 
        int sorter=LIBBSC_DEFAULT_BLOCKSORTER,
        int coder=LIBBSC_DEFAULT_CODER
    ):
        self._block_size = block_size
        self._hash_size = hash_size
        self._min_len = min_len
        self._sorter = sorter
        self._coder = coder

        status = bsc_init(LIBBSC_DEFAULT_FEATURES)
        if status != LIBBSC_NO_ERROR:
            raise ERROR_CODE_TO_EXCEPTION[status]

        self._buffer = b''
        self._cbuffer = NULL

    def compress(self, bytes data):
        """
        Compress `data`, returning a bytes object containing compressed data 
        for at least part of the data in `data`. Some input may be kept in 
        internal buffer for later processing.

        Parameter
        ---------
        data : bytes
            Byte array to compress. 

        Return
        ------
        TBA 
        """
        cdef bytes cdata = b''
        self._buffer += data

        cdef uchar *cdata_buf = <uchar *>PyMem_Malloc(
            sizeof(uchar) * (self._block_size+LIBBSC_HEADER_SIZE)
        )

        cdef int status
        while len(self._buffer) >= self._block_size:
            cdata += self._compress(self._buffer[:self._block_size])
            #EXPAND
            memcpy(
                cdata_buf, 
                self._buffer[:self._block_size], 
                self._block_size
            )
            status = bsc_compress(
                cdata_buf, 
                cdata_buf, 
                self._block_size,
                LIBBSC_DEFAULT_FEATURES
            )
            if status == LIBBSC_NOT_COMPRESSIBLE:
                memcpy(
                    cdata_buf, 
                    self._buffer[:self._block_size], 
                    self._block_size
                )
                status = bsc_store(
                    cdata_buf, 
                    cdata_buf, 
                    self._block_size, 
                    LIBBSC_DEFAULT_FEATURES
                )
            if status != LIBBSC_NO_ERROR:
                PyMem_Free(cdata_buf)
                raise ERROR_CODE_TO_EXCEPTION[status]
            
            # store compressed result
            cdata += cdata_buf[:status]
            # trim processed data
            self._buffer = self._buffer[self._block_size:]

        PyMem_Free(cdata_buf)

        return cdata

    def flush(self):
        """All pending input is processed."""
        cdef bytes cdata = 
    
    def copy(self):
        """Returns a copy of the compression object."""
        pass

def compressobj(*args, **kwargs):
    return Compress(*args, **kwargs)

def compress():
    pass

cdef class Decompress:
    def __init__(self):
        pass

    @property
    def unused_data(self):
        pass
    
    @property
    def unconsumed_tail(self):
        pass
    
    @property
    def eof(self):
        pass
    
    def decompress(self):
        pass
    
    def flush(self):
        pass
    
    def copy(self):
        pass

def decompressobj(*args, **kwargs):
    return Decompress(*args, **kwargs)

def decompress():
    pass
