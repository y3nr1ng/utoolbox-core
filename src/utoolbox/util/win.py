"""
Using ctypes to wrap WinAPI functions that interact with console modes.

Reference:
    How to enable Windows console QuickEdit Mode from python
    https://stackoverflow.com/a/37505496
"""
import atexit
import ctypes
import msvcrt
import os
from ctypes import wintypes
from enum import IntFlag

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# input flags
class ConsoleInputFlag(IntFlag):
    ProcessedInput = 0x01
    LineInput = 0x02
    EchoInput = 0x04
    WindowInput = 0x08
    MouseInput = 0x10
    InsertMode = 0x20
    QuickEditMode = 0x40
    ExtendedFlags = 0x80


ENABLE_PROCESSED_INPUT = 0x0001
ENABLE_LINE_INPUT = 0x0002
ENABLE_ECHO_INPUT = 0x0004
ENABLE_WINDOW_INPUT = 0x0008
ENABLE_MOUSE_INPUT = 0x0010
ENABLE_INSERT_MODE = 0x0020
ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080

# output flags
class ConsoleOutputFlag(IntFlag):
    ProcessedOutput = 0x01
    WrapAtEOL = 0x02
    VT100 = 0x04


ENABLE_PROCESSED_OUTPUT = 0x0001
ENABLE_WRAP_AT_EOL_OUTPUT = 0x0002
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004  # VT100 (Win 10)


def check_zero(result, func, args):
    if not result:
        err = ctypes.get_last_error()
        if err:
            raise ctypes.WinError(err)
    return args


kernel32.GetConsoleMode.errcheck = check_zero
kernel32.GetConsoleMode.argtypes = (wintypes.HANDLE, wintypes.LPDWORD)

kernel32.SetConsoleMode.errcheck = check_zero
kernel32.SetConsoleMode.argtypes = (wintypes.HANDLE, wintypes.DWORD)


class Console:
    def __init__(self, filename, *, read_only=True):
        rw_mode = os.O_RDONLY if read_only else os.O_RDWR
        self.fd = os.open(filename, rw_mode | os.O_BINARY)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        os.close(self.fd)

    ##

    @property
    def handle(self):
        return msvcrt.get_osfhandle(self.fd)


class ConsoleInput(Console):
    def __init__(self, **kwargs):
        super().__init__("CONIN$", **kwargs)


class ConsoleOutput(Console):
    def __init__(self, **kwargs):
        super().__init__("CONOUT$", **kwargs)


def get_console_mode(output=False):
    """
    Get the mode of the active console input or output　buffer. 
    
    Ｉf the process isn't attached to a　console, this function raises an EBADF IOError.

    Args:
        output (bool, optional)
    """
    con_klass = ConsoleOutput if output else ConsoleInput
    with con_klass() as con:
        mode = wintypes.DWORD()
        kernel32.GetConsoleMode(con.handle, ctypes.byref(mode))
        return mode.value


def set_console_mode(mode, output=False):
    """
    Set the mode of the active console input or output　buffer. 
    
    If the process isn't attached to a console, this function raises an EBADF IOError.

    Args:
        mode
        output (bool, optional)
    """
    con_klass = ConsoleOutput if output else ConsoleInput
    with con_klass(read_only=False) as con:
        kernel32.SetConsoleMode(con.handle, mode)


def update_console_mode(flags: IntFlag, mask: IntFlag, output=False, restore=False):
    """
    Update a masked subset of the current mode of the active console input or output 
    buffer. 

    If the process isn't attached to a console, this function raises an EBADF IOError.

    Args:
        flags (IntFlag): flags wanted
        mask (IntFlag): flags to remove (masked)
        output (bool, optional): select console output if True
        restore (bool, optional): restore console mode after termination
    
    TODO restore, atexit not working
    """
    current_mode = get_console_mode(output)
    if current_mode & mask != flags & mask:
        mode = current_mode & ~mask | flags & mask
        set_console_mode(mode, output)
    else:
        restore = False
    if restore:
        atexit.register(set_console_mode, current_mode, output)


if __name__ == "__main__":
    import time

    print("%#06x, %#06x" % (get_console_mode(), get_console_mode(output=True)))

    flags = mask = ENABLE_EXTENDED_FLAGS | ENABLE_INSERT_MODE
    update_console_mode(flags, mask, restore=True)

    print("%#06x, %#06x" % (get_console_mode(), get_console_mode(output=True)))

    time.sleep(10)  # check console properties
