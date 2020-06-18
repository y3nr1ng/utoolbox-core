"""
Using ctypes to wrap WinAPI functions that interact with console modes.

Reference:
    How to enable Windows console QuickEdit Mode from python
    https://stackoverflow.com/a/37505496
"""
import ctypes
import logging
import msvcrt
import os
from ctypes import wintypes
from enum import IntFlag
from typing import Optional

__all__ = ["get_console_mode", "set_console_mode", "console_mode"]

logger = logging.getLogger("utoolbox.util.win")


class ConsoleIOFlag(IntFlag):
    """Base class for consoole mode flags."""


class ConsoleInputFlag(ConsoleIOFlag):
    ProcessedInput = 0x01
    LineInput = 0x02
    EchoInput = 0x04
    WindowInput = 0x08
    MouseInput = 0x10
    InsertMode = 0x20
    QuickEditMode = 0x40
    ExtendedFlags = 0x80


class ConsoleOutputFlag(ConsoleIOFlag):
    ProcessedOutput = 0x01
    WrapAtEOL = 0x02
    VT100 = 0x04  # Win10 feature


def check_zero(result, func, args):
    if not result:
        err = ctypes.get_last_error()
        if err:
            raise ctypes.WinError(err)
    return args


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

kernel32.GetConsoleMode.errcheck = check_zero
kernel32.GetConsoleMode.argtypes = (
    wintypes.HANDLE,  # _In_  HANDLE  hConsoleHandle
    wintypes.LPDWORD,  # _Out_ LPDWORD lpMode
)

kernel32.SetConsoleMode.errcheck = check_zero
kernel32.SetConsoleMode.argtypes = (
    wintypes.HANDLE,  # _In_ HANDLE hConsoleHandle,
    wintypes.DWORD,  # _In_ DWORD  dwMode
)


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


def get_console_mode(output: bool = False):
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
        value = mode.value
    return ConsoleOutputFlag(value) if output else ConsoleInputFlag(value)


def set_console_mode(mode: ConsoleIOFlag, output: bool = False):
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


class console_mode:
    """
    This class provides a context managed console mode environment, ensure that flags are restored to its original state when exiting the scope.

    Args:
        flags (IntFlag): flags wanted
        mask (IntFlag): flags to remove (masked)
        output (bool, optional): select console output if True
    """

    def __init__(
        self,
        *,
        flags: Optional[ConsoleIOFlag] = 0,
        mask: Optional[ConsoleIOFlag] = 0,
        output: bool = False,
    ):
        self._flags, self._mask = flags, mask

        self._output = output

    def __enter__(self):
        current_mode = get_console_mode(self._output)

        # modify console mode if required
        target_mode = current_mode | self._flags & ~self._mask
        if current_mode != target_mode:
            logger.debug(
                f"modify console mode from {current_mode:#04x} to {target_mode:#04x}"
            )
            set_console_mode(target_mode, output=self._output)
        else:
            # we use None to hint that no need to restore console mode
            current_mode = None
        self._current_mode = current_mode

        return self

    def __exit__(self, *exc):
        if self._current_mode is not None:
            logger.debug(f"reset console mode to {self._current_mode:#04x}")
            set_console_mode(self._current_mode, output=self._output)


if __name__ == "__main__":
    import time
    import coloredlogs

    coloredlogs.install(
        level="DEBUG", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    with console_mode(
        flags=ConsoleInputFlag.InsertMode | ConsoleInputFlag.QuickEditMode
    ):
        time.sleep(5)  # check console properties
