"""
The Cocoa event loop needs to be started before the JVM is loaded, in order for
Java's AWT to run properly.

Reference
---------
[1] kivy/pyjnius, importing AWT classes hangs the script on OSX #15
    https://github.com/kivy/pyjnius/issues/151
[2] https://github.com/imglib/imglyb/blob/master/imglyb/OSXAWTwrapper.py
"""
import logging
import sys

logger = logging.getLogger(__name__)

__all__ = ["prelaunch_cocoa"]

# TODO wrap with stub
if not sys.platform.startswith("darwin"):
    raise RuntimeError("unable to import on non-macOS system")

import objc
from Foundation import *
from AppKit import *
from PyObjCTools import AppHelper


class AppDelegate(NSObject):
    def initWrappedFunc_(self, wrapped_func):
        self = objc.super(AppDelegate, self).init()
        if self is None:
            return None

        self.wrapped_func = wrapped_func

        # NOTE initializers must return self by pyobjc doc
        return self

    def applicationDidFinishLaunching_(self, notification):
        self.performSelectorInBackground_withObject_("runWrappedFunc:", None)

    def runWrappedFunc_(self, args):
        self.wrapped_func()
        # terminate explicitly, or it hangs when the wrapped code exits
        NSApp().terminate_(self)


def prelaunch_cocoa(func):
    def _launcher(*args, **kwargs):
        app = NSApplication.sharedApplication()

        # wrap the function in a delegate
        def _func():
            func(*args, **kwargs)

        delegate = AppDelegate.alloc().initWrappedFunc_(_func)
        NSApp().setDelegate_(delegate)

        # sent keyboard events to the UI
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)

        AppHelper.runEventLoop()

    return _launcher
