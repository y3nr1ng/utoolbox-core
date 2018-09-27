import logging
import sys

logger = logging.getLogger(__name__)

if not sys.platform.startswith('darwin'):
    raise RuntimeError("unable to import on non-macOS system")

import objc
from Foundation import *
from AppKit import *
from PyObjCTools import AppHelper

__all__ = [
    'prelaunch_cocoa'
]

class AppDelegate(NSObject):
    def init(self):
        self = objc.super(AppDelegate, self).init()
        if self is None:
            return None

        self.wrapped_func = None
        self.args, self.kwargs = None, None

        # NOTE initializers must return self by pyobjc doc
        return self

    def applicationDidFinishLaunching_(self, notification):
        self.performSelectorInBackground_withObject_("runWrappedFunc:", 0)

    def runWrappedFunc_(self, arg):
        self.wrapped_func()
        # terminate explicitly, or it hangs when the wrapped code exits
        NSApp().terminate_(self)

class ConcreteAppDelegate(AppDelegate):
    def initWrappedFunc_(self, wrapped_func):
        self = objc.super(ConcreteAppDelegate, self).init()
        if self is None:
            return None

        self.wrapped_func = wrapped_func
        # NOTE these are not expanded
#        self.args = args
#        self.kwargs = kwargs

        return self

def prelaunch_cocoa(func):
    def _launcher(*args, **kwargs):
        app = NSApplication.sharedApplication()

        # wrap the function in a delegate
        delegate = ConcreteAppDelegate.alloc().initWrappedFunc_(func)
        NSApp().setDelegate_(delegate)

        # sent keyboard events to the UI
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)

        AppHelper.runEventLoop()
    return _launcher
