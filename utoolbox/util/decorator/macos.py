import logging
import sys

logger = logging.getLogger(__name__)

if not sys.platform.sartswith('darwin'):
    raise RuntimeError("unable to import on non-macOS system")

import objc
from Foundation import *
from AppKit import *
from PyObjCTools import AppHelper

__all__ = [
    'prelaunch_cocoa'
]

class AppDelegate(NSObject):
    def init(self, wrapped_func, *args, **kwargs):
        """Designated initializer using NSObject."""
        self = objc.super(AppDelegate, self).init()
        if self is None:
            return None

        #TODO rework the wrapper
        self.wrapped_func = wrapped_func
        self.args = args
        self.kwargs = kwargs

        # initializers must return self by pyobjc doc
        return self

    def applicationDidFinishLaunching_(self, notification):
        self.performSelectorInBackground_withObject_("run_wrapped_func:", 0)

    def run_wrapped_func_(self, arg):
        self.wrapped_func(*self.args, **self.kwargs)
        # terminate explicitly, or it hangs when the wrapped code exits
        NSApp().terminate_(self)

def prelaunch_cocoa(func):
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init(func)
    NSApp().setDelegate_(delegate)

    # sent keyboard events to the UI
    NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    AppHelper.runEventLoop()
