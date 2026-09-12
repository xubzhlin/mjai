# Auto-import the native extension module that lives alongside this file.
from . import mjai_engine as _native
from .mjai_engine import *

__doc__ = _native.__doc__
if hasattr(_native, "__all__"):
    __all__ = _native.__all__
