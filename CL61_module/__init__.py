# CL61_module/__init__.py

# Import modules to expose them at the package level

from .module import CL61Processor  # Exposing class from the submodule
from . import noise_processing # submodule to access noise processing functions
from . import visualization  # submodule to access noise processing functions
from . import classification # submodule to access classification functions