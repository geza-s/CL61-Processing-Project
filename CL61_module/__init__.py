# CL61_module/__init__.py

# Import modules to expose them at the package level

from .module import CL61Processor  # Exposing class from the submodule
from . import process # submodule to access main processing submodule
from . import visualization  # submodule to access noise processing functions

# Other submodules to directly access functions of interest
from . import classification
from . import process_noise