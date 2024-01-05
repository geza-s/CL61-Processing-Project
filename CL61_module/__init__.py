# CL61_module/__init__.py

# Import modules to expose them at the package level
from .module import CL61Processor  # Exposing class from the submodule
from . import process # submodule to access main processing submodule
from . import visualization  # submodule to access noise processing functions

# To automated processing use
from .module import parse_arguments, validate_data_folder, validate_datetime

# Other submodules to directly access functions of interest
from . import classification