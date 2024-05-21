from importlib.resources import files as _files

import matplotlib.pyplot

from .binary_classifier import *
from .multi_classifier import *
from .timeseries import *

matplotlib.pyplot.style.use(_files(__package__) / "ux.mplstyle")
