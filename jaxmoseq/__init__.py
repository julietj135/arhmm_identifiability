import jax


# use double-precision by default
from jax import config

config.update("jax_enable_x64", True)

from . import models
from . import utils
from . import _version

__version__ = _version.get_versions()["version"]


