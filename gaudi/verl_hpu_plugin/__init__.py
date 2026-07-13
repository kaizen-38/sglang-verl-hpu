"""verl plugin adding Intel Gaudi (HPU) support.

Activated by verl at import time via::

    VERL_USE_EXTERNAL_MODULES=verl_hpu_plugin
    VERL_PLATFORM=hpu

Importing this package registers the HPU platform and re-registers verl's FSDP
engines under ``device='hpu'``. Nothing in verl's source tree is modified.
"""

from . import engine_hpu  # noqa: F401  (import for the registration side effect)
from .platform_hpu import PlatformHPU

__all__ = ["PlatformHPU"]
