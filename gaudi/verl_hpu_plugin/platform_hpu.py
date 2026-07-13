"""Intel Gaudi (HPU) platform backend for verl.

Loaded through ``VERL_USE_EXTERNAL_MODULES=verl_hpu_plugin`` so that verl's
device abstraction resolves to Gaudi without patching the verl source tree.
Select it explicitly with ``VERL_PLATFORM=hpu``.
"""

import logging
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)


def _ensure_torch_hpu() -> bool:
    """Import habana_frameworks.torch, which grafts the ``torch.hpu`` namespace on."""
    if hasattr(torch, "hpu"):
        return True
    try:
        import habana_frameworks.torch  # noqa: F401

        return hasattr(torch, "hpu")
    except Exception as e:
        logger.debug("The current machine has no torch.hpu, because: %s", e)
    return False


_ensure_torch_hpu()


@PlatformRegistry.register(platform="hpu")
class PlatformHPU(PlatformBase):
    """Platform backend for Intel Gaudi (Habana HPU)."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "hpu"

    @property
    def vendor_name(self) -> str:
        return "intel"

    @property
    def device_module(self) -> ModuleType:
        return torch.hpu

    def is_available(self) -> bool:
        return _ensure_torch_hpu() and torch.hpu.is_available()

    def is_platform_available(self, use_smi_check=False) -> bool:
        if not _ensure_torch_hpu():
            return False
        if use_smi_check:
            # Ray actors without device visibility still live on a Gaudi host;
            # importability of habana_frameworks is the strongest signal available.
            return True
        return torch.hpu.is_available()

    def current_device(self) -> int:
        return torch.hpu.current_device()

    def device_count(self) -> int:
        return torch.hpu.device_count()

    def set_device(self, device_index: int) -> None:
        torch.hpu.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.hpu.synchronize()

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.hpu.random.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.hpu.random.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        # Gaudi's allocator is configured through PT_HPU_* env vars at process
        # start; there is no runtime equivalent of _set_allocator_settings.
        logger.debug("set_allocator_settings is a no-op on HPU (requested: %s)", settings)

    def empty_cache(self) -> None:
        torch.hpu.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        return "hccl"

    def visible_devices_envvar(self) -> str:
        return "HABANA_VISIBLE_MODULES"

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        return "HPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"resources": {"HPU": num_gpus}}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES"]

    def rollout_env_vars(self) -> dict[str, str]:
        return {"PT_HPU_LAZY_MODE": os.getenv("PT_HPU_LAZY_MODE", "0")}

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        # Synapse exposes no device-to-device IPC handle equivalent to cudaIpc*.
        # Weight transfer to a disaggregated rollout engine must go through the
        # filesystem (update_weights_from_disk).
        return False

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        logger.debug("NVTX range (no-op on HPU): %s", msg)
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        return None
