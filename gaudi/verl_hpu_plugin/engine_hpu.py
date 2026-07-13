"""Register verl's FSDP engines for ``device='hpu'``.

verl declares its FSDP engines for cuda/npu only::

    @EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"],
                             device=["cuda", "npu"])

so ``main_ppo`` on Gaudi fails with::

    ValueError: No engine registered for device='hpu' ... backend='fsdp'

The engines themselves are device-agnostic — they go through
``verl.utils.device``, which resolves via the platform registry — so the same
classes can simply be registered again under the hpu key.
"""

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, FSDPEngineWithValueHead

EngineRegistry.register(
    model_type="language_model",
    backend=["fsdp", "fsdp2"],
    device="hpu",
)(FSDPEngineWithLMHead)

EngineRegistry.register(
    model_type="value_model",
    backend=["fsdp", "fsdp2"],
    device="hpu",
)(FSDPEngineWithValueHead)
