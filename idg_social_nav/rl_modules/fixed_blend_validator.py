"""Fixed-probability blend validator (the fixed-gamma baseline).

Whenever the advisor actually suggested something (advice != NONE), 
it overrides with a fixed probability p, otherwise it obeys. 
Sweeping p in [0, 1] traces the naive obedience/deference trade-off curve, 
so that we compare the learned and oracle validators against this baseline.

"""

from __future__ import annotations

from typing import Any

import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func

from idg_social_nav.core import Advice, ValidatorAction, to_numpy


class FixedBlendValidatorRLM(RLModule):

    DEFAULT_P = 0.5
    DEFAULT_SEED = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.model_config or {}
        self._p = float(cfg.get("blend_p", self.DEFAULT_P))
        self._rng = np.random.default_rng(cfg.get("seed", self.DEFAULT_SEED))

    @override(RLModule)
    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        advice = batch[SampleBatch.OBS]["advice"]
        actions = []
        for i in range(len(advice)):
            advice_idx = int(to_numpy(advice[i]).argmax())
            if advice_idx != Advice.NONE and self._rng.random() < self._p:
                actions.append(ValidatorAction.override.value)
            else:
                actions.append(ValidatorAction.obey.value)
        return {SampleBatch.ACTIONS: batch_func(actions)}


def make_fixed_blend_class(p: float, seed: int = 0) -> type:
    """Subclass of FixedBlendValidatorRLM with baked-in defaults (mirrors
    the old run_all_eval._make_llm_validator_class) so eval can sweep p."""
    slug = f"{p:g}".replace(".", "_").replace("-", "m")
    return type(
        f"FixedBlendValidatorRLM_p{slug}_seed{seed}",
        (FixedBlendValidatorRLM,),
        {"DEFAULT_P": float(p), "DEFAULT_SEED": int(seed)},
    )
