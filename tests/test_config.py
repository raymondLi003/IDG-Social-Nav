"""Tests for config utilities (config.py)."""


import numpy as np
import pytest
import torch
import tree
from ray.rllib import SampleBatch
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.rl_module import RLModuleSpec

from idg_social_nav.config import (
    DEFAULT_MULTI_AGENT_MODEL_CONFIG,
    ProposerPolicies,
    SocialAgentConfig,
    ValidatorPolicies,
    create_rllib_config,
    env_config_for,
)
from idg_social_nav.core import ProposerAction, ValidatorAction
from idg_social_nav.env import SocialNavEnv
from idg_social_nav.rl_modules.catalog.catalog import PPOCatalogWithImageActionEncoder


class TestCreateRLlibConfig:
    def test_default_ppo_config_validates(self):
        config = create_rllib_config(SocialAgentConfig())
        assert isinstance(config, AlgorithmConfig)
        assert set(config.policies) == {
            ProposerPolicies.SCRIPTED, ValidatorPolicies.LEARNED}
        assert ValidatorPolicies.LEARNED in config.policies_to_train
        assert ProposerPolicies.SCRIPTED not in config.policies_to_train

    def test_non_ppo_algorithm_raises(self):
        with pytest.raises(ValueError):
            create_rllib_config(SocialAgentConfig(algorithm_name="dqn"))


class TestEnvConfigFor:
    def test_default_kwargs(self):
        cfg = env_config_for(SocialAgentConfig())
        assert cfg == {
            "scenario": "all",
            "reward_variant": "binary",
            "override_semantics": "adopt",
            "randomize_variant": True,
        }

    def test_kwargs_construct_an_env(self):
        env = SocialNavEnv(**env_config_for(
            SocialAgentConfig(scenario="narrow_doorway")))
        obs, _ = env.reset(seed=0)
        assert "proposer" in obs


class TestLearnedValidatorModule:
    def test_dict_encoder_forward_inference(self):
        env = SocialNavEnv(scenario="frontal_approach", randomize_variant=False)
        module = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=env.observation_spaces["validator"],
            action_space=env.action_spaces["validator"],
            model_config=dict(DEFAULT_MULTI_AGENT_MODEL_CONFIG),
            catalog_class=PPOCatalogWithImageActionEncoder,
        ).build()

        env.reset(options={
            "scenario": "frontal_approach",
            "variant": {"ped_start_col": 7, "ped_delay": 0},
        })
        obs, *_ = env.step({"proposer": int(ProposerAction.forward)})
        batch = {SampleBatch.OBS: tree.map_structure(
            lambda x: torch.tensor(np.expand_dims(x, axis=0)),
            obs["validator"])}

        out = module.forward_inference(batch)
        logits = out[SampleBatch.ACTION_DIST_INPUTS]
        assert logits.shape == (1, len(ValidatorAction))
        assert torch.isfinite(logits).all()
