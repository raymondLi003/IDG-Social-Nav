"""Experiment configuration: policy registry, model configs, and the RLlib
(new API stack) builders for training the learned validator.

This script trains PPO only 
"""

from collections.abc import Hashable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

from ray.rllib.algorithms import AlgorithmConfig, PPOConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

from idg_social_nav.rl_modules.catalog.catalog import PPOCatalogWithImageActionEncoder

TRAINING_ITERATIONS = 300
EVAL_REPS_STOCHASTIC = 5


class ProposerPolicies(StrEnum):
    SCRIPTED = "scripted_proposer"
    LEARNED = "learned_proposer"


class ValidatorPolicies(StrEnum):
    LEARNED = "learned_validator"
    ORACLE = "oracle_validator"
    ALWAYS_OBEY = "always_obey_validator"
    FIXED_BLEND = "fixed_blend_validator"


@dataclass(frozen=True)
class SocialAgentConfig:
    proposer_policy: ProposerPolicies = ProposerPolicies.SCRIPTED
    validator_policy: ValidatorPolicies = ValidatorPolicies.LEARNED
    algorithm_name: str = "ppo"
    scenario: str = "all"
    reward_variant: str = "binary"
    override_semantics: str = "adopt"


def experiment_name(cfg: SocialAgentConfig) -> str:
    return (
        f"{cfg.algorithm_name}"
        f"_{cfg.proposer_policy}_{cfg.validator_policy}"
        f"__{cfg.scenario}__{cfg.reward_variant}"
    )


DEFAULT_CONV_MODEL_CONFIG = {
    "conv_filters": [
        [16, 3, 1],
        [32, 3, 1],
    ],
    "conv_activation": "relu",
    "head_fcnet_hiddens": [64],
    "fcnet_activation": "relu",
}

DEFAULT_MULTI_AGENT_MODEL_CONFIG = {
    "dict_encoder_config": {
        "cnn_config_dict": DEFAULT_CONV_MODEL_CONFIG.copy(),
        "mlp_config_dict": {
            "fcnet_hiddens": [8],
        },
    },
    "head_fcnet_hiddens": [64],
    "fcnet_activation": "relu",
}


# build the ppo algo
def _ppo_algorithm_config(learns_validator: bool) -> AlgorithmConfig:
    if learns_validator:
        return PPOConfig().training(
            lr=2.8e-4,
            gamma=0.95,
            entropy_coeff=0.01,
            clip_param=0.2,
            num_epochs=15,
            train_batch_size=512,
        )
    return PPOConfig().training(
        entropy_coeff=[
            (0, 0.2),
            (200_000, 0.05),
            (800_000, 0.005),
        ],
        train_batch_size=512,
    )


def env_config_for(cfg: SocialAgentConfig) -> dict:
    """SocialNavEnv kwargs for a training and eval run of this agent config."""
    return {
        "scenario": cfg.scenario,
        "reward_variant": cfg.reward_variant,
        "override_semantics": cfg.override_semantics,
        "randomize_variant": True,
    }


def register_envs(cfg: SocialAgentConfig) -> None:
    """Register "env" and "eval_env" with the Ray Tune registry
    """
    from ray.tune.registry import register_env

    from idg_social_nav.env import SocialNavEnv

    base = env_config_for(cfg)
    register_env("env", lambda ecfg, base=base: SocialNavEnv(**{**base, **(ecfg or {})}))
    register_env("eval_env", lambda ecfg, base=base: SocialNavEnv(**{**base, **(ecfg or {})}))


def add_env_config(config: AlgorithmConfig) -> AlgorithmConfig:
    config.environment("env")
    # complete episodes are needed for the turn-based env
    config.env_runners(
        batch_mode="complete_episodes",
        num_env_runners=6,
        num_cpus_per_env_runner=1,
    )
    return config


def agent_config_policy_mapping(
        agent_id: Hashable,
        episode: MultiAgentEpisode,
        agent_config: SocialAgentConfig,
) -> str:
    if agent_id == "proposer":
        return agent_config.proposer_policy
    if agent_id == "validator":
        return agent_config.validator_policy
    raise ValueError(f"Invalid agent: {agent_id}")


def get_multi_agent_rl_module_specs(
        policy_names: list[str],
        agent_config: SocialAgentConfig,
) -> dict[str, RLModuleSpec]:
    rl_module_specs = {}
    if ProposerPolicies.LEARNED in policy_names:
        rl_module_specs[ProposerPolicies.LEARNED] = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            model_config=DEFAULT_MULTI_AGENT_MODEL_CONFIG,
            catalog_class=PPOCatalogWithImageActionEncoder,
        )

    if ProposerPolicies.SCRIPTED in policy_names:
        from idg_social_nav.rl_modules.shortest_path_proposer import (
            ShortestPathProposerRLM,
        )
        rl_module_specs[ProposerPolicies.SCRIPTED] = RLModuleSpec(
            module_class=ShortestPathProposerRLM,
            inference_only=True,
        )

    if ValidatorPolicies.LEARNED in policy_names:
        rl_module_specs[ValidatorPolicies.LEARNED] = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            model_config=DEFAULT_MULTI_AGENT_MODEL_CONFIG,
            catalog_class=PPOCatalogWithImageActionEncoder,
        )

    if ValidatorPolicies.ORACLE in policy_names:
        from idg_social_nav.rl_modules.oracle_validator import OracleValidatorRLM
        rl_module_specs[ValidatorPolicies.ORACLE] = RLModuleSpec(
            module_class=OracleValidatorRLM,
            inference_only=True,
        )

    if ValidatorPolicies.ALWAYS_OBEY in policy_names:
        from idg_social_nav.rl_modules.always_obey_validator import (
            AlwaysObeyValidatorRLM,
        )
        rl_module_specs[ValidatorPolicies.ALWAYS_OBEY] = RLModuleSpec(
            module_class=AlwaysObeyValidatorRLM,
            inference_only=True,
        )

    if ValidatorPolicies.FIXED_BLEND in policy_names:
        from idg_social_nav.rl_modules.fixed_blend_validator import (
            FixedBlendValidatorRLM,
        )
        rl_module_specs[ValidatorPolicies.FIXED_BLEND] = RLModuleSpec(
            module_class=FixedBlendValidatorRLM,
            inference_only=True,
        )

    return rl_module_specs


def add_multi_agent_policies(
        config: AlgorithmConfig,
        agent_config: SocialAgentConfig,
) -> AlgorithmConfig:
    policies = [agent_config.proposer_policy, agent_config.validator_policy]
    policy_mapping_fn = partial(agent_config_policy_mapping, agent_config=agent_config)
    policies_to_train = []
    if agent_config.proposer_policy == ProposerPolicies.LEARNED:
        policies_to_train.append(ProposerPolicies.LEARNED)
    if agent_config.validator_policy == ValidatorPolicies.LEARNED:
        policies_to_train.append(ValidatorPolicies.LEARNED)

    return (
        config
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs=get_multi_agent_rl_module_specs(policies, agent_config),
            )
        )
    )


def create_rllib_config(cfg: SocialAgentConfig) -> AlgorithmConfig:
    if cfg.algorithm_name != "ppo":
        raise ValueError(
            f"Only ppo is supported (got {cfg.algorithm_name!r})")
    learns_validator = cfg.validator_policy == ValidatorPolicies.LEARNED
    config = _ppo_algorithm_config(learns_validator).framework("torch")
    config = add_env_config(config)
    config = add_multi_agent_policies(config, cfg)
    config.validate()
    return config
