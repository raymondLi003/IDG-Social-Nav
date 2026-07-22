"""Leader-follower social-navigation grid environment.

Architecture:
  1. Leader proposes: the goal planner proposes forward/turn_left/turn_right.
     It does not see the social-hazard (discomfort) channel.
  2. Advisor is queried only when a pedestrian is inside the
     validator's egocentric view
  3. Validator sees its egocentric view with the
     the discomfort field and the advice as a one-hot, and then
     outputs obey or override. if obey we execute the leader's action, otherwise override
     executes B_h (or waits when B_h says stop).
  4. Scoring: the the validator reward: good/bad disobedience, missed hazards, with an
     optional graded option proportional to discomfort avoided.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from idg_social_nav.core import (
    CH_DISCOMFORT,
    DIR_OFFSET,
    N_PROPOSER_CHANNELS,
    N_VALIDATOR_CHANNELS,
    OVERRIDE_PROTOCOLS,
    PROPOSER_TO_ENV_ACTION,
    Advice,
    Advisor,
    AdvisorContext,
    EnvironmentAction,
    Gesture,
    PedestrianSnapshot,
    ProposerAction,
    ValidatorAction,
    nearest_gesture,
    ped_channel_value,
)
from idg_social_nav.scenarios import (
    SCENARIO_NAMES,
    VIEW_RADIUS,
    PedestrianState,
    ScenarioConfig,
    enumerate_variants,
    make_scenario,
)


def egocentric_view(
        walls: np.ndarray,
        agent_pos: tuple[int, int],
        agent_dir: int,
        layers: np.ndarray,
        view_radius: int,
) -> np.ndarray:
    """Egocentric cropped view, agent at bottom-center facing up.

    walls: (H, W) binary grid
    layers: (H, W, K) float grids overlaid as channels 1..K. 
    Returns (view_radius+1, 2*view_radius+1, 1+K)
    Cells outside the grid read as walls.
    """
    pad = view_radius
    wall_padded = np.pad(
        walls.astype(np.float32), ((pad, pad), (pad, pad)),
        mode="constant", constant_values=1.0,
    )
    layers_padded = np.pad(
        layers.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)),
        mode="constant", constant_values=0.0,
    )
    padded = np.concatenate((wall_padded[..., None], layers_padded), axis=-1)

    row = agent_pos[0] + pad
    col = agent_pos[1] + pad
    local = padded[row - pad: row + pad + 1, col - pad: col + pad + 1, :]

    # rotate so the agent always faces up
    local = np.rot90(local, k=agent_dir, axes=(0, 1))

    # keep only the agent's row and everything ahead
    center = local.shape[1] // 2
    local = local[0: pad + 1, center - pad: center + pad + 1, :]

    return local.astype(np.float32).copy()


class SocialNavEnv(MultiAgentEnv):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(
            self,
            scenario: str | list[str] = "frontal_approach",
            variant: dict | None = None,
            advisor: Advisor | None = None,
            reward_variant: str = "binary",
            override_semantics: str = "adopt",
            proposer_sees_discomfort: bool = False,
            advice_gating: bool = True,
            randomize_variant: bool = True,
            max_steps: int | None = None,
            step_penalty: float = 0.0,
            record_render: bool = False,
            seed=None,
    ):
        super().__init__()
        if scenario == "all":
            self._scenario_names = list(SCENARIO_NAMES)
        elif isinstance(scenario, str):
            self._scenario_names = [scenario]
        else:
            self._scenario_names = list(scenario)
        for name in self._scenario_names:
            if name not in SCENARIO_NAMES:
                raise ValueError(f"Unknown scenario: {name!r}")

        assert reward_variant in ("binary", "graded")
        self.reward_variant = reward_variant
        self._operation_protocol = OVERRIDE_PROTOCOLS[override_semantics]
        self.override_semantics = override_semantics
        self.proposer_sees_discomfort = proposer_sees_discomfort
        self.advice_gating = advice_gating
        self.randomize_variant = randomize_variant
        self._fixed_variant = variant
        self._max_steps_override = max_steps
        self.step_penalty = step_penalty
        self.rng = np.random.default_rng(seed)

        if advisor is None:
            from idg_social_nav.advisor_scripted import ScriptedSocialAdvisor
            advisor = ScriptedSocialAdvisor()
        self.advisor = advisor

        self.agents = self.possible_agents = ["proposer", "validator"]

        self.view_radius = VIEW_RADIUS
        view_shape = (self.view_radius + 1, 2 * self.view_radius + 1)
        proposer_channels = (
            N_VALIDATOR_CHANNELS if proposer_sees_discomfort else N_PROPOSER_CHANNELS
        )
        self.action_spaces = {
            "proposer": gym.spaces.Discrete(len(ProposerAction)),
            "validator": gym.spaces.Discrete(len(ValidatorAction)),
        }
        self.observation_spaces = {
            "proposer": gym.spaces.Dict({
                "env": gym.spaces.Box(
                    low=0, high=1, shape=(*view_shape, proposer_channels),
                    dtype=np.float32),
                "pose": gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
                "validator_action": gym.spaces.Box(
                    low=0, high=1, shape=(len(ValidatorAction),), dtype=np.float32),
            }),
            "validator": gym.spaces.Dict({
                "env": gym.spaces.Box(
                    low=0, high=1, shape=(*view_shape, N_VALIDATOR_CHANNELS),
                    dtype=np.float32),
                "pose": gym.spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
                "proposer_action": gym.spaces.Box(
                    low=0, high=1, shape=(len(ProposerAction),), dtype=np.float32),
                "advice": gym.spaces.Box(
                    low=0, high=1, shape=(len(Advice),), dtype=np.float32),
                "gesture": gym.spaces.Box(
                    low=0, high=1, shape=(len(Gesture),), dtype=np.float32),
            }),
        }

        # episode state 
        self.scenario: ScenarioConfig | None = None
        self.agent_pos: np.ndarray | None = None
        self.agent_dir: int | None = None
        self.ped_states: list[PedestrianState] = []
        self.field: np.ndarray | None = None
        self.done = False
        self._turns = 0
        self._proposer_action: int | None = None
        self._validator_action: int | None = None
        self.last_advice: Advice = Advice.NONE
        self.last_info: dict = {}

        self._record_render = record_render
        self._frames: list[np.ndarray] = []

    @property
    def walls(self) -> np.ndarray:
        return self.scenario.walls

    @property
    def goal_pos(self) -> tuple[int, int]:
        return self.scenario.goal_pos

    @property
    def max_steps(self) -> int:
        if self._max_steps_override is not None:
            return self._max_steps_override
        return self.scenario.max_steps

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None,
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}

        name = options.get("scenario")
        if name is None:
            idx = int(self.rng.integers(0, len(self._scenario_names)))
            name = self._scenario_names[idx]

        variant = options.get("variant", self._fixed_variant)
        if variant is None and not self.randomize_variant:
            variant = enumerate_variants(name)[0]
        self.scenario = make_scenario(
            name, variant=variant,
            rng=self.rng if variant is None else None,
        )

        self.agent_pos = np.array(self.scenario.agent_start, dtype=np.int32)
        self.agent_dir = int(self.scenario.agent_dir)
        self.ped_states = [
            PedestrianState.from_config(cfg) for cfg in self.scenario.pedestrians
        ]
        self._recompute_field()

        self.done = False
        self._turns = 0
        self._proposer_action = None
        self._validator_action = None
        self.last_advice = Advice.NONE
        self.last_info = {}
        self._frames = []
        self.advisor.reset()

        if self._record_render:
            self._capture_frame()

        validator_one_hot = np.zeros(len(ValidatorAction), dtype=np.float32)
        validator_one_hot[ValidatorAction.obey] = 1.0
        return {
            "proposer": {
                "env": self._env_view("proposer"),
                "pose": self._pose_vector(),
                "validator_action": validator_one_hot,
            }
        }, {}

    def _recompute_field(self) -> None:
        from idg_social_nav.discomfort import discomfort_field
        self.field = discomfort_field(
            self.walls,
            [(p.pos, p.facing, p.gesture) for p in self.ped_states],
            self.scenario.discomfort_params,
        )

    # observations
    def _layer_grids(self) -> np.ndarray:
        """
        (H, W, 4) float layers: agent, goal, pedestrian(facing-coded), discomfort 
        Channel order matches core.CH_* minus the wall channel."""
        h, w = self.walls.shape
        layers = np.zeros((h, w, 4), dtype=np.float32)
        layers[self.agent_pos[0], self.agent_pos[1], 0] = 1.0
        layers[self.goal_pos[0], self.goal_pos[1], 1] = 1.0
        for ped in self.ped_states:
            layers[ped.pos[0], ped.pos[1], 2] = ped_channel_value(
                ped.facing, self.agent_dir)
        layers[:, :, 3] = self.field
        return layers

    def _env_view(self, agent_id: str) -> np.ndarray:
        view = egocentric_view(
            self.walls,
            (int(self.agent_pos[0]), int(self.agent_pos[1])),
            self.agent_dir,
            self._layer_grids(),
            self.view_radius,
        )
        if agent_id == "proposer" and not self.proposer_sees_discomfort:
            view = np.delete(view, CH_DISCOMFORT, axis=-1)
        return view

    def _pose_vector(self) -> np.ndarray:
        h, w = self.walls.shape
        pose = np.zeros(6, dtype=np.float32)
        pose[0] = self.agent_pos[0] / (h - 1)
        pose[1] = self.agent_pos[1] / (w - 1)
        pose[2 + self.agent_dir] = 1.0
        return pose

    def _visible_ped_indices(self) -> list[int]:
        """Pedestrians within detection range

        The gate is proximity-based and not heading based
        the rendered frame that serves as the advisor's camera feed shows the whole scene, 
        so a pedestrian the agent has turned away from is still detected
        """
        visible = []
        pad = self.view_radius
        for i, ped in enumerate(self.ped_states):
            dr = int(ped.pos[0]) - int(self.agent_pos[0])
            dc = int(ped.pos[1]) - int(self.agent_pos[1])
            if max(abs(dr), abs(dc)) <= pad:
                visible.append(i)
        return visible

    def _nearest_visible_gesture(self, visible: list[int]) -> Gesture:
        return nearest_gesture(
            [(tuple(self.ped_states[i].pos), self.ped_states[i].gesture)
             for i in visible],
            (int(self.agent_pos[0]), int(self.agent_pos[1])),
        )


    # advisor
    def _query_advisor(self) -> tuple[Advice, list[int]]:
        visible = self._visible_ped_indices()
        if self.advice_gating and not visible:
            return Advice.NONE, visible
        snapshots = [
            PedestrianSnapshot(
                pos=tuple(p.pos), facing=p.facing, gesture=p.gesture,
                visible=(i in visible),
            )
            for i, p in enumerate(self.ped_states)
        ]
        context = AdvisorContext(
            scenario_name=self.scenario.name,
            walls=self.walls,
            agent_pos=(int(self.agent_pos[0]), int(self.agent_pos[1])),
            agent_dir=self.agent_dir,
            goal_pos=tuple(self.goal_pos),
            pedestrians=snapshots,
            step=self._turns,
            frame_provider=self.render_frame,
        )
        advice = Advice(self.advisor.advise(context))
        return advice, visible

    
    # dynamics and actions
    def _forward_position(self) -> tuple[int, int]:
        dr, dc = DIR_OFFSET[self.agent_dir]
        return (int(self.agent_pos[0]) + dr, int(self.agent_pos[1]) + dc)

    def _resolve_action(
            self, env_action: EnvironmentAction,
    ) -> tuple[tuple[int, int], int, bool]:
        """
        Outcome of an env action from the current state:
        (resulting position, resulting direction, collision_attempt)
        """
        pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        direction = self.agent_dir
        collision = False
        if env_action == EnvironmentAction.TURN_LEFT:
            direction = (direction - 1) % 4
        elif env_action == EnvironmentAction.TURN_RIGHT:
            direction = (direction + 1) % 4
        elif env_action == EnvironmentAction.MOVE_FORWARD:
            fwd = self._forward_position()
            ped_cells = {tuple(p.pos) for p in self.ped_states}
            if fwd in ped_cells:
                collision = True  # blocked by the pedestrian's body
            elif self.walls[fwd[0], fwd[1]] == 0:
                pos = fwd
        return pos, direction, collision

    def _discomfort_of(self, pos: tuple[int, int], collision: bool) -> float:
        if collision:
            return 1.0
        return float(self.field[pos[0], pos[1]])

    
    # turn-based action
    def step(
            self, action_dict: MultiAgentDict,
    ) -> tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        if "proposer" in action_dict:
            obs, rewards, infos = self._step_proposer(int(action_dict["proposer"]))
        elif "validator" in action_dict:
            obs, rewards, infos = self._step_validator(int(action_dict["validator"]))
        else:
            raise ValueError("Invalid action dict.")

        terminated = {"__all__": self.done}
        for agent_id in self.agents:
            terminated[agent_id] = self.done

        truncate = (not self.done) and self._turns >= self.max_steps
        truncated = {"__all__": truncate}
        for agent_id in self.agents:
            truncated[agent_id] = truncate
        if truncate:
            if "proposer" not in obs:
                obs["proposer"] = {
                    "env": self._env_view("proposer"),
                    "pose": self._pose_vector(),
                    "validator_action": np.zeros(len(ValidatorAction), dtype=np.float32),
                }
            if "validator" not in obs:
                obs["validator"] = self._validator_obs(
                    np.zeros(len(ProposerAction), dtype=np.float32),
                    advice=Advice.NONE)

        return obs, rewards, terminated, truncated, infos

    def _validator_obs(self, proposer_one_hot: np.ndarray,
                       advice: Advice | None = None) -> dict:
        advice_one_hot = np.zeros(len(Advice), dtype=np.float32)
        advice_one_hot[self.last_advice if advice is None else advice] = 1.0
        gesture_one_hot = np.zeros(len(Gesture), dtype=np.float32)
        gesture_one_hot[self._nearest_visible_gesture(self._visible_ped_indices())] = 1.0
        return {
            "env": self._env_view("validator"),
            "pose": self._pose_vector(),
            "proposer_action": proposer_one_hot,
            "advice": advice_one_hot,
            "gesture": gesture_one_hot,
        }

    def _step_proposer(self, proposer_action: int):
        self._proposer_action = proposer_action
        self.last_advice, _ = self._query_advisor()

        proposer_one_hot = np.zeros(len(ProposerAction), dtype=np.float32)
        proposer_one_hot[proposer_action] = 1.0
        obs = {"validator": self._validator_obs(proposer_one_hot)}
        return obs, {}, {}

    def _step_validator(self, validator_action: int):
        self._validator_action = validator_action
        advice = self.last_advice
        tau = self.scenario.discomfort_params.high_threshold

        env_action = self._operation_protocol(
            ProposerAction(self._proposer_action),
            ValidatorAction(validator_action),
            advice,
        )
        leader_action = PROPOSER_TO_ENV_ACTION[ProposerAction(self._proposer_action)]
        # gesture that the validator actually saw
        gesture_seen = self._nearest_visible_gesture(self._visible_ped_indices())

        lead_pos, _, lead_collision = self._resolve_action(leader_action)
        exec_pos, exec_dir, exec_collision = self._resolve_action(env_action)
        d_lead = self._discomfort_of(lead_pos, lead_collision)
        d_exec = self._discomfort_of(exec_pos, exec_collision)

        # execute
        self.agent_pos = np.array(exec_pos, dtype=np.int32)
        self.agent_dir = exec_dir

        # proposer reward
        proposer_reward = -self.step_penalty
        if tuple(exec_pos) == tuple(self.goal_pos):
            proposer_reward = 1.0
            self.done = True

        # validator reward
        overridden = validator_action == ValidatorAction.override
        good_override = overridden and d_lead >= tau and d_exec < tau
        bad_override = overridden and d_lead < tau
        failed_override = overridden and d_lead >= tau and d_exec >= tau
        missed_hazard = (not overridden) and d_lead >= tau

        if self.reward_variant == "binary":
            if good_override:
                validator_reward = 1.0
            elif bad_override or missed_hazard:
                validator_reward = -1.0
            else:
                validator_reward = 0.0
        else:  
            # grade on discomfort avoided by the validator's override
            if overridden:
                validator_reward = -1.0 if bad_override else d_lead - d_exec
            else:
                validator_reward = -d_lead if missed_hazard else 0.0

        # pedestrians move after the agent
        ped_cells = {tuple(p.pos) for p in self.ped_states}
        for ped in self.ped_states:
            ped_cells.discard(tuple(ped.pos))
            ped.step(tuple(self.agent_pos), ped_cells)
            ped_cells.add(tuple(ped.pos))
        self._recompute_field()

        # actual discomfort experienced after the full turn (intrusion metric)
        intrusion = float(self.field[self.agent_pos[0], self.agent_pos[1]])

        self._turns += 1
        if self._record_render:
            self._capture_frame()

        info = {
            "turn": self._turns,
            "scenario": self.scenario.name,
            "variant": self.scenario.variant,
            "proposer_action": int(self._proposer_action),
            "validator_action": int(validator_action),
            "advice": int(advice),
            "executed_action": int(env_action),
            "d_lead": d_lead,
            "d_exec": d_exec,
            "overridden": overridden,
            "good_override": good_override,
            "bad_override": bad_override,
            "failed_override": failed_override,
            "missed_hazard": missed_hazard,
            "collision_attempt": exec_collision,
            "intrusion": intrusion,
            "gesture": int(gesture_seen),
            "reached_goal": self.done,
        }
        self.last_info = info

        rewards = {"proposer": proposer_reward, "validator": validator_reward}
        validator_one_hot = np.zeros(len(ValidatorAction), dtype=np.float32)
        validator_one_hot[validator_action] = 1.0
        obs = {
            "proposer": {
                "env": self._env_view("proposer"),
                "pose": self._pose_vector(),
                "validator_action": validator_one_hot,
            }
        }
        infos = {"proposer": dict(info), "validator": dict(info)}
        return obs, rewards, infos

    # rendering
    def render_frame(self) -> np.ndarray:
        from idg_social_nav.render import render_frame
        return render_frame(self)

    def _capture_frame(self) -> None:
        try:
            self._frames.append(self.render_frame())
        except Exception:
            pass

    def save_video(self, path) -> None:
        import imageio
        if self._frames:
            with imageio.get_writer(path, fps=5) as writer:
                for frame in self._frames:
                    writer.append_data(frame)
