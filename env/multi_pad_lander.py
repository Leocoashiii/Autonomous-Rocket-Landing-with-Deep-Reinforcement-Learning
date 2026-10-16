"""Multi-pad rocket-landing environment.

Canonical course presets are baseline/level1/level2/level3. The optional
train_easy preset is available as a warm-start convenience.
"""
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from Box2D.b2 import (
    contactListener,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    world,
)
from gymnasium import spaces

from env.reward_utils import (
    ACTION_LABELS,
    BASE_OBSERVATION_FEATURES,
    PAD_OBSERVATION_FEATURES,
    REWARD_COMPONENTS,
    TERMINATION_CATEGORIES,
    empty_reward_breakdown,
)

# =========================
# Constants
# =========================

FPS = 50
SCALE = 30.0

VIEWPORT_W = 1200
VIEWPORT_H = 800

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6
INITIAL_RANDOM = 250.0

LANDER_POLY = [
    (-14, +17),
    (-17, 0),
    (-17, -10),
    (+17, -10),
    (+17, 0),
    (+14, +17),
]

LEG_AWAY = 20
LEG_DOWN = 18
LEG_H = 10
LEG_W = 3
LEG_SPRING_TORQUE = 40

PAD_COLOR = (80, 220, 90)
PAD_SELECTED_COLOR = (255, 220, 80)
PAD_RISK_COLOR = (220, 90, 90)
GROUND_COLOR = (170, 170, 170)
LANDER_COLOR = (220, 220, 220)
LEG_COLOR = (180, 180, 200)
SKY_COLOR = (12, 18, 28)
WAVE_COLOR = (30, 90, 150)
TEXT_COLOR = (240, 240, 240)
THRUST_COLOR = (255, 170, 60)
WIND_COLOR = (135, 206, 250)


# =========================
# Data Structures
# =========================

@dataclass
class Pad:
    pad_id: int
    left_x: float
    right_x: float
    y_left: float
    y_right: float
    hazard: float

    @property
    def center_x(self) -> float:
        return 0.5 * (self.left_x + self.right_x)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y_left + self.y_right)

    @property
    def width(self) -> float:
        return self.right_x - self.left_x

    @property
    def slope(self) -> float:
        dx = max(self.width, 1e-6)
        return (self.y_right - self.y_left) / dx


# =========================
# Contact Listener
# =========================

class LanderContactDetector(contactListener):
    def __init__(self, env: "MultiPadLanderEnv"):
        super().__init__()
        self.env = env

    def BeginContact(self, contact):
        a = contact.fixtureA.body
        b = contact.fixtureB.body

        if self.env.lander is None:
            return

        if a == self.env.lander or b == self.env.lander:
            other = b if a == self.env.lander else a
            if other not in self.env.leg_bodies:
                self.env.game_over = True

        for leg in self.env.leg_bodies:
            if a == leg or b == leg:
                leg.ground_contact = True

    def EndContact(self, contact):
        a = contact.fixtureA.body
        b = contact.fixtureB.body

        for leg in self.env.leg_bodies:
            if a == leg or b == leg:
                leg.ground_contact = False


# =========================
# Environment
# =========================

class MultiPadLanderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
        "train_easy": {
            "wind_enabled": False,
            "wind_strength": 0.0,
            "gust_strength": 0.0,
            "gust_decay": 0.85,

            # Easier geometry for single-policy DQN warm start
            "pad_width_range": (1.8, 3.0),
            "terrain_height_jitter": 3.0,
            "terrain_segments": 45,
            "pad_slope_range": (-0.01, 0.01),
            "min_pad_gap": 2.0,
        },
        "single_pad_easy": {
            "wind_enabled": False,
            "wind_strength": 0.0,
            "gust_strength": 0.0,
            "gust_decay": 0.85,
            "min_pads": 1,
            "max_pads": 1,
            "pad_width_range": (4.0, 6.5),
            "terrain_height_jitter": 2.0,
            "terrain_segments": 36,
            "pad_slope_range": (-0.005, 0.005),
            "min_pad_gap": 0.0,
        },
        "single_pad_level1": {
            "wind_enabled": True,
            "wind_strength": 0.8,
            "gust_strength": 0.0,
            "gust_decay": 0.85,
            "min_pads": 1,
            "max_pads": 1,
            "pad_width_range": (4.0, 6.5),
            "terrain_height_jitter": 2.0,
            "terrain_segments": 36,
            "pad_slope_range": (-0.005, 0.005),
            "min_pad_gap": 0.0,
        },
        "single_pad_level2": {
            "wind_enabled": True,
            "wind_strength": 0.8,
            "gust_strength": 0.45,
            "gust_decay": 0.92,
            "min_pads": 1,
            "max_pads": 1,
            "pad_width_range": (4.0, 6.5),
            "terrain_height_jitter": 2.0,
            "terrain_segments": 36,
            "pad_slope_range": (-0.005, 0.005),
            "min_pad_gap": 0.0,
        },
        "single_pad_level3": {
            "wind_enabled": True,
            "wind_strength": 1.8,
            "gust_strength": 0.9,
            "gust_decay": 0.94,
            "min_pads": 1,
            "max_pads": 1,
            "pad_width_range": (4.0, 6.5),
            "terrain_height_jitter": 2.0,
            "terrain_segments": 36,
            "pad_slope_range": (-0.005, 0.005),
            "min_pad_gap": 0.0,
        },
        "baseline": {
            "wind_enabled": False,
            "wind_strength": 0.0,
            "gust_strength": 0.0,
            "gust_decay": 0.85,
            "pad_width_range": (1.2, 2.4),
            "terrain_height_jitter": 6.0,
            "terrain_segments": 55,
            "pad_slope_range": (-0.02, 0.02),
            "min_pad_gap": 2.5,
        },
        "level1": {
            "wind_enabled": True,
            "wind_strength": 0.8,   # horizontal acceleration in world units / s^2
            "gust_strength": 0.0,
            "gust_decay": 0.85,
            "pad_width_range": (1.2, 2.4),
            "terrain_height_jitter": 6.0,
            "terrain_segments": 55,
            "pad_slope_range": (-0.02, 0.02),
            "min_pad_gap": 2.5,
        },
        "level2": {
            "wind_enabled": True,
            "wind_strength": 0.8,
            "gust_strength": 0.45,
            "gust_decay": 0.92,
            "pad_width_range": (1.2, 2.4),
            "terrain_height_jitter": 6.0,
            "terrain_segments": 60,
            "pad_slope_range": (-0.025, 0.025),
            "min_pad_gap": 2.6,
        },
        "level3": {
            "wind_enabled": True,
            "wind_strength": 1.8,
            "gust_strength": 0.9,
            "gust_decay": 0.94,
            "pad_width_range": (0.8, 1.7),
            "terrain_height_jitter": 9.5,
            "terrain_segments": 75,
            "pad_slope_range": (-0.035, 0.035),
            "min_pad_gap": 3.0,
        },
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        difficulty_level: str = "baseline",
        world_width: float = 42.0,
        world_height: float = 28.0,
        min_pads: int = 3,
        max_pads: int = 5,
        max_encoded_pads: int = 5,
        pad_width_range: Tuple[float, float] = (1.2, 2.4),
        pad_slope_range: Tuple[float, float] = (-0.02, 0.02),
        pad_hazard_range: Tuple[float, float] = (0.0, 1.0),
        min_pad_gap: float = 2.5,
        terrain_segments: int = 55,
        base_ground_y: float = 5.0,
        terrain_height_jitter: float = 6.0,
        water_level_offset: float = 2.0,
        pad_water_clearance: float = 1.0,
        wind_enabled: Optional[bool] = None,
        wind_strength: Optional[float] = None,
        gust_strength: Optional[float] = None,
        gust_decay: Optional[float] = None,
        engine_dispersion_scale: float = 1.0,
        torque_disturbance: float = 0.0,
        max_episode_steps: int = 1000,
        reward_config: Optional[Dict[str, Any]] = None,
        hard_landing_speed: float = 1.75,
        hard_landing_vertical_speed: float = 1.25,
        seed: Optional[int] = None,
    ):
        super().__init__()

        assert render_mode in (None, "human", "rgb_array")
        assert min_pads >= 1
        assert max_pads >= min_pads
        assert max_encoded_pads >= 1

        preset = self._get_preset_config(difficulty_level)

        # Allow explicit constructor args to override preset values.
        self.render_mode = render_mode
        self.difficulty_level = difficulty_level

        self.world_width = world_width
        self.world_height = world_height

        self.min_pads = min_pads
        self.max_pads = max_pads
        self.max_encoded_pads = max_encoded_pads

        self.pad_width_range = pad_width_range
        self.pad_slope_range = pad_slope_range
        self.pad_hazard_range = pad_hazard_range
        self.min_pad_gap = min_pad_gap

        self.terrain_segments = terrain_segments
        self.base_ground_y = base_ground_y
        self.terrain_height_jitter = terrain_height_jitter
        self.water_level_offset = water_level_offset
        self.pad_water_clearance = pad_water_clearance

        self.wind_enabled = bool(
            preset["wind_enabled"] if wind_enabled is None else wind_enabled
        )
        self.wind_strength = float(
            preset["wind_strength"] if wind_strength is None else wind_strength
        )
        self.gust_strength = float(
            preset["gust_strength"] if gust_strength is None else gust_strength
        )
        self.gust_decay = float(
            preset["gust_decay"] if gust_decay is None else gust_decay
        )
        self.engine_dispersion_scale = float(engine_dispersion_scale)
        self.torque_disturbance = float(torque_disturbance)
        self.hard_landing_speed = float(hard_landing_speed)
        self.hard_landing_vertical_speed = float(hard_landing_vertical_speed)

        # Apply preset geometry only when user did not explicitly change the geometry.
        if min_pads == 3:
            self.min_pads = int(preset.get("min_pads", min_pads))
        if max_pads == 5:
            self.max_pads = int(preset.get("max_pads", max_pads))
        if pad_width_range == (1.2, 2.4):
            self.pad_width_range = preset["pad_width_range"]
        if pad_slope_range == (-0.02, 0.02):
            self.pad_slope_range = preset["pad_slope_range"]
        if min_pad_gap == 2.5:
            self.min_pad_gap = preset["min_pad_gap"]
        if terrain_segments == 55:
            self.terrain_segments = preset["terrain_segments"]
        if terrain_height_jitter == 6.0:
            self.terrain_height_jitter = preset["terrain_height_jitter"]

        self.max_episode_steps = max_episode_steps
        self.rng = np.random.default_rng(seed)

        self.screen = None
        self.clock = None
        self.font = None
        self.isopen = True

        self.world: Optional[world] = None
        self.lander = None
        self.leg_bodies: List[Any] = []
        self.terrain_bodies: List[Any] = []
        self.terrain_vertices: List[Tuple[float, float]] = []
        self.pads: List[Pad] = []
        self.selected_pad_id: Optional[int] = None

        self.prev_shaping = None
        self.game_over = False
        self.step_count = 0

        self.last_main_thrust = 0.0
        self.last_side_thrust = 0.0
        self.last_wind = 0.0
        self.gust_state = 0.0
        self.episode_reward_total = 0.0
        self.action_counts = self._empty_action_counts()
        self.episode_trace: List[Dict[str, Any]] = []
        self.prev_contact_bonus = 0.0
        self.last_reward_breakdown = empty_reward_breakdown()
        self.hard_landing_crash = False
        self.target_settle_steps = 0
        self.wrong_pad_settle_steps = 0
        self.target_stall_steps = 0
        self.wrong_pad_stall_steps = 0

        self.contact_listener = None

        self.action_space = spaces.Discrete(len(ACTION_LABELS))

        # base = [x, y, vx, vy, angle, ang_vel, leg1, leg2]
        # pad feature = [dx, dy, half_width, slope, hazard, selected]
        obs_dim = 8 + self.max_encoded_pads * 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.reward_spec = self._build_reward_spec(reward_config)

    def _default_reward_spec(self) -> Dict[str, Any]:
        return {
            "components": list(REWARD_COMPONENTS),
            "shaping": {
                "distance_weight": 18.0,
                "vertical_distance_scale": 0.5,
                "speed_weight": 4.0,
                "horizontal_speed_weight": 3.0,
                "downward_speed_weight": 6.0,
                "angle_weight": 10.0,
                "angular_velocity_weight": 2.5,
                "hazard_weight": 3.0,
                "slope_weight": 4.0,
                "centered_bonus": 16.0,
                "angle_penalty_small": 1.0,
                "angle_penalty_large": 3.0,
                "angle_threshold": 0.5,
                "angular_velocity_extra_weight": 3.5,
                "wrong_pad_hover_penalty": 1.0,
                "near_ground_height": 2.5,
                "near_ground_speed_weight": 6.0,
                "near_ground_horizontal_speed_weight": 14.0,
                "near_ground_horizontal_speed_safe": 0.35,
                "near_ground_downward_speed_weight": 10.0,
                "near_ground_safe_speed": 0.75,
                "near_ground_angle_weight": 8.0,
                "near_ground_angular_velocity_weight": 6.0,
                "braking_zone_height": 6.0,
                "braking_zone_vertical_speed": 1.5,
                "main_engine_brake_bonus": 0.14,
                "missed_brake_penalty": 0.14,
                "target_zone_step_bonus": 0.06,
                "low_speed_step_bonus": 0.10,
                "low_speed_threshold": 1.0,
                "upright_step_bonus": 0.06,
                "upright_angle_threshold": 0.20,
                "center_alignment_bonus": 0.08,
                "near_ground_center_bonus": 0.10,
                "controlled_descent_bonus": 0.10,
                "controlled_descent_speed": 1.2,
                "controlled_descent_horizontal_speed": 0.40,
                "stable_descent_bonus": 0.18,
                "stable_descent_height": 6.0,
                "stable_descent_downward_speed_min": 0.15,
                "stable_descent_downward_speed_max": 1.20,
                "stable_descent_horizontal_speed": 0.45,
                "stable_descent_angle": 0.12,
                "stable_descent_angular_velocity": 0.60,
                "lateral_drift_penalty": 0.12,
                "lateral_drift_speed_threshold": 0.55,
                "tilt_step_penalty": 0.12,
                "tilt_angle_threshold": 0.18,
                "spin_step_penalty": 0.10,
                "spin_angular_velocity_threshold": 0.75,
                "landing_window_step_bonus": 0.14,
                "landing_window_height": 1.25,
                "landing_window_speed": 0.65,
                "landing_window_horizontal_speed": 0.20,
                "landing_window_angle": 0.20,
                "landing_window_angular_velocity": 1.0,
                "settle_height": 1.0,
                "settle_speed": 0.65,
                "settle_horizontal_speed": 0.28,
                "settle_downward_speed": 0.35,
                "settle_angle": 0.24,
                "settle_angular_velocity": 0.70,
                "settle_step_bonus": 0.28,
                "settle_center_bonus": 0.14,
                "settle_engine_off_bonus": 0.06,
                "post_contact_side_penalty": 0.14,
                "post_contact_main_penalty": 0.04,
                "post_contact_motion_penalty": 0.10,
                "post_contact_spin_penalty": 0.08,
                "post_contact_offcenter_penalty": 0.10,
                "scale": 0.03,
                "clip": 15.0,
            },
            "contact": {
                "leg_contact_bonus": 8.0,
                "scale": 0.03,
            },
            "fuel_penalty": {
                "main_engine": 0.12,
                "side_thruster": 0.04,
            },
            "terminal": {
                "safe_target_landing": 220.0,
                "safe_wrong_pad_landing": 60.0,
                "wrong_pad_contact": -30.0,
                "unsafe_target_contact": -55.0,
                "out_of_bounds": -45.0,
                "time_limit": -20.0,
                "body_crash_or_other": -60.0,
                "impact_speed_weight": 12.0,
                "impact_speed_clip": 4.0,
                "hard_landing_speed": self.hard_landing_speed,
                "hard_landing_vertical_speed": self.hard_landing_vertical_speed,
                "settle_steps_required": 5,
                "settle_altitude_threshold": 1.0,
                "settle_speed_threshold": 0.65,
                "settle_horizontal_speed_threshold": 0.28,
                "settle_downward_speed_threshold": 0.35,
                "settle_angle_threshold": 0.24,
                "settle_angular_velocity_threshold": 0.70,
                "stall_steps_required": 25,
                "stall_contact_margin": 0.75,
                "stall_altitude_threshold": 1.0,
                "stall_speed_threshold": 0.25,
                "stall_horizontal_speed_threshold": 0.12,
                "stall_downward_speed_threshold": 0.12,
                "stall_angle_threshold": 0.35,
                "stall_angular_velocity_threshold": 0.35,
            },
        }

    @staticmethod
    def _deep_merge_mapping(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = MultiPadLanderEnv._deep_merge_mapping(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def _build_reward_spec(self, reward_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        reward_spec = self._default_reward_spec()
        if reward_config:
            reward_spec = self._deep_merge_mapping(reward_spec, reward_config)

        terminal_cfg = reward_spec.setdefault("terminal", {})
        self.hard_landing_speed = float(
            terminal_cfg.get("hard_landing_speed", self.hard_landing_speed)
        )
        self.hard_landing_vertical_speed = float(
            terminal_cfg.get("hard_landing_vertical_speed", self.hard_landing_vertical_speed)
        )
        terminal_cfg["hard_landing_speed"] = self.hard_landing_speed
        terminal_cfg["hard_landing_vertical_speed"] = self.hard_landing_vertical_speed
        return reward_spec

    # =========================
    # Config Helpers
    # =========================

    @classmethod
    def _get_preset_config(cls, difficulty_level: str) -> Dict[str, Any]:
        key = difficulty_level.lower().replace(" ", "")
        alias = {
            "easy": "train_easy",
            "traineasy": "train_easy",
            "train_easy": "train_easy",
            "warmup": "train_easy",
            "warm_start": "train_easy",
            "singlepadeasy": "single_pad_easy",
            "single_pad_easy": "single_pad_easy",
            "widepadeasy": "single_pad_easy",
            "wide_pad_easy": "single_pad_easy",
            "singlepad": "single_pad_easy",
            "singlepadlevel1": "single_pad_level1",
            "single_pad_level1": "single_pad_level1",
            "singlepad1": "single_pad_level1",
            "widepadlevel1": "single_pad_level1",
            "singlepadlevel2": "single_pad_level2",
            "single_pad_level2": "single_pad_level2",
            "singlepad2": "single_pad_level2",
            "widepadlevel2": "single_pad_level2",
            "singlepadlevel3": "single_pad_level3",
            "single_pad_level3": "single_pad_level3",
            "singlepad3": "single_pad_level3",
            "widepadlevel3": "single_pad_level3",

            "0": "baseline",
            "1": "level1",
            "2": "level2",
            "3": "level3",
            "none": "baseline",
            "nofwind": "baseline",
            "weakwind": "level1",
            "weakwind+gust": "level2",
            "weakwindgust": "level2",
            "strongwind": "level3",
        }
        key = alias.get(key, key)
        if key not in cls.PRESET_CONFIGS:
            raise ValueError(
                f"Unknown difficulty_level={difficulty_level!r}. "
                f"Use one of {list(cls.PRESET_CONFIGS.keys())}."
            )
        return dict(cls.PRESET_CONFIGS[key])

    def set_difficulty(self, difficulty_level: str):
        preset = self._get_preset_config(difficulty_level)
        self.difficulty_level = difficulty_level
        self.wind_enabled = bool(preset["wind_enabled"])
        self.wind_strength = float(preset["wind_strength"])
        self.gust_strength = float(preset["gust_strength"])
        self.gust_decay = float(preset["gust_decay"])
        self.min_pads = int(preset.get("min_pads", self.min_pads))
        self.max_pads = int(preset.get("max_pads", self.max_pads))
        self.pad_width_range = preset["pad_width_range"]
        self.terrain_height_jitter = float(preset["terrain_height_jitter"])
        self.terrain_segments = int(preset["terrain_segments"])
        self.pad_slope_range = preset["pad_slope_range"]
        self.min_pad_gap = float(preset["min_pad_gap"])

    # =========================
    # Core API
    # =========================

    def _water_level(self) -> float:
        return self.base_ground_y - self.water_level_offset

    def _pad_above_water(self, left_x: float, right_x: float, y_left: float, y_right: float) -> bool:
        del left_x, right_x
        water_y = self._water_level()
        min_pad_y = min(y_left, y_right)
        return min_pad_y >= water_y + self.pad_water_clearance

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options is not None and "difficulty_level" in options:
            self.set_difficulty(str(options["difficulty_level"]))

        self.safe_landing = False
        self.safe_target_landing = False
        self.safe_wrong_pad_landing = False
        self.wrong_pad_contact = False
        self.unsafe_target_contact = False
        self.out_of_bounds = False
        self.hard_landing_crash = False

        self._destroy()
        self._build_world()

        self.step_count = 0
        self.game_over = False
        self.prev_shaping = None
        self.last_main_thrust = 0.0
        self.last_side_thrust = 0.0
        self.last_wind = 0.0
        self.gust_state = 0.0
        self.episode_reward_total = 0.0
        self.action_counts = self._empty_action_counts()
        self.episode_trace = []
        self.prev_contact_bonus = 0.0
        self.last_reward_breakdown = empty_reward_breakdown()
        self.target_settle_steps = 0
        self.wrong_pad_settle_steps = 0
        self.target_stall_steps = 0
        self.wrong_pad_stall_steps = 0
        self._record_trace(action=None, reward=0.0, terminated=False, truncated=False)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        assert self.lander is not None
        assert self.world is not None

        self.safe_landing = False
        self.safe_target_landing = False
        self.safe_wrong_pad_landing = False
        self.wrong_pad_contact = False
        self.unsafe_target_contact = False
        self.out_of_bounds = False
        self.hard_landing_crash = False

        self.step_count += 1
        self.last_main_thrust = 0.0
        self.last_side_thrust = 0.0

        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        fire_main = action in (2, 4, 5)
        side_direction = -1 if action in (1, 4) else +1 if action in (3, 5) else 0

        if fire_main:
            self._apply_main_engine_impulse(tip, side)
            self.last_main_thrust = 1.0

        if side_direction != 0:
            self._apply_side_engine_impulse(tip, side, direction=side_direction)
            self.last_side_thrust = float(side_direction)

        self._apply_environment_disturbances()
        pre_step_impact_metrics = self._impact_metrics(self.lander.linearVelocity)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        obs = self._get_obs()
        reward_breakdown = self._compute_reward(obs, action)

        terminated = False
        truncated = False
        terminal_reward = 0.0
        impact_penalty_applies = False

        x, y = self.lander.position
        vel = self.lander.linearVelocity
        post_step_impact_metrics = self._impact_metrics(vel)
        impact_metrics = {
            "speed": max(pre_step_impact_metrics["speed"], post_step_impact_metrics["speed"]),
            "horizontal_speed": max(
                pre_step_impact_metrics["horizontal_speed"],
                post_step_impact_metrics["horizontal_speed"],
            ),
            "downward_speed": max(
                pre_step_impact_metrics["downward_speed"],
                post_step_impact_metrics["downward_speed"],
            ),
        }
        hard_landing = self._is_hard_landing_metrics(impact_metrics)

        if x < 0 or x > self.world_width or y < 0:
            self.out_of_bounds = True
            terminal_reward += float(self.reward_spec["terminal"]["out_of_bounds"])
            terminated = True
            impact_penalty_applies = True

        target = self.pads[self.selected_pad_id]
        target_y = self._pad_height_at_x(target, x)
        on_target = self._is_x_on_pad(x, target)
        current_speed = post_step_impact_metrics["speed"]
        current_horizontal_speed = post_step_impact_metrics["horizontal_speed"]
        current_downward_speed = post_step_impact_metrics["downward_speed"]
        any_leg_contact = self.leg_bodies[0].ground_contact or self.leg_bodies[1].ground_contact
        both_legs = self.leg_bodies[0].ground_contact and self.leg_bodies[1].ground_contact
        thrusters_off = self.last_main_thrust <= 0.0 and abs(self.last_side_thrust) <= 0.0
        close_to_pad = on_target and abs(y - target_y) < 1.0
        target_surface_contact = close_to_pad and (any_leg_contact or self.game_over)
        terminal_cfg = self.reward_spec["terminal"]

        target_settled = self._is_settled_on_pad(
            target,
            x=x,
            y=y,
            angle=float(self.lander.angle),
            angular_velocity=float(self.lander.angularVelocity),
            speed=float(current_speed),
            horizontal_speed=float(current_horizontal_speed),
            downward_speed=float(current_downward_speed),
            both_legs=both_legs,
        )
        target_stalled = self._is_stalled_near_pad(
            target,
            x=x,
            y=y,
            angle=float(self.lander.angle),
            angular_velocity=float(self.lander.angularVelocity),
            speed=float(current_speed),
            horizontal_speed=float(current_horizontal_speed),
            downward_speed=float(current_downward_speed),
            any_leg_contact=any_leg_contact,
        )
        if target_settled:
            self.target_settle_steps += 1
        else:
            self.target_settle_steps = 0
        if target_stalled and not target_settled:
            self.target_stall_steps += 1
        else:
            self.target_stall_steps = 0

        if not terminated:
            if self.target_settle_steps >= int(terminal_cfg["settle_steps_required"]):
                self.safe_landing = True
                self.safe_target_landing = True
                terminal_reward += float(terminal_cfg["safe_target_landing"])
                terminated = True
            elif self.target_stall_steps >= int(terminal_cfg["stall_steps_required"]):
                self.unsafe_target_contact = True
                terminal_reward += float(terminal_cfg["unsafe_target_contact"])
                terminated = True
            elif target_surface_contact:
                if hard_landing:
                    self.hard_landing_crash = True
                    terminal_reward += float(terminal_cfg["body_crash_or_other"])
                elif self.game_over:
                    self.unsafe_target_contact = True
                    terminal_reward += float(terminal_cfg["unsafe_target_contact"])
                else:
                    target_surface_contact = False
                if target_surface_contact:
                    terminated = True
                    impact_penalty_applies = True

        if not terminated:
            nearest = self._nearest_pad(x)
            if nearest is not None and nearest.pad_id != target.pad_id:
                wrong_pad_y = self._pad_height_at_x(nearest, x)
                on_wrong_pad = self._is_x_on_pad(x, nearest)
                wrong_pad_settled = self._is_settled_on_pad(
                    nearest,
                    x=x,
                    y=y,
                    angle=float(self.lander.angle),
                    angular_velocity=float(self.lander.angularVelocity),
                    speed=float(current_speed),
                    horizontal_speed=float(current_horizontal_speed),
                    downward_speed=float(current_downward_speed),
                    both_legs=both_legs,
                )
                if wrong_pad_settled:
                    self.wrong_pad_settle_steps += 1
                else:
                    self.wrong_pad_settle_steps = 0
                wrong_pad_stalled = self._is_stalled_near_pad(
                    nearest,
                    x=x,
                    y=y,
                    angle=float(self.lander.angle),
                    angular_velocity=float(self.lander.angularVelocity),
                    speed=float(current_speed),
                    horizontal_speed=float(current_horizontal_speed),
                    downward_speed=float(current_downward_speed),
                    any_leg_contact=any_leg_contact,
                )
                if wrong_pad_stalled and not wrong_pad_settled:
                    self.wrong_pad_stall_steps += 1
                else:
                    self.wrong_pad_stall_steps = 0
                near_wrong_pad = (
                    on_wrong_pad
                    and abs(y - wrong_pad_y) < 0.6
                    and (any_leg_contact or self.game_over)
                )
                if self.wrong_pad_settle_steps >= int(terminal_cfg["settle_steps_required"]):
                    self.safe_landing = True
                    self.safe_wrong_pad_landing = True
                    terminal_reward += float(terminal_cfg["safe_wrong_pad_landing"])
                    terminated = True
                elif self.wrong_pad_stall_steps >= int(terminal_cfg["stall_steps_required"]):
                    self.wrong_pad_contact = True
                    terminal_reward += float(terminal_cfg["wrong_pad_contact"])
                    terminated = True
                elif near_wrong_pad:
                    if hard_landing:
                        self.hard_landing_crash = True
                        terminal_reward += float(terminal_cfg["body_crash_or_other"])
                    elif self.game_over:
                        self.wrong_pad_contact = True
                        terminal_reward += float(terminal_cfg["wrong_pad_contact"])
                    else:
                        near_wrong_pad = False
                    if near_wrong_pad:
                        terminated = True
                        impact_penalty_applies = True
            else:
                self.wrong_pad_settle_steps = 0
                self.wrong_pad_stall_steps = 0
        else:
            self.wrong_pad_settle_steps = 0
            self.wrong_pad_stall_steps = 0

        if not terminated and self.game_over:
            self.hard_landing_crash = bool(hard_landing)
            terminal_reward += float(terminal_cfg["body_crash_or_other"])
            terminated = True
            impact_penalty_applies = True

        if self.step_count >= self.max_episode_steps and not terminated:
            truncated = True
            terminal_reward += float(terminal_cfg["time_limit"])

        if impact_penalty_applies:
            impact_speed = min(
                impact_metrics["speed"],
                float(terminal_cfg["impact_speed_clip"]),
            )
            terminal_reward -= impact_speed * float(terminal_cfg["impact_speed_weight"])

        reward_breakdown["terminal_reward"] = float(terminal_reward)
        reward_breakdown["total_reward"] = float(
            reward_breakdown["shaping_reward"]
            + reward_breakdown["contact_reward"]
            + reward_breakdown["fuel_penalty"]
            + reward_breakdown["terminal_reward"]
        )
        self.last_reward_breakdown = reward_breakdown
        reward = float(reward_breakdown["total_reward"])

        self.action_counts[ACTION_LABELS[int(action)]] += 1
        self.episode_reward_total += reward
        self._record_trace(action=action, reward=reward, terminated=terminated, truncated=truncated)

        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _apply_environment_disturbances(self):
        assert self.lander is not None

        self.last_wind = 0.0

        if self.wind_enabled:
            noise = self.rng.normal(0.0, self.gust_strength) if self.gust_strength > 0.0 else 0.0
            self.gust_state = self.gust_decay * self.gust_state + (1.0 - self.gust_decay) * noise
            wind_accel_x = self.wind_strength + self.gust_state
            wind_force_x = self.lander.mass * wind_accel_x
            self.lander.ApplyForceToCenter((wind_force_x, 0.0), True)
            self.last_wind = float(wind_accel_x)

        if self.torque_disturbance > 0.0:
            torque = self.rng.normal(0.0, self.torque_disturbance)
            self.lander.ApplyTorque(float(torque), True)

    def _sample_engine_dispersion(self) -> Tuple[float, float]:
        return (
            self.engine_dispersion_scale * self.rng.uniform(-1.0, +1.0) / SCALE,
            self.engine_dispersion_scale * self.rng.uniform(-1.0, +1.0) / SCALE,
        )

    def _apply_main_engine_impulse(self, tip: Tuple[float, float], side: Tuple[float, float]) -> None:
        assert self.lander is not None
        dispersion = self._sample_engine_dispersion()
        ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
        oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
        impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
        impulse = (-ox * MAIN_ENGINE_POWER, -oy * MAIN_ENGINE_POWER)
        self.lander.ApplyLinearImpulse(impulse, impulse_pos, True)

    def _apply_side_engine_impulse(
        self,
        tip: Tuple[float, float],
        side: Tuple[float, float],
        *,
        direction: int,
    ) -> None:
        assert self.lander is not None
        dispersion = self._sample_engine_dispersion()
        ox = tip[0] * dispersion[0] + side[0] * (3 * direction + dispersion[1]) / SCALE
        oy = -tip[1] * dispersion[0] - side[1] * (3 * direction + dispersion[1]) / SCALE
        impulse_pos = (
            self.lander.position[0] + ox - tip[0] * 17 / SCALE,
            self.lander.position[1] + oy + tip[1] * LEG_AWAY / SCALE,
        )
        impulse = (-ox * SIDE_ENGINE_POWER, -oy * SIDE_ENGINE_POWER)
        self.lander.ApplyLinearImpulse(impulse, impulse_pos, True)

    def render(self):
        if self.render_mode is None:
            return None

        self._init_pygame()
        canvas = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        canvas.fill(SKY_COLOR)

        self._draw_background(canvas)
        self._draw_terrain(canvas)
        self._draw_pads(canvas)
        self._draw_lander(canvas)
        self._draw_hud(canvas)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump()
            self.clock.tick(FPS)
            pygame.display.flip()
            return None

        frame = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)),
            (1, 0, 2),
        )
        return frame.copy()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.screen = None

    # =========================
    # World Build
    # =========================

    def _build_world(self):
        self.world = world(gravity=(0, -10))
        self.contact_listener = LanderContactDetector(self)
        self.world.contactListener = self.contact_listener

        self._generate_terrain_and_pads()
        self._create_ground_bodies()
        self._create_lander()

    def _destroy(self):
        if self.world is None:
            return

        bodies = []
        if self.lander is not None:
            bodies.append(self.lander)
        bodies.extend(self.leg_bodies)
        bodies.extend(self.terrain_bodies)

        for body in bodies:
            try:
                self.world.DestroyBody(body)
            except Exception:
                pass

        self.world = None
        self.lander = None
        self.leg_bodies = []
        self.terrain_bodies = []
        self.terrain_vertices = []
        self.pads = []
        self.selected_pad_id = None

    def _generate_terrain_and_pads(self):
        n_pads = int(self.rng.integers(self.min_pads, self.max_pads + 1))
        pads = self._sample_pads(n_pads)
        pads = sorted(pads, key=lambda p: p.center_x)

        self.pads = pads
        self.terrain_vertices = self._build_terrain_profile_from_pads(pads)
        self.selected_pad_id = self._select_target_pad()

    def _sample_pads(self, n_pads: int) -> List[Pad]:
        pads: List[Pad] = []

        x_margin = 3.0
        usable_left = x_margin
        usable_right = self.world_width - x_margin

        max_tries = 800
        tries = 0

        while len(pads) < n_pads and tries < max_tries:
            tries += 1

            width = float(self.rng.uniform(*self.pad_width_range))
            left_x = float(self.rng.uniform(usable_left, usable_right - width))
            right_x = left_x + width

            slope = float(self.rng.uniform(*self.pad_slope_range))
            hazard = float(self.rng.uniform(*self.pad_hazard_range))

            water_y = self._water_level()
            y_min_center = water_y + self.pad_water_clearance + 0.5 * abs(slope) * width
            y_max_center = self.base_ground_y + self.terrain_height_jitter

            if y_min_center >= y_max_center:
                continue

            y_center = float(self.rng.uniform(y_min_center, y_max_center))
            y_left = y_center - 0.5 * slope * width
            y_right = y_center + 0.5 * slope * width

            if not self._pad_above_water(left_x, right_x, y_left, y_right):
                continue

            candidate = Pad(
                pad_id=len(pads),
                left_x=left_x,
                right_x=right_x,
                y_left=y_left,
                y_right=y_right,
                hazard=hazard,
            )

            if self._pad_valid(candidate, pads):
                pads.append(candidate)

        if len(pads) == 0:
            safe_y = max(self.base_ground_y, self._water_level() + self.pad_water_clearance + 1.0)
            fallback_width = 0.5 * (self.pad_width_range[0] + self.pad_width_range[1])
            pads = [
                Pad(
                    pad_id=0,
                    left_x=self.world_width * 0.5 - 0.5 * fallback_width,
                    right_x=self.world_width * 0.5 + 0.5 * fallback_width,
                    y_left=safe_y,
                    y_right=safe_y,
                    hazard=0.0,
                )
            ]

        for i, p in enumerate(sorted(pads, key=lambda z: z.center_x)):
            p.pad_id = i

        return pads

    def _pad_valid(self, candidate: Pad, existing: List[Pad]) -> bool:
        if not self._pad_above_water(
            candidate.left_x,
            candidate.right_x,
            candidate.y_left,
            candidate.y_right,
        ):
            return False

        for p in existing:
            non_overlap = (
                candidate.right_x + self.min_pad_gap <= p.left_x
                or candidate.left_x - self.min_pad_gap >= p.right_x
            )
            if not non_overlap:
                return False

        return True

    def _build_terrain_profile_from_pads(self, pads: List[Pad]) -> List[Tuple[float, float]]:
        control_x = np.linspace(0.0, self.world_width, self.terrain_segments + 1)
        heights = self.base_ground_y + self.rng.uniform(
            -self.terrain_height_jitter,
            self.terrain_height_jitter,
            size=len(control_x),
        )

        vertices: List[Tuple[float, float]] = []

        def terrain_height_nominal(x: float) -> float:
            return float(np.interp(x, control_x, heights))

        x_cursor = 0.0
        vertices.append((0.0, terrain_height_nominal(0.0)))

        for pad in pads:
            if pad.left_x > x_cursor:
                mid1 = x_cursor + (pad.left_x - x_cursor) * 0.33
                mid2 = x_cursor + (pad.left_x - x_cursor) * 0.66
                y1 = terrain_height_nominal(mid1)
                y2 = 0.5 * terrain_height_nominal(mid2) + 0.5 * pad.y_left
                vertices.extend([
                    (mid1, y1),
                    (mid2, y2),
                    (pad.left_x, pad.y_left),
                ])
            else:
                vertices.append((pad.left_x, pad.y_left))

            vertices.append((pad.right_x, pad.y_right))
            x_cursor = pad.right_x

        if x_cursor < self.world_width:
            end_mid1 = x_cursor + (self.world_width - x_cursor) * 0.33
            end_mid2 = x_cursor + (self.world_width - x_cursor) * 0.66
            vertices.extend([
                (end_mid1, terrain_height_nominal(end_mid1)),
                (end_mid2, terrain_height_nominal(end_mid2)),
                (self.world_width, terrain_height_nominal(self.world_width)),
            ])

        cleaned: List[Tuple[float, float]] = []
        for x, y in vertices:
            x = float(np.clip(x, 0.0, self.world_width))
            y = float(np.clip(y, 1.0, self.world_height - 2.0))
            if not cleaned or (abs(cleaned[-1][0] - x) > 1e-6 or abs(cleaned[-1][1] - y) > 1e-6):
                cleaned.append((x, y))

        return cleaned

    def _select_target_pad(self) -> int:
        scores = []
        for p in self.pads:
            score = 1.0 / (1.0 + 1.4 * p.hazard + 5.0 * abs(p.slope))
            scores.append(score)
        probs = np.asarray(scores, dtype=np.float64)
        probs /= probs.sum()
        return int(self.rng.choice(len(self.pads), p=probs))

    def _create_ground_bodies(self):
        assert self.world is not None

        self.terrain_bodies = []
        for i in range(len(self.terrain_vertices) - 1):
            p1 = self.terrain_vertices[i]
            p2 = self.terrain_vertices[i + 1]
            body = self.world.CreateStaticBody(shapes=edgeShape(vertices=[p1, p2]))
            body.color = GROUND_COLOR
            self.terrain_bodies.append(body)

    def _create_lander(self):
        assert self.world is not None

        initial_x = self.world_width / 2.0
        initial_y = self.world_height - 3.0

        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0xFFFF,
                restitution=0.0,
            ),
        )
        self.lander.color = LANDER_COLOR
        self.lander.ApplyForceToCenter(
            (
                self.rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        self.leg_bodies = []
        for sign in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(
                    initial_x + sign * LEG_AWAY / SCALE,
                    initial_y - LEG_DOWN / SCALE,
                ),
                angle=0.30 * sign,
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0xFFFF,
                ),
            )
            leg.ground_contact = False
            leg.color = LEG_COLOR

            joint = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(sign * LEG_AWAY / SCALE, -10 / SCALE),
                localAnchorB=(0, LEG_H / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=0.0,
            )

            if sign == -1:
                joint.lowerAngle = -0.55
                joint.upperAngle = -0.10
            else:
                joint.lowerAngle = +0.10
                joint.upperAngle = +0.55

            self.world.CreateJoint(joint)
            self.leg_bodies.append(leg)

    # =========================
    # Observation / Reward
    # =========================

    def _get_obs(self) -> np.ndarray:
        assert self.lander is not None

        x, y = self.lander.position
        vx, vy = self.lander.linearVelocity
        angle = self.lander.angle
        ang_vel = self.lander.angularVelocity
        leg1 = 1.0 if self.leg_bodies[0].ground_contact else 0.0
        leg2 = 1.0 if self.leg_bodies[1].ground_contact else 0.0

        base = np.array(
            [
                (x - self.world_width / 2.0) / (self.world_width / 2.0),
                (y - self.base_ground_y) / self.world_height,
                vx / 10.0,
                vy / 10.0,
                angle / math.pi,
                ang_vel / 10.0,
                leg1,
                leg2,
            ],
            dtype=np.float32,
        )

        target_pad = self.pads[self.selected_pad_id]
        other_pads = [p for p in self.pads if p.pad_id != self.selected_pad_id]
        other_pads.sort(key=lambda p: p.pad_id)
        pads_sorted = [target_pad] + other_pads

        pad_feats = []
        for i in range(self.max_encoded_pads):
            if i < len(pads_sorted):
                p = pads_sorted[i]
                dx = (p.center_x - x) / self.world_width
                dy = (p.center_y - y) / self.world_height
                half_width = 0.5 * p.width / self.world_width
                slope = p.slope
                hazard = p.hazard
                selected = 1.0 if p.pad_id == self.selected_pad_id else 0.0
                pad_feats.extend([dx, dy, half_width, slope, hazard, selected])
            else:
                pad_feats.extend([0.0] * 6)

        return np.concatenate([base, np.asarray(pad_feats, dtype=np.float32)], axis=0)

    def _compute_reward(self, obs: np.ndarray, action: int) -> Dict[str, float]:
        del obs
        assert self.lander is not None

        target = self.pads[self.selected_pad_id]
        x, y = self.lander.position
        vx, vy = self.lander.linearVelocity
        angle = self.lander.angle
        ang_vel = self.lander.angularVelocity

        dx = x - target.center_x
        target_y = self._pad_height_at_x(target, x)
        dy = y - target_y
        horizontal_speed = abs(vx)
        speed = math.sqrt(vx * vx + vy * vy)
        downward_speed = max(0.0, -vy)
        on_target = abs(dx) <= target.width * 0.5
        pad_half_width = max(target.width * 0.5, 1e-6)
        center_alignment = max(0.0, 1.0 - abs(dx) / pad_half_width)

        shaping_cfg = self.reward_spec["shaping"]
        contact_cfg = self.reward_spec["contact"]
        fuel_cfg = self.reward_spec["fuel_penalty"]
        terminal_cfg = self.reward_spec["terminal"]

        shaping = 0.0
        shaping -= shaping_cfg["distance_weight"] * math.sqrt(
            dx * dx + shaping_cfg["vertical_distance_scale"] * dy * dy
        )
        shaping -= shaping_cfg["speed_weight"] * speed
        shaping -= shaping_cfg["horizontal_speed_weight"] * horizontal_speed
        shaping -= shaping_cfg["downward_speed_weight"] * downward_speed
        shaping -= shaping_cfg["angle_weight"] * abs(angle)
        shaping -= shaping_cfg["angular_velocity_weight"] * abs(ang_vel)
        shaping -= shaping_cfg["hazard_weight"] * target.hazard
        shaping -= shaping_cfg["slope_weight"] * abs(target.slope)

        if on_target:
            shaping += shaping_cfg["centered_bonus"]

        if abs(dy) < shaping_cfg["near_ground_height"]:
            shaping -= shaping_cfg["near_ground_speed_weight"] * max(
                0.0,
                speed - shaping_cfg["near_ground_safe_speed"],
            )
            shaping -= shaping_cfg["near_ground_horizontal_speed_weight"] * max(
                0.0,
                horizontal_speed - shaping_cfg["near_ground_horizontal_speed_safe"],
            )
            shaping -= shaping_cfg["near_ground_downward_speed_weight"] * max(
                0.0,
                downward_speed - shaping_cfg["near_ground_safe_speed"],
            )
            shaping -= shaping_cfg["near_ground_angle_weight"] * abs(angle)
            shaping -= shaping_cfg["near_ground_angular_velocity_weight"] * abs(ang_vel)

        angle_penalty = (
            shaping_cfg["angle_penalty_large"]
            if abs(angle) > shaping_cfg["angle_threshold"]
            else shaping_cfg["angle_penalty_small"]
        )
        shaping -= angle_penalty * abs(angle)
        shaping -= shaping_cfg["angular_velocity_extra_weight"] * abs(ang_vel)
        shaping = shaping * shaping_cfg["scale"]
        shaping = float(np.clip(shaping, -shaping_cfg["clip"], shaping_cfg["clip"]))

        shaping_reward = 0.0
        if self.prev_shaping is not None:
            shaping_reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        any_leg_contact = self.leg_bodies[0].ground_contact or self.leg_bodies[1].ground_contact
        both_legs = self.leg_bodies[0].ground_contact and self.leg_bodies[1].ground_contact
        thrusters_off = self.last_main_thrust <= 0.0 and abs(self.last_side_thrust) <= 0.0

        if on_target:
            shaping_reward += shaping_cfg["target_zone_step_bonus"]
            shaping_reward += shaping_cfg["center_alignment_bonus"] * center_alignment
            if speed < shaping_cfg["low_speed_threshold"]:
                shaping_reward += shaping_cfg["low_speed_step_bonus"]
            if abs(angle) < shaping_cfg["upright_angle_threshold"]:
                shaping_reward += shaping_cfg["upright_step_bonus"]
            if abs(dy) < shaping_cfg["near_ground_height"]:
                shaping_reward += shaping_cfg["near_ground_center_bonus"] * center_alignment
            if (
                0.0 < dy < shaping_cfg["braking_zone_height"]
                and downward_speed > shaping_cfg["braking_zone_vertical_speed"]
            ):
                if self.last_main_thrust > 0.0:
                    shaping_reward += shaping_cfg["main_engine_brake_bonus"]
                else:
                    shaping_reward -= shaping_cfg["missed_brake_penalty"]
            if (
                0.0 < dy < shaping_cfg["braking_zone_height"]
                and 0.0 < downward_speed <= shaping_cfg["controlled_descent_speed"]
                and horizontal_speed <= shaping_cfg["controlled_descent_horizontal_speed"]
                and abs(angle) < shaping_cfg["upright_angle_threshold"]
            ):
                shaping_reward += shaping_cfg["controlled_descent_bonus"]
            if 0.0 < dy < shaping_cfg["stable_descent_height"]:
                if horizontal_speed > shaping_cfg["lateral_drift_speed_threshold"]:
                    shaping_reward -= shaping_cfg["lateral_drift_penalty"] * (
                        horizontal_speed - shaping_cfg["lateral_drift_speed_threshold"]
                    )
                if abs(angle) > shaping_cfg["tilt_angle_threshold"]:
                    shaping_reward -= shaping_cfg["tilt_step_penalty"] * (
                        abs(angle) - shaping_cfg["tilt_angle_threshold"]
                    )
                if abs(ang_vel) > shaping_cfg["spin_angular_velocity_threshold"]:
                    shaping_reward -= shaping_cfg["spin_step_penalty"] * (
                        abs(ang_vel) - shaping_cfg["spin_angular_velocity_threshold"]
                    )
                if (
                    shaping_cfg["stable_descent_downward_speed_min"] <= downward_speed <= shaping_cfg["stable_descent_downward_speed_max"]
                    and horizontal_speed <= shaping_cfg["stable_descent_horizontal_speed"]
                    and abs(angle) <= shaping_cfg["stable_descent_angle"]
                    and abs(ang_vel) <= shaping_cfg["stable_descent_angular_velocity"]
                ):
                    shaping_reward += shaping_cfg["stable_descent_bonus"]
            if (
                abs(dy) < shaping_cfg["landing_window_height"]
                and speed < shaping_cfg["landing_window_speed"]
                and horizontal_speed < shaping_cfg["landing_window_horizontal_speed"]
                and abs(angle) < shaping_cfg["landing_window_angle"]
                and abs(ang_vel) < shaping_cfg["landing_window_angular_velocity"]
            ):
                shaping_reward += shaping_cfg["landing_window_step_bonus"]
            if any_leg_contact and abs(dy) < shaping_cfg["settle_height"]:
                shaping_reward -= shaping_cfg["post_contact_side_penalty"] * abs(self.last_side_thrust)
                shaping_reward -= shaping_cfg["post_contact_offcenter_penalty"] * (1.0 - center_alignment)
            if both_legs and abs(dy) < shaping_cfg["settle_height"]:
                shaping_reward -= shaping_cfg["post_contact_main_penalty"] * self.last_main_thrust
                shaping_reward -= shaping_cfg["post_contact_motion_penalty"] * max(
                    0.0,
                    speed - shaping_cfg["settle_speed"],
                )
                shaping_reward -= shaping_cfg["post_contact_spin_penalty"] * max(
                    0.0,
                    abs(ang_vel) - shaping_cfg["settle_angular_velocity"],
                )
                if (
                    speed <= shaping_cfg["settle_speed"]
                    and horizontal_speed <= shaping_cfg["settle_horizontal_speed"]
                    and downward_speed <= shaping_cfg["settle_downward_speed"]
                    and abs(angle) <= shaping_cfg["settle_angle"]
                    and abs(ang_vel) <= shaping_cfg["settle_angular_velocity"]
                ):
                    shaping_reward += shaping_cfg["settle_step_bonus"]
                    shaping_reward += shaping_cfg["settle_center_bonus"] * center_alignment
                    if thrusters_off:
                        shaping_reward += shaping_cfg["settle_engine_off_bonus"]

        contact_bonus = 0.0
        if self.leg_bodies[0].ground_contact:
            contact_bonus += contact_cfg["leg_contact_bonus"]
        if self.leg_bodies[1].ground_contact:
            contact_bonus += contact_cfg["leg_contact_bonus"]
        contact_reward = (contact_bonus - self.prev_contact_bonus) * contact_cfg["scale"]
        self.prev_contact_bonus = contact_bonus

        fuel_penalty = (
            -fuel_cfg["main_engine"] * self.last_main_thrust
            -fuel_cfg["side_thruster"] * abs(self.last_side_thrust)
        )

        nearest = self._nearest_pad(x)
        if nearest is not None and nearest.pad_id != target.pad_id and self._is_x_on_pad(x, nearest):
            shaping_reward -= shaping_cfg["wrong_pad_hover_penalty"]

        breakdown = empty_reward_breakdown()
        breakdown["shaping_reward"] = float(shaping_reward)
        breakdown["contact_reward"] = float(contact_reward)
        breakdown["fuel_penalty"] = float(fuel_penalty)
        breakdown["total_reward"] = float(shaping_reward + contact_reward + fuel_penalty)
        return breakdown

    @staticmethod
    def _velocity_components(velocity: Any) -> Tuple[float, float]:
        if hasattr(velocity, "x") and hasattr(velocity, "y"):
            return float(velocity.x), float(velocity.y)
        return float(velocity[0]), float(velocity[1])

    def _impact_metrics(self, velocity: Any) -> Dict[str, float]:
        vx, vy = self._velocity_components(velocity)
        speed = math.sqrt(vx * vx + vy * vy)
        return {
            "speed": float(speed),
            "horizontal_speed": float(abs(vx)),
            "downward_speed": float(max(0.0, -vy)),
        }

    def _is_hard_landing_metrics(self, impact: Dict[str, float]) -> bool:
        terminal_cfg = self.reward_spec["terminal"]
        return bool(
            impact["speed"] > float(terminal_cfg["hard_landing_speed"])
            or impact["downward_speed"] > float(terminal_cfg["hard_landing_vertical_speed"])
        )

    def _is_hard_landing(self, velocity: Any) -> bool:
        return self._is_hard_landing_metrics(self._impact_metrics(velocity))

    def _is_settled_on_pad(
        self,
        pad: Pad,
        *,
        x: float,
        y: float,
        angle: float,
        angular_velocity: float,
        speed: float,
        horizontal_speed: float,
        downward_speed: float,
        both_legs: bool,
    ) -> bool:
        terminal_cfg = self.reward_spec["terminal"]
        pad_y = self._pad_height_at_x(pad, x)
        return bool(
            self._is_x_on_pad(x, pad)
            and both_legs
            and abs(y - pad_y) < float(terminal_cfg["settle_altitude_threshold"])
            and speed < float(terminal_cfg["settle_speed_threshold"])
            and horizontal_speed < float(terminal_cfg["settle_horizontal_speed_threshold"])
            and downward_speed < float(terminal_cfg["settle_downward_speed_threshold"])
            and abs(angle) < float(terminal_cfg["settle_angle_threshold"])
            and abs(angular_velocity) < float(terminal_cfg["settle_angular_velocity_threshold"])
        )

    def _is_stalled_near_pad(
        self,
        pad: Pad,
        *,
        x: float,
        y: float,
        angle: float,
        angular_velocity: float,
        speed: float,
        horizontal_speed: float,
        downward_speed: float,
        any_leg_contact: bool,
    ) -> bool:
        terminal_cfg = self.reward_spec["terminal"]
        pad_y = self._pad_height_at_x(pad, x)
        contact_margin = float(terminal_cfg["stall_contact_margin"])
        in_contact_zone = bool((pad.left_x - contact_margin) <= x <= (pad.right_x + contact_margin))
        return bool(
            any_leg_contact
            and in_contact_zone
            and abs(y - pad_y) < float(terminal_cfg["stall_altitude_threshold"])
            and speed < float(terminal_cfg["stall_speed_threshold"])
            and horizontal_speed < float(terminal_cfg["stall_horizontal_speed_threshold"])
            and downward_speed < float(terminal_cfg["stall_downward_speed_threshold"])
            and abs(angle) < float(terminal_cfg["stall_angle_threshold"])
            and abs(angular_velocity) < float(terminal_cfg["stall_angular_velocity_threshold"])
        )

    def _get_info(self) -> Dict[str, Any]:
        assert self.lander is not None

        target = self.pads[self.selected_pad_id]
        x, y = self.lander.position
        vx, vy = self.lander.linearVelocity
        impact_metrics = self._impact_metrics((vx, vy))
        return {
            "difficulty_level": self.difficulty_level,
            "wind_enabled": self.wind_enabled,
            "wind_strength": self.wind_strength,
            "gust_strength": self.gust_strength,
            "current_wind": self.last_wind,
            "selected_pad_id": self.selected_pad_id,
            "safe_landing": self.safe_landing,
            "safe_target_landing": self.safe_target_landing,
            "safe_wrong_pad_landing": self.safe_wrong_pad_landing,
            "wrong_pad_contact": self.wrong_pad_contact,
            "unsafe_target_contact": self.unsafe_target_contact,
            "out_of_bounds": self.out_of_bounds,
            "hard_landing_crash": self.hard_landing_crash,
            "target_settle_steps": int(self.target_settle_steps),
            "wrong_pad_settle_steps": int(self.wrong_pad_settle_steps),
            "target_stall_steps": int(self.target_stall_steps),
            "wrong_pad_stall_steps": int(self.wrong_pad_stall_steps),
            "water_level": self._water_level(),
            "pad_count": len(self.pads),
            "terrain_vertex_count": len(self.terrain_vertices),
            "speed": impact_metrics["speed"],
            "horizontal_speed": impact_metrics["horizontal_speed"],
            "downward_speed": impact_metrics["downward_speed"],
            "angle": float(self.lander.angle),
            "angular_velocity": float(self.lander.angularVelocity),
            "action_counts": dict(self.action_counts),
            "episode_reward_total": float(self.episode_reward_total),
            "trajectory_length": len(self.episode_trace),
            "reward_breakdown": dict(self.last_reward_breakdown),
            "pads": [
                self._pad_to_dict(p)
                for p in self.pads
            ],
            "target_dx": x - target.center_x,
            "target_dy": y - target.center_y,
            "step_count": self.step_count,
        }

    @staticmethod
    def _empty_action_counts() -> Dict[str, int]:
        return {label: 0 for label in ACTION_LABELS.values()}

    def get_action_spec(self) -> Dict[int, str]:
        return {int(action): label for action, label in ACTION_LABELS.items()}

    def get_observation_spec(self) -> Dict[str, Any]:
        return {
            "dimension": int(self.observation_space.shape[0]),
            "base_features": list(BASE_OBSERVATION_FEATURES),
            "pad_features": list(PAD_OBSERVATION_FEATURES),
            "base_feature_count": len(BASE_OBSERVATION_FEATURES),
            "pad_feature_count": len(PAD_OBSERVATION_FEATURES),
            "max_encoded_pads": int(self.max_encoded_pads),
        }

    def get_reward_spec(self) -> Dict[str, Any]:
        return {
            "components": list(self.reward_spec["components"]),
            "shaping": dict(self.reward_spec["shaping"]),
            "contact": dict(self.reward_spec["contact"]),
            "fuel_penalty": dict(self.reward_spec["fuel_penalty"]),
            "terminal": dict(self.reward_spec["terminal"]),
            "termination_categories": list(TERMINATION_CATEGORIES),
        }

    def _pad_to_dict(self, pad: Pad) -> Dict[str, Any]:
        return {
            "pad_id": int(pad.pad_id),
            "left_x": float(pad.left_x),
            "right_x": float(pad.right_x),
            "y_left": float(pad.y_left),
            "y_right": float(pad.y_right),
            "center_x": float(pad.center_x),
            "center_y": float(pad.center_y),
            "width": float(pad.width),
            "slope": float(pad.slope),
            "hazard": float(pad.hazard),
            "is_target": bool(pad.pad_id == self.selected_pad_id),
        }

    def _classify_outcome(self, truncated: bool = False) -> str:
        if self.safe_target_landing:
            return "safe_target_landing"
        if self.safe_wrong_pad_landing:
            return "safe_wrong_pad_landing"
        if self.wrong_pad_contact:
            return "wrong_pad_contact"
        if self.unsafe_target_contact:
            return "unsafe_target_contact"
        if self.out_of_bounds:
            return "out_of_bounds"
        if truncated:
            return "time_limit"
        return "body_crash_or_other"

    def _record_trace(
        self,
        *,
        action: Optional[int],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        assert self.lander is not None

        x, y = self.lander.position
        vx, vy = self.lander.linearVelocity
        trace_item = {
            "step": int(self.step_count),
            "action_id": None if action is None else int(action),
            "action_name": "reset" if action is None else ACTION_LABELS[int(action)],
            "reward": float(reward),
            "episode_reward_total": float(self.episode_reward_total),
            "reward_breakdown": dict(self.last_reward_breakdown),
            "x": float(x),
            "y": float(y),
            "vx": float(vx),
            "vy": float(vy),
            "speed": float(math.sqrt(vx * vx + vy * vy)),
            "angle": float(self.lander.angle),
            "angular_velocity": float(self.lander.angularVelocity),
            "leg_contact_left": bool(self.leg_bodies[0].ground_contact),
            "leg_contact_right": bool(self.leg_bodies[1].ground_contact),
            "current_wind": float(self.last_wind),
            "selected_pad_id": None if self.selected_pad_id is None else int(self.selected_pad_id),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "outcome": (
                self._classify_outcome(truncated)
                if (terminated or truncated)
                else "in_progress"
            ),
            "flags": {
                "safe_landing": bool(self.safe_landing),
                "safe_target_landing": bool(self.safe_target_landing),
                "safe_wrong_pad_landing": bool(self.safe_wrong_pad_landing),
                "wrong_pad_contact": bool(self.wrong_pad_contact),
                "unsafe_target_contact": bool(self.unsafe_target_contact),
                "out_of_bounds": bool(self.out_of_bounds),
                "hard_landing_crash": bool(self.hard_landing_crash),
            },
        }
        self.episode_trace.append(trace_item)

    def get_episode_trace(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.episode_trace]

    def get_layout_snapshot(self) -> Dict[str, Any]:
        return {
            "difficulty_level": self.difficulty_level,
            "world_width": float(self.world_width),
            "world_height": float(self.world_height),
            "water_level": float(self._water_level()),
            "terrain_segments": int(self.terrain_segments),
            "terrain_height_jitter": float(self.terrain_height_jitter),
            "wind_enabled": bool(self.wind_enabled),
            "wind_strength": float(self.wind_strength),
            "gust_strength": float(self.gust_strength),
            "gust_decay": float(self.gust_decay),
            "min_pads": int(self.min_pads),
            "max_pads": int(self.max_pads),
            "selected_pad_id": None if self.selected_pad_id is None else int(self.selected_pad_id),
            "terrain_vertices": [
                [float(x), float(y)]
                for x, y in self.terrain_vertices
            ],
            "pads": [
                self._pad_to_dict(p)
                for p in self.pads
            ],
        }

    def save_episode_trace(self, path: str | Path, include_layout: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "difficulty_level": self.difficulty_level,
            "action_spec": self.get_action_spec(),
            "observation_spec": self.get_observation_spec(),
            "reward_spec": self.get_reward_spec(),
            "summary": {
                "step_count": int(self.step_count),
                "episode_reward_total": float(self.episode_reward_total),
                "selected_pad_id": None if self.selected_pad_id is None else int(self.selected_pad_id),
                "pad_count": int(len(self.pads)),
                "final_outcome": (
                    self.episode_trace[-1]["outcome"]
                    if self.episode_trace
                    else "not_started"
                ),
                "action_counts": dict(self.action_counts),
            },
            "layout": self.get_layout_snapshot() if include_layout else None,
            "trace": self.get_episode_trace(),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    # =========================
    # Geometry Helpers
    # =========================

    def _nearest_pad(self, x: float) -> Optional[Pad]:
        if not self.pads:
            return None
        return min(self.pads, key=lambda p: abs(p.center_x - x))

    def _is_x_on_pad(self, x: float, pad: Pad) -> bool:
        return pad.left_x <= x <= pad.right_x

    def _pad_height_at_x(self, pad: Pad, x: float) -> float:
        if pad.right_x <= pad.left_x:
            return pad.center_y
        t = np.clip((x - pad.left_x) / max(pad.right_x - pad.left_x, 1e-6), 0.0, 1.0)
        return float((1.0 - t) * pad.y_left + t * pad.y_right)

    # =========================
    # Rendering
    # =========================

    def _init_pygame(self):
        if self.screen is not None:
            return

        if self.render_mode != "human":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        if self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)

    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        sx = int((x / self.world_width) * VIEWPORT_W)
        sy = int(VIEWPORT_H - (y / self.world_height) * VIEWPORT_H)
        return sx, sy

    def _draw_background(self, canvas: pygame.Surface):
        ocean_y = self._water_level()
        _, sy = self._world_to_screen(0.0, ocean_y)
        pygame.draw.rect(canvas, WAVE_COLOR, pygame.Rect(0, sy, VIEWPORT_W, VIEWPORT_H - sy))

        for i in range(7):
            pts = []
            y = ocean_y + 0.25 * i
            for x in np.linspace(0, self.world_width, 80):
                wy = y + 0.08 * math.sin(0.8 * x + 0.7 * i)
                pts.append(self._world_to_screen(x, wy))
            pygame.draw.lines(canvas, (60, 130, 190), False, pts, 1)

        self._draw_wind_indicator(canvas)

    def _draw_wind_indicator(self, canvas: pygame.Surface):
        arrow_len = int(np.clip(abs(self.last_wind) * 35, 0, 130))
        if arrow_len <= 2:
            return

        y = 40
        if self.last_wind >= 0:
            x0, x1 = 200, 200 + arrow_len
            pygame.draw.line(canvas, WIND_COLOR, (x0, y), (x1, y), 4)
            pygame.draw.polygon(canvas, WIND_COLOR, [(x1, y), (x1 - 12, y - 7), (x1 - 12, y + 7)])
        else:
            x0, x1 = 200, 200 - arrow_len
            pygame.draw.line(canvas, WIND_COLOR, (x0, y), (x1, y), 4)
            pygame.draw.polygon(canvas, WIND_COLOR, [(x1, y), (x1 + 12, y - 7), (x1 + 12, y + 7)])

    def _draw_terrain(self, canvas: pygame.Surface):
        pts = [self._world_to_screen(x, y) for x, y in self.terrain_vertices]
        if len(pts) >= 2:
            pygame.draw.lines(canvas, GROUND_COLOR, False, pts, 3)

        poly = pts + [(VIEWPORT_W, VIEWPORT_H), (0, VIEWPORT_H)]
        pygame.draw.polygon(canvas, (55, 55, 55), poly)

    def _draw_pads(self, canvas: pygame.Surface):
        for p in self.pads:
            p1 = self._world_to_screen(p.left_x, p.y_left)
            p2 = self._world_to_screen(p.right_x, p.y_right)

            if p.pad_id == self.selected_pad_id:
                color = PAD_SELECTED_COLOR
                thick = 6
            else:
                risk = float(np.clip(p.hazard, 0.0, 1.0))
                color = (
                    int((1.0 - risk) * PAD_COLOR[0] + risk * PAD_RISK_COLOR[0]),
                    int((1.0 - risk) * PAD_COLOR[1] + risk * PAD_RISK_COLOR[1]),
                    int((1.0 - risk) * PAD_COLOR[2] + risk * PAD_RISK_COLOR[2]),
                )
                thick = 4

            pygame.draw.line(canvas, color, p1, p2, thick)

            cx, cy = self._world_to_screen(p.center_x, p.center_y)
            pygame.draw.line(canvas, color, (cx, cy - 14), (cx, cy + 14), 2)

            label = f"P{p.pad_id} w={p.width:.1f} s={p.slope:+.2f} h={p.hazard:.2f}"
            surf = self.font.render(label, True, color)
            canvas.blit(surf, (cx - 55, cy - 34))

    def _draw_lander(self, canvas: pygame.Surface):
        assert self.lander is not None

        transformed = []
        for vx, vy in LANDER_POLY:
            wx, wy = self.lander.transform * (vx / SCALE, vy / SCALE)
            transformed.append(self._world_to_screen(wx, wy))
        pygame.draw.polygon(canvas, LANDER_COLOR, transformed)
        pygame.draw.polygon(canvas, (50, 50, 50), transformed, 2)

        for leg in self.leg_bodies:
            for fixture in leg.fixtures:
                shape = fixture.shape
                pts = [self._world_to_screen(*(leg.transform * v)) for v in shape.vertices]
                pygame.draw.polygon(canvas, LEG_COLOR, pts)
                pygame.draw.polygon(canvas, (50, 50, 50), pts, 1)

        if self.last_main_thrust > 0.0:
            flame_local = [(-0.12, -0.20), (0.12, -0.20), (0.0, -1.10)]
            flame_world = [self.lander.transform * v for v in flame_local]
            flame_pts = [self._world_to_screen(x, y) for x, y in flame_world]
            pygame.draw.polygon(canvas, THRUST_COLOR, flame_pts)

        if self.last_side_thrust != 0.0:
            sign = 1 if self.last_side_thrust > 0 else -1
            flame_local = [(0.35 * sign, 0.10), (0.35 * sign, -0.10), (0.90 * sign, 0.0)]
            flame_world = [self.lander.transform * v for v in flame_local]
            flame_pts = [self._world_to_screen(x, y) for x, y in flame_world]
            pygame.draw.polygon(canvas, (255, 120, 80), flame_pts)

    def _draw_hud(self, canvas: pygame.Surface):
        assert self.lander is not None

        target = self.pads[self.selected_pad_id]
        x, y = self.lander.position
        vx, vy = self.lander.linearVelocity
        speed = math.sqrt(vx * vx + vy * vy)

        lines = [
            f"difficulty: {self.difficulty_level}",
            f"step: {self.step_count}",
            f"target pad: P{self.selected_pad_id}",
            f"wind: {self.last_wind:+.2f}",
            f"base wind: {self.wind_strength:+.2f}",
            f"gust std: {self.gust_strength:.2f}",
            f"pos: ({x:.2f}, {y:.2f})",
            f"vel: ({vx:.2f}, {vy:.2f})",
            f"speed: {speed:.2f}",
            f"angle: {self.lander.angle:.3f}",
            f"dx to target: {x - target.center_x:+.2f}",
            f"dy to target: {y - target.center_y:+.2f}",
            f"reward total: {self.episode_reward_total:+.2f}",
            (
                "actions n/l/m/r/ml/mr: "
                f"{self.action_counts['noop']}/"
                f"{self.action_counts['left_thruster']}/"
                f"{self.action_counts['main_engine']}/"
                f"{self.action_counts['right_thruster']}/"
                f"{self.action_counts['main_engine_left']}/"
                f"{self.action_counts['main_engine_right']}"
            ),
        ]

        for i, line in enumerate(lines):
            surf = self.font.render(line, True, TEXT_COLOR)
            canvas.blit(surf, (16, 16 + i * 24))


if __name__ == "__main__":
    env = MultiPadLanderEnv(
        render_mode="human",
        difficulty_level="baseline",
        min_pads=3,
        max_pads=5,
        max_encoded_pads=5,
        seed=111,
    )

    obs, info = env.reset()
    print("obs shape:", obs.shape)
    print("difficulty:", info["difficulty_level"])
    print("selected pad:", info["selected_pad_id"])
    print("num pads:", len(info["pads"]))

    terminated = False
    truncated = False

    try:
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
    finally:
        env.close()
