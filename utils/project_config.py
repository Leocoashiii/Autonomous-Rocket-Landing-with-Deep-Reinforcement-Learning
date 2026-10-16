"""Helpers for loading the repository's shared project configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when the shared YAML config is malformed."""


def load_project_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")
    return data


def _mapping(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"config[{key!r}] must be a mapping")
    return value


def get_env_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(_mapping(_mapping(config, "env"), "defaults"))


def get_eval_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(_mapping(config, "evaluation"))


def get_output_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(_mapping(config, "output"))


def get_seed_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(_mapping(config, "reproducibility"))


def get_dqn_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(_mapping(config, "dqn"))


def get_curriculum_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(_mapping(config, "curriculum"))


def get_training_profile_defaults(config: Dict[str, Any], profile: str) -> Dict[str, Any]:
    profiles = _mapping(config, "training_profiles")
    value = profiles.get(profile)
    if not isinstance(value, dict):
        raise ConfigError(f"training profile {profile!r} not found in config")
    return copy.deepcopy(value)


def build_env_kwargs(
    config: Dict[str, Any],
    *,
    difficulty: str | None = None,
    render_mode: str | None = None,
    seed: int | None = None,
) -> Dict[str, Any]:
    defaults = get_env_defaults(config)
    kwargs = {
        "render_mode": defaults.get("render_mode"),
        "difficulty_level": defaults.get("difficulty", "baseline"),
        "world_width": defaults.get("world_width", 42.0),
        "world_height": defaults.get("world_height", 28.0),
        "min_pads": defaults.get("min_pads", 3),
        "max_pads": defaults.get("max_pads", 5),
        "max_encoded_pads": defaults.get("max_encoded_pads", 5),
        "pad_width_range": tuple(defaults.get("pad_width_range", [1.2, 2.4])),
        "pad_slope_range": tuple(defaults.get("pad_slope_range", [-0.02, 0.02])),
        "pad_hazard_range": tuple(defaults.get("pad_hazard_range", [0.0, 1.0])),
        "min_pad_gap": defaults.get("min_pad_gap", 2.5),
        "terrain_segments": defaults.get("terrain_segments", 55),
        "base_ground_y": defaults.get("base_ground_y", 5.0),
        "terrain_height_jitter": defaults.get("terrain_height_jitter", 6.0),
        "water_level_offset": defaults.get("water_level_offset", 2.0),
        "pad_water_clearance": defaults.get("pad_water_clearance", 1.0),
        "max_episode_steps": defaults.get("max_episode_steps", 1000),
        "reward_config": copy.deepcopy(_mapping(config, "reward")),
        "seed": defaults.get("seed"),
    }

    if difficulty is not None:
        kwargs["difficulty_level"] = difficulty
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if seed is not None:
        kwargs["seed"] = seed

    return kwargs


def resolve_output_root(config: Dict[str, Any], project_root: str | Path) -> Path:
    output_defaults = get_output_defaults(config)
    output_root = Path(output_defaults.get("root_dir", "outputs"))
    if not output_root.is_absolute():
        output_root = Path(project_root) / output_root
    return output_root


def get_run_name_template(config: Dict[str, Any]) -> str:
    output = get_output_defaults(config)
    return output.get(
        "run_name_template",
        "dqn_single_{profile}_{difficulty}_seed{seed}_{timestamp}",
    )
