"""Shared reward and termination helpers for environment-facing modules."""

from __future__ import annotations

from typing import Any, Dict

ACTION_LABELS = {
    0: "noop",
    1: "left_thruster",
    2: "main_engine",
    3: "right_thruster",
    4: "main_engine_left",
    5: "main_engine_right",
}

BASE_OBSERVATION_FEATURES = (
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "angular_velocity",
    "leg_contact_left",
    "leg_contact_right",
)

PAD_OBSERVATION_FEATURES = (
    "dx",
    "dy",
    "half_width",
    "slope",
    "hazard",
    "is_target",
)

REWARD_COMPONENTS = (
    "shaping_reward",
    "contact_reward",
    "fuel_penalty",
    "terminal_reward",
    "total_reward",
)

TERMINATION_CATEGORIES = (
    "safe_target_landing",
    "safe_wrong_pad_landing",
    "wrong_pad_contact",
    "unsafe_target_contact",
    "out_of_bounds",
    "time_limit",
    "body_crash_or_other",
)


def empty_reward_breakdown() -> Dict[str, float]:
    """Return a zero-initialized reward breakdown with stable keys."""

    return {key: 0.0 for key in REWARD_COMPONENTS}


def classify_info_outcome(info: Dict[str, Any], truncated: bool = False) -> str:
    """Classify an episode outcome from environment info flags."""

    if info.get("safe_target_landing", False):
        return "safe_target_landing"
    if info.get("safe_wrong_pad_landing", False):
        return "safe_wrong_pad_landing"
    if info.get("wrong_pad_contact", False):
        return "wrong_pad_contact"
    if info.get("unsafe_target_contact", False):
        return "unsafe_target_contact"
    if info.get("out_of_bounds", False):
        return "out_of_bounds"
    if truncated:
        return "time_limit"
    return "body_crash_or_other"
