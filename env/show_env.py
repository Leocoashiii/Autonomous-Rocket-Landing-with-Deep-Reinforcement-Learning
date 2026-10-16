"""CLI for visualizing the environment without loading a trained checkpoint."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.multi_pad_lander import MultiPadLanderEnv
from env.reward_utils import ACTION_LABELS, classify_info_outcome
from utils.project_config import build_env_kwargs, get_env_defaults, load_project_config


PROJECT_CONFIG = load_project_config(os.path.join(PROJECT_ROOT, "config.yaml"))
ENV_DEFAULTS = get_env_defaults(PROJECT_CONFIG)
POLICY_CHOICES = ("random", *ACTION_LABELS.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the MultiPadLander environment from the command line.")

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--difficulty", type=str, default=str(ENV_DEFAULTS["difficulty"]))
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array", "none"],
    )
    parser.add_argument("--policy", type=str, default="random", choices=POLICY_CHOICES)

    parser.add_argument(
        "--pad-count",
        type=int,
        default=4,
        help="Fixed pad count for quick visual inspection when --min-pads/--max-pads are omitted.",
    )
    parser.add_argument("--min-pads", type=int, default=None)
    parser.add_argument("--max-pads", type=int, default=None)
    parser.add_argument("--max-encoded-pads", type=int, default=int(ENV_DEFAULTS["max_encoded_pads"]))

    parser.add_argument("--world-width", type=float, default=float(ENV_DEFAULTS["world_width"]))
    parser.add_argument("--world-height", type=float, default=float(ENV_DEFAULTS["world_height"]))
    parser.add_argument("--pad-width-min", type=float, default=float(ENV_DEFAULTS["pad_width_range"][0]))
    parser.add_argument("--pad-width-max", type=float, default=float(ENV_DEFAULTS["pad_width_range"][1]))
    parser.add_argument("--pad-slope-min", type=float, default=float(ENV_DEFAULTS["pad_slope_range"][0]))
    parser.add_argument("--pad-slope-max", type=float, default=float(ENV_DEFAULTS["pad_slope_range"][1]))
    parser.add_argument("--pad-hazard-min", type=float, default=float(ENV_DEFAULTS["pad_hazard_range"][0]))
    parser.add_argument("--pad-hazard-max", type=float, default=float(ENV_DEFAULTS["pad_hazard_range"][1]))
    parser.add_argument("--min-pad-gap", type=float, default=float(ENV_DEFAULTS["min_pad_gap"]))
    parser.add_argument("--terrain-segments", type=int, default=int(ENV_DEFAULTS["terrain_segments"]))
    parser.add_argument("--base-ground-y", type=float, default=float(ENV_DEFAULTS["base_ground_y"]))
    parser.add_argument("--terrain-height-jitter", type=float, default=float(ENV_DEFAULTS["terrain_height_jitter"]))
    parser.add_argument("--water-level-offset", type=float, default=float(ENV_DEFAULTS["water_level_offset"]))
    parser.add_argument("--pad-water-clearance", type=float, default=float(ENV_DEFAULTS["pad_water_clearance"]))
    parser.add_argument("--max-episode-steps", type=int, default=int(ENV_DEFAULTS["max_episode_steps"]))

    parser.add_argument("--wind-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wind-strength", type=float, default=None)
    parser.add_argument("--gust-strength", type=float, default=None)
    parser.add_argument("--gust-decay", type=float, default=None)
    parser.add_argument("--engine-dispersion-scale", type=float, default=1.0)
    parser.add_argument("--torque-disturbance", type=float, default=0.0)
    parser.add_argument("--hard-landing-speed", type=float, default=1.75)
    parser.add_argument("--hard-landing-vertical-speed", type=float, default=1.25)

    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--include-layout-in-trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-layout", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def resolve_pad_bounds(args: argparse.Namespace) -> tuple[int, int]:
    if args.min_pads is None and args.max_pads is None:
        if args.pad_count is None:
            return int(ENV_DEFAULTS["min_pads"]), int(ENV_DEFAULTS["max_pads"])
        return int(args.pad_count), int(args.pad_count)

    min_pads = int(args.min_pads if args.min_pads is not None else ENV_DEFAULTS["min_pads"])
    max_pads = int(args.max_pads if args.max_pads is not None else ENV_DEFAULTS["max_pads"])
    if min_pads < 1 or max_pads < min_pads:
        raise ValueError("pad bounds must satisfy 1 <= min_pads <= max_pads")
    return min_pads, max_pads


def choose_action(policy: str, env: MultiPadLanderEnv) -> int:
    if policy == "random":
        return int(env.action_space.sample())
    for action_id, action_name in ACTION_LABELS.items():
        if action_name == policy:
            return int(action_id)
    raise ValueError(f"unknown policy: {policy}")


def print_layout(env: MultiPadLanderEnv, info: dict) -> None:
    snapshot = env.get_layout_snapshot()
    print("=" * 90)
    print(
        f"difficulty={snapshot['difficulty_level']} "
        f"pads={len(snapshot['pads'])} "
        f"selected_pad={snapshot['selected_pad_id']}"
    )
    print(
        f"wind_enabled={snapshot['wind_enabled']} "
        f"wind_strength={snapshot['wind_strength']:.2f} "
        f"gust_strength={snapshot['gust_strength']:.2f} "
        f"gust_decay={snapshot['gust_decay']:.2f}"
    )
    print(
        f"world=({snapshot['world_width']:.1f}, {snapshot['world_height']:.1f}) "
        f"terrain_segments={snapshot['terrain_segments']} "
        f"terrain_height_jitter={snapshot['terrain_height_jitter']:.2f}"
    )
    print(f"obs_dim={env.observation_space.shape[0]} action_dim={env.action_space.n}")
    print(f"reported_pad_count={len(info['pads'])}")
    for pad in snapshot["pads"]:
        print(
            f"pad[{pad['pad_id']}] center=({pad['center_x']:.2f}, {pad['center_y']:.2f}) "
            f"width={pad['width']:.2f} slope={pad['slope']:.3f} "
            f"hazard={pad['hazard']:.2f} target={pad['pad_id'] == snapshot['selected_pad_id']}"
        )


def main() -> int:
    args = parse_args()
    min_pads, max_pads = resolve_pad_bounds(args)
    render_mode = None if args.render_mode == "none" else args.render_mode

    env_kwargs = build_env_kwargs(
        PROJECT_CONFIG,
        difficulty=args.difficulty,
        render_mode=render_mode,
        seed=args.seed,
    )
    env_kwargs.update(
        {
            "world_width": float(args.world_width),
            "world_height": float(args.world_height),
            "min_pads": int(min_pads),
            "max_pads": int(max_pads),
            "max_encoded_pads": int(args.max_encoded_pads),
            "pad_width_range": (float(args.pad_width_min), float(args.pad_width_max)),
            "pad_slope_range": (float(args.pad_slope_min), float(args.pad_slope_max)),
            "pad_hazard_range": (float(args.pad_hazard_min), float(args.pad_hazard_max)),
            "min_pad_gap": float(args.min_pad_gap),
            "terrain_segments": int(args.terrain_segments),
            "base_ground_y": float(args.base_ground_y),
            "terrain_height_jitter": float(args.terrain_height_jitter),
            "water_level_offset": float(args.water_level_offset),
            "pad_water_clearance": float(args.pad_water_clearance),
            "max_episode_steps": int(args.max_episode_steps),
            "engine_dispersion_scale": float(args.engine_dispersion_scale),
            "torque_disturbance": float(args.torque_disturbance),
            "hard_landing_speed": float(args.hard_landing_speed),
            "hard_landing_vertical_speed": float(args.hard_landing_vertical_speed),
        }
    )
    if args.wind_enabled is not None:
        env_kwargs["wind_enabled"] = bool(args.wind_enabled)
    if args.wind_strength is not None:
        env_kwargs["wind_strength"] = float(args.wind_strength)
    if args.gust_strength is not None:
        env_kwargs["gust_strength"] = float(args.gust_strength)
    if args.gust_decay is not None:
        env_kwargs["gust_decay"] = float(args.gust_decay)

    env = MultiPadLanderEnv(**env_kwargs)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

    returns: list[float] = []
    lengths: list[int] = []
    outcomes: Counter[str] = Counter()

    try:
        for episode_idx in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode_idx)
            if args.print_layout:
                print_layout(env, info)

            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action = choose_action(args.policy, env)
                obs, reward, terminated, truncated, info = env.step(action)
                if render_mode is not None:
                    env.render()
                total_reward += reward
                steps += 1

            outcome = classify_info_outcome(info, truncated)
            returns.append(float(total_reward))
            lengths.append(int(steps))
            outcomes[outcome] += 1

            print(
                f"[episode {episode_idx + 1}] policy={args.policy} return={total_reward:.2f} "
                f"steps={steps} outcome={outcome}"
            )

            if trace_dir is not None:
                trace_path = trace_dir / (
                    f"{args.difficulty}_policy-{args.policy}_seed{args.seed}_episode_{episode_idx + 1:03d}.json"
                )
                env.save_episode_trace(trace_path, include_layout=bool(args.include_layout_in_trace))
                print(f"[trace] saved {trace_path}")
    finally:
        env.close()

    print("=" * 90)
    print(
        f"episodes={args.episodes} difficulty={args.difficulty} policy={args.policy} "
        f"render_mode={args.render_mode} seed={args.seed}"
    )
    print(f"pad_bounds=min:{min_pads} max:{max_pads} max_encoded:{args.max_encoded_pads}")
    print(f"return mean±std: {statistics.fmean(returns):.2f} ± {statistics.pstdev(returns):.2f}")
    print(f"length mean±std: {statistics.fmean(lengths):.2f} ± {statistics.pstdev(lengths):.2f}")
    print("outcomes:")
    for outcome, count in sorted(outcomes.items()):
        print(f"  {outcome}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
