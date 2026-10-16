"""Evaluation utilities and CLI for trained agents."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.dqn_agent import DQNAgent, DQNConfig
from env.multi_pad_lander import MultiPadLanderEnv
from env.reward_utils import classify_info_outcome
from training.logger import write_json
from utils.project_config import (
    build_env_kwargs,
    get_eval_defaults,
    load_project_config,
)
from utils.seeds import episode_seed


def resolve_checkpoint_path(checkpoint_path: str | os.PathLike[str]) -> Path:
    raw_path = Path(checkpoint_path).expanduser()
    candidates: list[Path] = []

    if raw_path.exists():
        return raw_path

    if raw_path.is_absolute() and len(raw_path.parts) > 1 and raw_path.parts[1] == "outputs":
        candidates.append(Path(PROJECT_ROOT) / Path(*raw_path.parts[1:]))

    if not raw_path.is_absolute():
        candidates.append(Path(PROJECT_ROOT) / raw_path)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
        if candidate.exists():
            return candidate

    message = [f"checkpoint file not found: {raw_path}"]
    if raw_path.is_absolute() and len(raw_path.parts) > 1 and raw_path.parts[1] == "outputs":
        message.append(
            "It looks like you used '/outputs/...'. For files inside this repository, "
            "use 'outputs/...' or the full project path."
        )
    if unique_candidates:
        message.append("Tried:")
        message.extend(f"  - {candidate}" for candidate in unique_candidates)
    raise FileNotFoundError("\n".join(message))


def load_dqn_agent(checkpoint_path: str, env: MultiPadLanderEnv, device: str = "cpu") -> DQNAgent:
    """Load a DQN checkpoint for an environment with matching dimensions."""

    resolved_checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved_checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = dict(checkpoint.get("config", {}))
    hidden_dims = tuple(checkpoint_config.get("hidden_dims", (256, 128, 64)))

    config = DQNConfig(
        state_dim=int(checkpoint_config.get("state_dim", int(env.observation_space.shape[0]))),
        action_dim=int(checkpoint_config.get("action_dim", int(env.action_space.n))),
        hidden_dims=hidden_dims,
        learning_rate=float(checkpoint_config.get("learning_rate", 1e-3)),
        gamma=float(checkpoint_config.get("gamma", 0.99)),
        replay_capacity=int(checkpoint_config.get("replay_capacity", 100_000)),
        batch_size=int(checkpoint_config.get("batch_size", 64)),
        min_replay_size=int(checkpoint_config.get("min_replay_size", 2_000)),
        epsilon_start=float(checkpoint_config.get("epsilon_start", 1.0)),
        epsilon_end=float(checkpoint_config.get("epsilon_end", 0.01)),
        epsilon_decay_steps=int(checkpoint_config.get("epsilon_decay_steps", 50_000)),
        target_update_freq=int(checkpoint_config.get("target_update_freq", 100)),
        grad_clip_norm=float(checkpoint_config.get("grad_clip_norm", 10.0)),
        use_double_dqn=bool(checkpoint_config.get("use_double_dqn", True)),
        reward_clip_value=checkpoint_config.get("reward_clip_value", 20.0),
        target_q_clip_value=checkpoint_config.get("target_q_clip_value", 50.0),
        target_value_clip_value=checkpoint_config.get("target_value_clip_value", 75.0),
        q_value_hard_limit=float(checkpoint_config.get("q_value_hard_limit", 1_000.0)),
        td_error_hard_limit=float(checkpoint_config.get("td_error_hard_limit", 1_000.0)),
        loss_hard_limit=float(checkpoint_config.get("loss_hard_limit", 10_000.0)),
        device=device,
    )
    agent = DQNAgent(config)
    agent.load(str(resolved_checkpoint_path))
    agent.set_eval_mode()
    return agent


def evaluate_agent(
    agent: DQNAgent,
    *,
    episodes: int,
    seed_base: int,
    env_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Run deterministic evaluation episodes and aggregate metrics."""

    env = MultiPadLanderEnv(**env_kwargs)
    returns = []
    lengths = []
    final_speeds = []
    final_horizontal_speeds = []
    final_downward_speeds = []
    touchdown_speeds = []
    touchdown_horizontal_speeds = []
    touchdown_downward_speeds = []
    outcome_counts: Counter[str] = Counter()

    agent.set_eval_mode()

    try:
        for ep in range(episodes):
            state, info = env.reset(seed=episode_seed(seed_base, ep))
            done = False
            truncated = False
            total_reward = 0.0
            steps = 0
            final_info = info

            while not done:
                action = agent.select_action(state, greedy=True)
                next_state, reward, terminated, truncated, final_info = env.step(action)
                state = next_state
                total_reward += reward
                steps += 1
                done = terminated or truncated

            returns.append(float(total_reward))
            lengths.append(int(steps))
            outcome = classify_info_outcome(final_info, truncated)
            outcome_counts[outcome] += 1

            final_speed = float(final_info.get("speed", 0.0))
            final_horizontal_speed = float(final_info.get("horizontal_speed", 0.0))
            final_downward_speed = float(final_info.get("downward_speed", 0.0))
            final_speeds.append(final_speed)
            final_horizontal_speeds.append(final_horizontal_speed)
            final_downward_speeds.append(final_downward_speed)

            if outcome not in {"out_of_bounds", "time_limit"}:
                touchdown_speeds.append(final_speed)
                touchdown_horizontal_speeds.append(final_horizontal_speed)
                touchdown_downward_speeds.append(final_downward_speed)
    finally:
        env.close()

    target_success_rate = outcome_counts.get("safe_target_landing", 0) / max(1, episodes)
    any_safe_landing_rate = (
        outcome_counts.get("safe_target_landing", 0) + outcome_counts.get("safe_wrong_pad_landing", 0)
    ) / max(1, episodes)

    metrics = {
        "difficulty": env_kwargs.get("difficulty_level", "baseline"),
        "eval_episodes": int(episodes),
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
        "eval_length_std": float(np.std(lengths)),
        "eval_success_rate": float(target_success_rate),
        "eval_any_safe_landing_rate": float(any_safe_landing_rate),
        "eval_final_speed_mean": float(np.mean(final_speeds)),
        "eval_final_speed_std": float(np.std(final_speeds)),
        "eval_final_horizontal_speed_mean": float(np.mean(final_horizontal_speeds)),
        "eval_final_horizontal_speed_std": float(np.std(final_horizontal_speeds)),
        "eval_final_downward_speed_mean": float(np.mean(final_downward_speeds)),
        "eval_final_downward_speed_std": float(np.std(final_downward_speeds)),
        "eval_touchdown_count": int(len(touchdown_speeds)),
        "eval_outcomes": dict(outcome_counts),
    }

    if touchdown_speeds:
        metrics["eval_touchdown_speed_mean"] = float(np.mean(touchdown_speeds))
        metrics["eval_touchdown_speed_std"] = float(np.std(touchdown_speeds))
        metrics["eval_touchdown_horizontal_speed_mean"] = float(np.mean(touchdown_horizontal_speeds))
        metrics["eval_touchdown_horizontal_speed_std"] = float(np.std(touchdown_horizontal_speeds))
        metrics["eval_touchdown_downward_speed_mean"] = float(np.mean(touchdown_downward_speeds))
        metrics["eval_touchdown_downward_speed_std"] = float(np.std(touchdown_downward_speeds))
    else:
        metrics["eval_touchdown_speed_mean"] = None
        metrics["eval_touchdown_speed_std"] = None
        metrics["eval_touchdown_horizontal_speed_mean"] = None
        metrics["eval_touchdown_horizontal_speed_std"] = None
        metrics["eval_touchdown_downward_speed_mean"] = None
        metrics["eval_touchdown_downward_speed_std"] = None

    return metrics


def build_checkpoint_selection_key(metrics: Dict[str, Any]) -> tuple[float, ...]:
    """Score evaluation metrics for checkpoint selection.

    We prioritize actual safe-landings first, then prefer checkpoints that
    touch down more often and avoid timeout / fly-away behavior.

    When no checkpoint is achieving safe landings yet, using touchdown count as
    the dominant tiebreaker can prefer very early "hit the ground a lot" models
    over later checkpoints that hover, slow down, and collect much better return.
    In that zero-success regime we instead prioritize return first, then prefer
    target-pad contact and fewer catastrophic failures.
    """

    eval_episodes = max(1, int(metrics.get("eval_episodes", 1)))
    outcomes = dict(metrics.get("eval_outcomes", {}))
    success_rate = float(metrics.get("eval_success_rate", 0.0))
    any_safe_landing_rate = float(metrics.get("eval_any_safe_landing_rate", 0.0))
    target_contact_rate = (
        float(outcomes.get("safe_target_landing", 0) + outcomes.get("unsafe_target_contact", 0)) / eval_episodes
    )
    touchdown_speed = metrics.get("eval_touchdown_speed_mean")
    touchdown_speed_score = -1.0e9 if touchdown_speed is None else -float(touchdown_speed)
    eval_return_mean = float(metrics.get("eval_return_mean", -1.0e9))
    time_limit_rate = -float(outcomes.get("time_limit", 0)) / eval_episodes
    out_of_bounds_rate = -float(outcomes.get("out_of_bounds", 0)) / eval_episodes
    body_crash_rate = -float(outcomes.get("body_crash_or_other", 0)) / eval_episodes
    touchdown_rate = float(metrics.get("eval_touchdown_count", 0)) / eval_episodes

    if success_rate <= 0.0 and any_safe_landing_rate <= 0.0:
        return (
            eval_return_mean,
            target_contact_rate,
            out_of_bounds_rate,
            body_crash_rate,
            time_limit_rate,
            touchdown_speed_score,
            touchdown_rate,
        )

    return (
        success_rate,
        any_safe_landing_rate,
        target_contact_rate,
        eval_return_mean,
        out_of_bounds_rate,
        body_crash_rate,
        time_limit_rate,
        touchdown_speed_score,
        touchdown_rate,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "config.yaml"))
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()
    resolved_checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    project_config = load_project_config(args.config)
    eval_defaults = get_eval_defaults(project_config)
    difficulty = args.difficulty or eval_defaults.get("difficulty", "baseline")
    episodes = args.episodes or int(eval_defaults.get("episodes", 20))

    env_kwargs = build_env_kwargs(
        project_config,
        difficulty=difficulty,
        render_mode=None,
        seed=args.seed,
    )
    env = MultiPadLanderEnv(**env_kwargs)
    try:
        agent = load_dqn_agent(str(resolved_checkpoint_path), env, device=args.device)
    finally:
        env.close()

    metrics = evaluate_agent(
        agent,
        episodes=episodes,
        seed_base=args.seed,
        env_kwargs=env_kwargs,
    )

    print(json.dumps(metrics, indent=2))
    if resolved_checkpoint_path != Path(args.checkpoint).expanduser():
        print(f"[INFO] Resolved checkpoint path: {resolved_checkpoint_path}")
    if args.output_json:
        write_json(args.output_json, metrics)
        print(f"[INFO] Wrote evaluation metrics to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
