import argparse
import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.reward_utils import classify_info_outcome
from evaluation.eval import load_dqn_agent, resolve_checkpoint_path
from utils.project_config import build_env_kwargs, load_project_config


def hold_final_frame(env, hold_seconds: float, fps: float = 30.0) -> None:
    """Keep rendering the final frame briefly for easier visual inspection."""

    if hold_seconds <= 0.0:
        return
    frame_delay = 1.0 / max(1.0, fps)
    deadline = time.monotonic() + hold_seconds
    while time.monotonic() < deadline:
        env.render()
        time.sleep(frame_delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "config.yaml"))
    parser.add_argument("--difficulty", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--touchdown_hold_seconds", type=float, default=1.0)
    args = parser.parse_args()
    resolved_checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    project_config = load_project_config(args.config)
    env = build_env_kwargs(
        project_config,
        difficulty=args.difficulty,
        render_mode="human",
        seed=args.seed,
    )
    from env.multi_pad_lander import MultiPadLanderEnv

    env = MultiPadLanderEnv(**env)
    agent = load_dqn_agent(str(resolved_checkpoint_path), env, device=args.device)

    print(f"[INFO] Loaded checkpoint: {resolved_checkpoint_path}")
    print(
        f"[INFO] obs_dim={env.observation_space.shape[0]}, "
        f"action_dim={env.action_space.n}, difficulty={args.difficulty}"
    )

    for ep in range(args.episodes):
        state, info = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = agent.select_action(state, greedy=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            env.render()

            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated

        outcome = classify_info_outcome(info, truncated)
        left_leg = bool(info.get("leg_contact_left", False))
        right_leg = bool(info.get("leg_contact_right", False))

        print(
            f"[Episode {ep + 1}] return={total_reward:.2f} "
            f"steps={steps} outcome={outcome}"
        )
        print(
            f"  final: speed={float(info.get('speed', 0.0)):.3f} "
            f"horizontal={float(info.get('horizontal_speed', 0.0)):.3f} "
            f"downward={float(info.get('downward_speed', 0.0)):.3f} "
            f"angle={float(info.get('angle', 0.0)):.3f} "
            f"angular_velocity={float(info.get('angular_velocity', 0.0)):.3f} "
            f"target_dx={float(info.get('target_dx', 0.0)):.3f} "
            f"legs=({int(left_leg)},{int(right_leg)})"
        )

        if left_leg and right_leg:
            hold_final_frame(env, hold_seconds=float(args.touchdown_hold_seconds))

    env.close()


if __name__ == "__main__":
    main()
