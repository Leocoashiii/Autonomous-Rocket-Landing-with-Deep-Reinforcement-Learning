import argparse
import os
import sys
import time
from collections import deque
from dataclasses import asdict

import numpy as np
import torch

# Make project root importable when running: python3 training/train_single.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.multi_pad_lander import MultiPadLanderEnv
from env.reward_utils import classify_info_outcome
from evaluation.eval import build_checkpoint_selection_key, evaluate_agent
from agents.dqn_agent import DQNAgent, DQNConfig
from training.curriculum import add_curriculum_arguments, run_curriculum_pipeline
from training.logger import (
    append_csv_row,
    append_jsonl_row,
    build_run_name,
    prepare_run_paths,
    write_json,
)
from utils.plotting import create_run_figures
from utils.project_config import (
    build_env_kwargs,
    get_dqn_defaults,
    get_env_defaults,
    get_eval_defaults,
    get_output_defaults,
    get_run_name_template,
    get_seed_defaults,
    get_training_profile_defaults,
    load_project_config,
    resolve_output_root,
)
from utils.seeds import episode_seed, set_global_seed


PROJECT_CONFIG = load_project_config(os.path.join(PROJECT_ROOT, "config.yaml"))
ENV_DEFAULTS = get_env_defaults(PROJECT_CONFIG)
EVAL_DEFAULTS = get_eval_defaults(PROJECT_CONFIG)
OUTPUT_DEFAULTS = get_output_defaults(PROJECT_CONFIG)
SEED_DEFAULTS = get_seed_defaults(PROJECT_CONFIG)
TRAIN_DEFAULTS = get_training_profile_defaults(PROJECT_CONFIG, "gpu")
DQN_DEFAULTS = get_dqn_defaults(PROJECT_CONFIG)


def is_better_eval_checkpoint(metrics: dict, best_key: tuple[float, ...] | None) -> tuple[bool, tuple[float, ...]]:
    selection_key = build_checkpoint_selection_key(metrics)
    if best_key is None:
        return True, selection_key
    return selection_key > best_key, selection_key


def make_agent(env: MultiPadLanderEnv, device: str, args) -> DQNAgent:
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    hidden_dims = tuple(int(dim) for dim in DQN_DEFAULTS.get("hidden_dims", [256, 128, 64]))
    stability_defaults = dict(DQN_DEFAULTS.get("stability", {}))

    config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        min_replay_size=args.min_replay_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_freq=args.target_update_freq,
        grad_clip_norm=args.grad_clip_norm,
        use_double_dqn=args.use_double_dqn,
        reward_clip_value=(
            None
            if stability_defaults.get("reward_clip_value") is None
            else float(stability_defaults["reward_clip_value"])
        ),
        target_q_clip_value=(
            None
            if stability_defaults.get("target_q_clip_value") is None
            else float(stability_defaults["target_q_clip_value"])
        ),
        target_value_clip_value=(
            None
            if stability_defaults.get("target_value_clip_value") is None
            else float(stability_defaults["target_value_clip_value"])
        ),
        q_value_hard_limit=float(stability_defaults.get("q_value_hard_limit", 1_000.0)),
        td_error_hard_limit=float(stability_defaults.get("td_error_hard_limit", 1_000.0)),
        loss_hard_limit=float(stability_defaults.get("loss_hard_limit", 10_000.0)),
        device=device,
    )
    return DQNAgent(config)


def apply_resume_transition(agent: DQNAgent, args) -> dict:
    info = {
        "resume_path": args.resume_path,
        "applied": {
            "reset_epsilon_on_resume": False,
            "resume_epsilon_value": None,
            "reset_optimizer_on_resume": False,
            "reset_global_step_on_resume": False,
            "reset_learn_step_on_resume": False,
        },
    }
    if not args.resume_path:
        return info

    info["before"] = agent.training_state_snapshot()

    if args.reset_global_step_on_resume:
        agent.reset_global_step(0)
        info["applied"]["reset_global_step_on_resume"] = True

    if args.reset_learn_step_on_resume:
        agent.reset_learn_step(0)
        info["applied"]["reset_learn_step_on_resume"] = True

    if args.reset_optimizer_on_resume:
        agent.reset_optimizer()
        info["applied"]["reset_optimizer_on_resume"] = True

    if args.reset_epsilon_on_resume:
        epsilon_value = args.resume_epsilon_value
        if epsilon_value is None:
            epsilon_value = args.epsilon_start
        agent.reset_epsilon_schedule(epsilon=float(epsilon_value))
        info["applied"]["reset_epsilon_on_resume"] = True
        info["applied"]["resume_epsilon_value"] = float(epsilon_value)

    info["after"] = agent.training_state_snapshot()
    return info


def maybe_enable_compile(agent: DQNAgent, use_compile: bool) -> None:
    if not use_compile:
        return
    if not hasattr(torch, "compile"):
        print("[WARN] torch.compile unavailable in this PyTorch version.")
        return
    try:
        agent.q_net = torch.compile(agent.q_net)
        agent.target_net = torch.compile(agent.target_net)
        print("[INFO] torch.compile enabled.")
    except Exception as e:
        print(f"[WARN] torch.compile failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train a DQN policy on MultiPadLander with CUDA.")
    parser.add_argument("--episodes", type=int, default=int(TRAIN_DEFAULTS["episodes"]))
    parser.add_argument("--max_steps_per_episode", type=int, default=None)
    parser.add_argument("--difficulty", type=str, default=str(ENV_DEFAULTS["difficulty"]))
    parser.add_argument("--seed", type=int, default=int(SEED_DEFAULTS["train_seed"]))

    parser.add_argument("--device", type=str, default=str(TRAIN_DEFAULTS["device"]))
    parser.add_argument("--learning_rate", type=float, default=float(TRAIN_DEFAULTS["learning_rate"]))
    parser.add_argument("--gamma", type=float, default=float(TRAIN_DEFAULTS["gamma"]))
    parser.add_argument("--batch_size", type=int, default=int(TRAIN_DEFAULTS["batch_size"]))
    parser.add_argument("--replay_capacity", type=int, default=int(TRAIN_DEFAULTS["replay_capacity"]))
    parser.add_argument("--min_replay_size", type=int, default=int(TRAIN_DEFAULTS["min_replay_size"]))

    parser.add_argument("--epsilon_start", type=float, default=float(TRAIN_DEFAULTS["epsilon_start"]))
    parser.add_argument("--epsilon_end", type=float, default=float(TRAIN_DEFAULTS["epsilon_end"]))
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(TRAIN_DEFAULTS["epsilon_decay_steps"]))
    parser.add_argument("--target_update_freq", type=int, default=int(TRAIN_DEFAULTS["target_update_freq"]))
    parser.add_argument("--grad_clip_norm", type=float, default=float(TRAIN_DEFAULTS["grad_clip_norm"]))
    parser.add_argument("--use_double_dqn", action="store_true", default=bool(TRAIN_DEFAULTS["use_double_dqn"]))

    parser.add_argument("--updates_per_step", type=int, default=int(TRAIN_DEFAULTS["updates_per_step"]))
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_compile", action="store_true")

    parser.add_argument("--eval_every", type=int, default=int(TRAIN_DEFAULTS["eval_every"]))
    parser.add_argument("--eval_episodes", type=int, default=int(EVAL_DEFAULTS["episodes"]))
    parser.add_argument("--save_every", type=int, default=int(TRAIN_DEFAULTS["save_every"]))
    parser.add_argument("--log_every", type=int, default=int(TRAIN_DEFAULTS["log_every"]))

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(resolve_output_root(PROJECT_CONFIG, PROJECT_ROOT)))
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--reset_epsilon_on_resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume_epsilon_value", type=float, default=None)
    parser.add_argument("--reset_optimizer_on_resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reset_global_step_on_resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reset_learn_step_on_resume", action=argparse.BooleanOptionalAction, default=None)
    add_curriculum_arguments(parser)

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    if args.batch_size < 128 and args.device.startswith("cuda"):
        print("[WARN] batch_size is small for CUDA training. Consider 128 or higher.")

    set_global_seed(
        args.seed,
        deterministic_torch=bool(SEED_DEFAULTS.get("torch_deterministic", False)),
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    default_run_name = build_run_name(
        get_run_name_template(PROJECT_CONFIG),
        profile="gpu",
        difficulty=args.difficulty,
        seed=args.seed,
    )

    curriculum_summary = run_curriculum_pipeline(
        args=args,
        config=PROJECT_CONFIG,
        script_path=os.path.abspath(__file__),
        project_root=PROJECT_ROOT,
        profile="gpu",
        default_run_name=default_run_name,
    )
    if curriculum_summary is not None:
        return

    run_name = args.run_name or default_run_name

    run_paths = prepare_run_paths(
        output_root=args.output_dir,
        run_name=run_name,
        checkpoints_subdir=str(OUTPUT_DEFAULTS["checkpoints_subdir"]),
        logs_subdir=str(OUTPUT_DEFAULTS["logs_subdir"]),
        figures_subdir=str(OUTPUT_DEFAULTS["figures_subdir"]),
    )
    write_json(run_paths.config_path, vars(args))

    env = MultiPadLanderEnv(
        **build_env_kwargs(
            PROJECT_CONFIG,
            difficulty=args.difficulty,
            render_mode=None,
            seed=args.seed,
        )
    )

    agent = make_agent(env, args.device, args)
    agent.set_train_mode()
    maybe_enable_compile(agent, args.use_compile)
    resume_info = {
        "resume_path": None,
        "applied": {
            "reset_epsilon_on_resume": False,
            "resume_epsilon_value": None,
            "reset_optimizer_on_resume": False,
            "reset_global_step_on_resume": False,
            "reset_learn_step_on_resume": False,
        },
    }

    print(f"[INFO] Q-net device: {next(agent.q_net.parameters()).device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    if args.resume_path:
        print(f"[INFO] Loading checkpoint from: {args.resume_path}")
        agent.load(args.resume_path)
        agent.set_train_mode()
        resume_info = apply_resume_transition(agent, args)
        print(f"[INFO] Resume transition: {resume_info['applied']}")

    recent_returns = deque(maxlen=20)

    print("=" * 90)
    print("Training single-policy DQN (CUDA)")
    print(f"run_dir: {run_paths.run_dir}")
    print(f"obs_dim: {env.observation_space.shape[0]}")
    print(f"action_dim: {env.action_space.n}")
    print(f"difficulty: {args.difficulty}")
    print(f"episodes: {args.episodes}")
    print(f"device: {args.device}")
    print(f"batch_size: {args.batch_size}")
    print(f"updates_per_step: {args.updates_per_step}")
    print(f"use_amp: {args.use_amp}")
    print(f"use_compile: {args.use_compile}")
    print("=" * 90)

    best_eval_return_mean = -float("inf")
    best_eval_success_rate = 0.0
    max_eval_return_mean_observed = -float("inf")
    max_eval_success_rate_observed = 0.0
    best_eval_selection_key = None
    train_start_time = time.time()
    total_env_steps = 0
    total_gradient_updates = 0
    max_abs_q_observed = 0.0
    max_abs_td_error_observed = 0.0
    max_loss_observed = 0.0

    for episode in range(1, args.episodes + 1):
        ep_start_time = time.time()

        state, info = env.reset(seed=args.seed + episode)
        done = False
        episode_return = 0.0
        episode_steps = 0

        loss_values = []
        td_values = []
        q_values = []
        max_abs_q_values = []
        target_means = []
        target_max_abs_values = []
        final_info = {}
        final_truncated = False

        while not done:
            action = agent.select_action(state, greedy=False)
            next_state, reward, terminated, truncated, info = env.step(action)

            done_flag = bool(terminated or truncated)
            agent.store_transition(state, action, reward, next_state, done_flag)
            agent.global_step += 1
            agent._update_epsilon()

            if agent.can_learn():
                for _ in range(args.updates_per_step):
                    stats = agent.learn()
                    if stats is not None:
                        loss_values.append(stats["loss"])
                        td_values.append(stats["td_error"])
                        q_values.append(stats["mean_q"])
                        max_abs_q_values.append(stats["max_abs_q"])
                        target_means.append(stats["target_mean"])
                        target_max_abs_values.append(stats["target_max_abs"])
                        max_abs_q_observed = max(max_abs_q_observed, float(stats["max_abs_q"]))
                        max_abs_td_error_observed = max(max_abs_td_error_observed, float(stats["td_error"]))
                        max_loss_observed = max(max_loss_observed, float(stats["loss"]))
                        total_gradient_updates += 1

            state = next_state
            episode_return += reward
            episode_steps += 1
            total_env_steps += 1
            final_info = info
            final_truncated = truncated

            done = terminated or truncated

            if args.max_steps_per_episode is not None and episode_steps >= args.max_steps_per_episode:
                final_truncated = True
                done = True

        recent_returns.append(episode_return)
        ep_time = time.time() - ep_start_time
        env_steps_per_sec = episode_steps / max(ep_time, 1e-8)

        outcome = classify_info_outcome(final_info, final_truncated)

        row = {
            "episode": episode,
            "global_step": agent.global_step,
            "return": float(episode_return),
            "length": int(episode_steps),
            "epsilon": float(agent.epsilon),
            "loss_mean": float(np.mean(loss_values)) if loss_values else "",
            "td_error_mean": float(np.mean(td_values)) if td_values else "",
            "q_mean": float(np.mean(q_values)) if q_values else "",
            "max_abs_q": float(np.max(max_abs_q_values)) if max_abs_q_values else "",
            "target_mean": float(np.mean(target_means)) if target_means else "",
            "target_max_abs": float(np.max(target_max_abs_values)) if target_max_abs_values else "",
            "recent_return_mean_20": float(np.mean(recent_returns)),
            "outcome": outcome,
            "env_steps_per_sec": float(env_steps_per_sec),
            "gradient_updates": int(total_gradient_updates),
        }

        append_csv_row(
            run_paths.episode_csv_path,
            row,
            fieldnames=[
                "episode",
                "global_step",
                "return",
                "length",
                "epsilon",
                "loss_mean",
                "td_error_mean",
                "q_mean",
                "max_abs_q",
                "target_mean",
                "target_max_abs",
                "recent_return_mean_20",
                "outcome",
                "env_steps_per_sec",
                "gradient_updates",
            ],
        )

        if episode % args.log_every == 0 or episode == 1:
            print(
                f"[Episode {episode:04d}] "
                f"return={episode_return:8.2f} | "
                f"len={episode_steps:4d} | "
                f"eps={agent.epsilon:6.4f} | "
                f"recent20={np.mean(recent_returns):8.2f} | "
                f"env_sps={env_steps_per_sec:7.2f} | "
                f"updates={total_gradient_updates:7d} | "
                f"outcome={outcome}"
            )

        if torch.cuda.is_available() and episode % args.log_every == 0:
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"[GPU] allocated={mem_alloc:.2f} GB | reserved={mem_reserved:.2f} GB")

        if episode % args.save_every == 0:
            ckpt_path = run_paths.checkpoints_dir / f"episode_{episode}.pt"
            agent.save(ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

        if episode % args.eval_every == 0:
            metrics = evaluate_agent(
                agent=agent,
                episodes=args.eval_episodes,
                seed_base=episode_seed(
                    args.seed,
                    episode,
                    offset=int(SEED_DEFAULTS["eval_seed_offset"]),
                ),
                env_kwargs=build_env_kwargs(
                    PROJECT_CONFIG,
                    difficulty=args.difficulty,
                    render_mode=None,
                    seed=args.seed,
                ),
            )
            metrics["episode"] = episode
            metrics["global_step"] = agent.global_step

            append_jsonl_row(run_paths.eval_jsonl_path, metrics)
            agent.set_train_mode()

            print(
                f"[EVAL {episode:04d}] "
                f"return_mean={metrics['eval_return_mean']:.2f} | "
                f"success_rate={metrics['eval_success_rate']:.3f} | "
                f"len_mean={metrics['eval_length_mean']:.2f} | "
                f"outcomes={metrics['eval_outcomes']}"
            )

            max_eval_return_mean_observed = max(max_eval_return_mean_observed, float(metrics["eval_return_mean"]))
            max_eval_success_rate_observed = max(max_eval_success_rate_observed, float(metrics["eval_success_rate"]))

            is_better, selection_key = is_better_eval_checkpoint(metrics, best_eval_selection_key)
            if is_better:
                best_eval_selection_key = selection_key
                best_eval_return_mean = float(metrics["eval_return_mean"])
                best_eval_success_rate = float(metrics["eval_success_rate"])
                best_path = run_paths.checkpoints_dir / "best_eval.pt"
                agent.save(best_path)
                print(
                    f"[INFO] New best eval checkpoint saved: {best_path} "
                    f"(success={metrics['eval_success_rate']:.3f}, "
                    f"touchdown_speed={metrics['eval_touchdown_speed_mean']}, "
                    f"return={metrics['eval_return_mean']:.2f})"
                )

    total_time = time.time() - train_start_time

    final_path = run_paths.checkpoints_dir / "final.pt"
    agent.save(final_path)
    print(f"[INFO] Final checkpoint saved: {final_path}")

    summary = {
        "run_name": run_name,
        "difficulty": args.difficulty,
        "episodes": args.episodes,
        "seed": args.seed,
        "global_step": agent.global_step,
        "final_epsilon": agent.epsilon,
        "best_eval_return_mean": best_eval_return_mean,
        "best_eval_success_rate": best_eval_success_rate,
        "max_eval_return_mean_observed": max_eval_return_mean_observed,
        "max_eval_success_rate_observed": max_eval_success_rate_observed,
        "dqn_config": asdict(agent.config),
        "total_env_steps": total_env_steps,
        "total_gradient_updates": total_gradient_updates,
        "total_wall_time_sec": total_time,
        "avg_env_steps_per_sec": total_env_steps / max(total_time, 1e-8),
        "resume_info": resume_info,
        "max_abs_q_observed": max_abs_q_observed,
        "max_abs_td_error_observed": max_abs_td_error_observed,
        "max_loss_observed": max_loss_observed,
    }

    figure_files = create_run_figures(str(run_paths.run_dir))
    summary["figure_files"] = figure_files
    if recent_returns:
        summary["final_recent_return_mean_20"] = float(np.mean(recent_returns))

    write_json(run_paths.summary_path, summary)

    env.close()
    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
