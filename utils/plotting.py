import argparse
import json
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "comp4107_matplotlib"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_episode_metrics(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"episode metrics file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "episode" not in df.columns:
        raise ValueError("episode_metrics.csv must contain an 'episode' column")
    return df


def load_eval_metrics(jsonl_path: str) -> pd.DataFrame:
    if not os.path.exists(jsonl_path):
        return pd.DataFrame()

    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def plot_training_return(df: pd.DataFrame, output_dir: str, ma_window: int = 20) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df["episode"], df["return"], label="Episode Return")

    if len(df) >= ma_window:
        df["return_ma"] = df["return"].rolling(window=ma_window, min_periods=1).mean()
        plt.plot(df["episode"], df["return_ma"], label=f"Moving Average ({ma_window})")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Episode Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_return.png"))
    plt.close()


def plot_epsilon(df: pd.DataFrame, output_dir: str) -> None:
    if "epsilon" not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["episode"], df["epsilon"])
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "epsilon_decay.png"))
    plt.close()


def plot_loss_q(df: pd.DataFrame, output_dir: str) -> None:
    has_loss = "loss_mean" in df.columns and df["loss_mean"].notna().any()
    has_q = "q_mean" in df.columns and df["q_mean"].notna().any()

    if has_loss:
        plt.figure(figsize=(10, 6))
        loss_series = pd.to_numeric(df["loss_mean"], errors="coerce")
        plt.plot(df["episode"], loss_series)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Mean Training Loss per Episode")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_loss.png"))
        plt.close()

    if has_q:
        plt.figure(figsize=(10, 6))
        q_series = pd.to_numeric(df["q_mean"], errors="coerce")
        plt.plot(df["episode"], q_series)
        plt.xlabel("Episode")
        plt.ylabel("Mean Q")
        plt.title("Mean Q-Value per Episode")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mean_q.png"))
        plt.close()


def plot_eval_metrics(eval_df: pd.DataFrame, output_dir: str) -> None:
    if eval_df.empty:
        return

    if "episode" in eval_df.columns and "eval_return_mean" in eval_df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(eval_df["episode"], eval_df["eval_return_mean"])
        plt.xlabel("Episode")
        plt.ylabel("Eval Return Mean")
        plt.title("Evaluation Return Mean")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eval_return_mean.png"))
        plt.close()

    if "episode" in eval_df.columns and "eval_success_rate" in eval_df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(eval_df["episode"], eval_df["eval_success_rate"])
        plt.xlabel("Episode")
        plt.ylabel("Eval Success Rate")
        plt.title("Evaluation Success Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eval_success_rate.png"))
        plt.close()

    has_touchdown_speed = (
        "episode" in eval_df.columns
        and "eval_touchdown_speed_mean" in eval_df.columns
        and pd.to_numeric(eval_df["eval_touchdown_speed_mean"], errors="coerce").notna().any()
    )
    if has_touchdown_speed:
        plt.figure(figsize=(10, 6))
        touchdown_speed = pd.to_numeric(eval_df["eval_touchdown_speed_mean"], errors="coerce")
        plt.plot(eval_df["episode"], touchdown_speed)
        plt.xlabel("Episode")
        plt.ylabel("Touchdown Speed Mean")
        plt.title("Evaluation Touchdown Speed Mean")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eval_touchdown_speed_mean.png"))
        plt.close()


def build_eval_selection_key(row: pd.Series) -> tuple[float, ...]:
    eval_episodes = max(1, int(row.get("eval_episodes", 1)))
    outcomes = row.get("eval_outcomes")
    if not isinstance(outcomes, dict):
        outcomes = {}

    touchdown_speed = pd.to_numeric(
        pd.Series([row.get("eval_touchdown_speed_mean")]),
        errors="coerce",
    ).iloc[0]
    touchdown_speed_score = -1.0e9 if pd.isna(touchdown_speed) else -float(touchdown_speed)

    return (
        float(row.get("eval_success_rate", 0.0)),
        float(row.get("eval_any_safe_landing_rate", 0.0)),
        float(row.get("eval_touchdown_count", 0)) / eval_episodes,
        -float(outcomes.get("time_limit", 0)) / eval_episodes,
        -float(outcomes.get("out_of_bounds", 0)) / eval_episodes,
        -float(outcomes.get("body_crash_or_other", 0)) / eval_episodes,
        touchdown_speed_score,
        float(row.get("eval_return_mean", -1.0e9)),
    )


def create_run_figures(run_dir: str, ma_window: int = 20) -> list[str]:
    logs_dir = os.path.join(run_dir, "logs")
    figures_dir = os.path.join(run_dir, "figures")
    ensure_dir(figures_dir)

    episode_csv = os.path.join(logs_dir, "episode_metrics.csv")
    eval_jsonl = os.path.join(logs_dir, "eval_metrics.jsonl")

    df = load_episode_metrics(episode_csv)
    eval_df = load_eval_metrics(eval_jsonl)

    plot_training_return(df, figures_dir, ma_window=ma_window)
    plot_epsilon(df, figures_dir)
    plot_loss_q(df, figures_dir)
    plot_eval_metrics(eval_df, figures_dir)

    return sorted(
        filename
        for filename in os.listdir(figures_dir)
        if filename.endswith(".png")
    )


def print_summary(df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    print("=" * 70)
    print("Training summary")
    print(f"Episodes: {len(df)}")
    print(f"Return mean: {df['return'].mean():.2f}")
    print(f"Return std:  {df['return'].std():.2f}")
    print(f"Return max:  {df['return'].max():.2f}")
    print(f"Return min:  {df['return'].min():.2f}")

    if "recent_return_mean_20" in df.columns:
        recent = pd.to_numeric(df["recent_return_mean_20"], errors="coerce").dropna()
        if not recent.empty:
            print(f"Last moving-average return: {recent.iloc[-1]:.2f}")

    if "outcome" in df.columns:
        print("\nOutcome counts:")
        print(df["outcome"].value_counts().to_string())

    if not eval_df.empty:
        print("\nEvaluation summary:")
        if "eval_return_mean" in eval_df.columns:
            print(f"Max eval return mean observed: {eval_df['eval_return_mean'].max():.2f}")
        if "eval_success_rate" in eval_df.columns:
            print(f"Max eval success rate observed: {eval_df['eval_success_rate'].max():.3f}")

        best_idx = max(range(len(eval_df)), key=lambda i: build_eval_selection_key(eval_df.iloc[i]))
        best_row = eval_df.iloc[best_idx]
        if "episode" in best_row.index:
            print(f"Saved best-checkpoint episode: {int(best_row['episode'])}")
        if "eval_return_mean" in best_row.index:
            print(f"Saved best-checkpoint return mean: {float(best_row['eval_return_mean']):.2f}")
        if "eval_success_rate" in best_row.index:
            print(f"Saved best-checkpoint success rate: {float(best_row['eval_success_rate']):.3f}")
        if "eval_touchdown_speed_mean" in best_row.index:
            touchdown_speed = pd.to_numeric(
                pd.Series([best_row.get("eval_touchdown_speed_mean")]),
                errors="coerce",
            ).iloc[0]
            if not pd.isna(touchdown_speed):
                print(f"Saved best-checkpoint touchdown speed mean: {float(touchdown_speed):.3f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves for DQN runs.")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to one run directory, e.g. outputs/dqn_single_baseline_seed42_20260313_184242",
    )
    parser.add_argument("--ma_window", type=int, default=20)
    args = parser.parse_args()

    logs_dir = os.path.join(args.run_dir, "logs")
    figures_dir = os.path.join(args.run_dir, "figures")
    ensure_dir(figures_dir)

    episode_csv = os.path.join(logs_dir, "episode_metrics.csv")
    eval_jsonl = os.path.join(logs_dir, "eval_metrics.jsonl")

    df = load_episode_metrics(episode_csv)
    eval_df = load_eval_metrics(eval_jsonl)

    create_run_figures(args.run_dir, ma_window=args.ma_window)
    print_summary(df, eval_df)

    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
