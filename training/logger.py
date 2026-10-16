"""Shared logging and output-path helpers for experiment scripts."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    figures_dir: Path
    config_path: Path
    summary_path: Path
    episode_csv_path: Path
    eval_jsonl_path: Path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_run_name(
    template: str,
    *,
    profile: str,
    difficulty: str,
    seed: int,
    timestamp: str | None = None,
) -> str:
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    return template.format(
        profile=profile,
        difficulty=difficulty,
        seed=seed,
        timestamp=timestamp,
    )


def prepare_run_paths(
    *,
    output_root: str | Path,
    run_name: str,
    checkpoints_subdir: str = "checkpoints",
    logs_subdir: str = "logs",
    figures_subdir: str = "figures",
) -> RunPaths:
    output_root = ensure_dir(output_root)
    run_dir = ensure_dir(output_root / run_name)
    checkpoints_dir = ensure_dir(run_dir / checkpoints_subdir)
    logs_dir = ensure_dir(run_dir / logs_subdir)
    figures_dir = ensure_dir(run_dir / figures_subdir)

    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        figures_dir=figures_dir,
        config_path=run_dir / "train_config.json",
        summary_path=run_dir / "summary.json",
        episode_csv_path=logs_dir / "episode_metrics.csv",
        eval_jsonl_path=logs_dir / "eval_metrics.jsonl",
    )


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2)


def append_csv_row(path: str | Path, row: Mapping[str, Any], fieldnames: Iterable[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not file_exists:
            writer.writeheader()
        writer.writerow(dict(row))


def append_jsonl_row(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(payload)) + "\n")
