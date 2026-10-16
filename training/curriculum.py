"""Shared helpers for multi-stage curriculum training entrypoints."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.logger import ensure_dir, write_json
from utils.project_config import get_curriculum_defaults


@dataclass(frozen=True)
class CurriculumStage:
    index: int
    difficulty: str
    episodes: int
    overrides: dict[str, Any]


@dataclass(frozen=True)
class ResumeTransitionPolicy:
    reset_epsilon_on_resume: bool = False
    resume_epsilon_value: float | None = None
    reset_optimizer_on_resume: bool = False
    reset_global_step_on_resume: bool = False
    reset_learn_step_on_resume: bool = False


def add_curriculum_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Run the configured multi-stage curriculum instead of a single-stage difficulty.",
    )
    parser.add_argument(
        "--curriculum_stages",
        type=str,
        default=None,
        help="Comma-separated curriculum difficulties, for example: train_easy,baseline",
    )
    parser.add_argument(
        "--curriculum_stage_episodes",
        type=str,
        default=None,
        help="Comma-separated episode counts aligned with --curriculum_stages.",
    )


def curriculum_requested(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "curriculum", False)
        or getattr(args, "curriculum_stages", None)
        or getattr(args, "curriculum_stage_episodes", None)
    )


def _parse_csv_items(raw: str | None) -> list[str]:
    if raw is None:
        return []
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


def _parse_episode_counts(raw: str | None) -> list[int]:
    counts: list[int] = []
    for item in _parse_csv_items(raw):
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"invalid curriculum episode count: {item!r}") from exc
        if value <= 0:
            raise ValueError("curriculum episode counts must be positive integers")
        counts.append(value)
    return counts


def _curriculum_transition_mapping(config: dict[str, Any]) -> dict[str, Any]:
    defaults = get_curriculum_defaults(config)
    default_plan = defaults.get("defaults", {})
    if not isinstance(default_plan, dict):
        raise ValueError("config curriculum.defaults must be a mapping")
    transition = default_plan.get("transition", {})
    if not isinstance(transition, dict):
        raise ValueError("config curriculum.defaults.transition must be a mapping")
    return transition


def _curriculum_stage_overrides(
        config: dict[str, Any],
        *,
        section: str = "defaults",
) -> list[dict[str, Any]]:
    defaults = get_curriculum_defaults(config)
    plan = defaults.get(section, {})
    if not isinstance(plan, dict):
        raise ValueError(f"config curriculum.{section} must be a mapping")
    raw_overrides = plan.get("stage_overrides", [])
    if raw_overrides is None:
        return []
    if not isinstance(raw_overrides, list):
        raise ValueError(f"config curriculum.{section}.stage_overrides must be a list")
    overrides: list[dict[str, Any]] = []
    for item in raw_overrides:
        if item is None:
            overrides.append({})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"config curriculum.{section}.stage_overrides entries must be mappings")
        overrides.append(dict(item))
    return overrides


def resolve_curriculum_plan(args: argparse.Namespace, config: dict[str, Any]) -> list[CurriculumStage] | None:
    if not curriculum_requested(args):
        return None

    defaults = get_curriculum_defaults(config)
    default_plan = defaults.get("defaults", {})
    if not isinstance(default_plan, dict):
        raise ValueError("config curriculum.defaults must be a mapping")

    stage_names = _parse_csv_items(getattr(args, "curriculum_stages", None))
    if not stage_names:
        stage_names = [str(item) for item in default_plan.get("stages", [])]

    stage_episodes = _parse_episode_counts(getattr(args, "curriculum_stage_episodes", None))
    if not stage_episodes:
        stage_episodes = [int(item) for item in default_plan.get("stage_episodes", [])]
    stage_overrides = _curriculum_stage_overrides(config, section="defaults")

    if not stage_names:
        raise ValueError("curriculum requested but no stages were provided or configured")
    if len(stage_names) < 2:
        raise ValueError("curriculum mode requires at least two stages")
    if len(stage_names) != len(stage_episodes):
        raise ValueError(
            "curriculum stages and curriculum stage episodes must have the same number of items"
        )
    if stage_overrides and len(stage_overrides) != len(stage_names):
        if getattr(args, "curriculum_stages", None) or getattr(args, "curriculum_stage_episodes", None):
            stage_overrides = [{} for _ in stage_names]
        else:
            raise ValueError(
                "config curriculum.defaults.stage_overrides must have the same number of items as stages"
            )
    if not stage_overrides:
        stage_overrides = [{} for _ in stage_names]

    return [
        CurriculumStage(
            index=index,
            difficulty=stage_name,
            episodes=stage_episodes[index - 1],
            overrides=stage_overrides[index - 1],
        )
        for index, stage_name in enumerate(stage_names, start=1)
    ]


def resolve_resume_transition_policy(
        args: argparse.Namespace,
        config: dict[str, Any],
) -> ResumeTransitionPolicy:
    transition = _curriculum_transition_mapping(config)

    def choose(key: str, fallback: Any) -> Any:
        value = getattr(args, key, None)
        if value is not None:
            return value
        return transition.get(key, fallback)

    return ResumeTransitionPolicy(
        reset_epsilon_on_resume=bool(choose("reset_epsilon_on_resume", False)),
        resume_epsilon_value=choose("resume_epsilon_value", None),
        reset_optimizer_on_resume=bool(choose("reset_optimizer_on_resume", False)),
        reset_global_step_on_resume=bool(choose("reset_global_step_on_resume", False)),
        reset_learn_step_on_resume=bool(choose("reset_learn_step_on_resume", False)),
    )


def build_curriculum_run_name(base_run_name: str) -> str:
    return f"curriculum_{base_run_name}"


def build_stage_run_name(base_run_name: str, stage: CurriculumStage) -> str:
    safe_difficulty = stage.difficulty.replace("/", "_")
    return f"{base_run_name}/stage{stage.index:02d}_{safe_difficulty}"


def _flag_name(key: str) -> str:
    return f"--{key}"


def _append_arg(cmd: list[str], key: str, value: Any) -> None:
    flag = _flag_name(key)
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if value is None:
        return
    cmd.extend([flag, str(value)])


def build_stage_command(
    *,
    script_path: str | Path,
    args: argparse.Namespace,
    stage: CurriculumStage,
    stage_run_name: str,
    resume_path: str | Path | None,
    resume_transition: ResumeTransitionPolicy,
) -> list[str]:
    skip_keys = {
        "curriculum",
        "curriculum_stages",
        "curriculum_stage_episodes",
        "difficulty",
        "episodes",
        "resume_path",
        "run_name",
        "reset_epsilon_on_resume",
        "resume_epsilon_value",
        "reset_optimizer_on_resume",
        "reset_global_step_on_resume",
        "reset_learn_step_on_resume",
    }
    reserved_override_keys = skip_keys | {
        "output_dir",
        "seed",
        "device",
    }
    command = [sys.executable, str(script_path)]
    for key, value in vars(args).items():
        if key in skip_keys:
            continue
        _append_arg(command, key, value)

    command.extend(
        [
            "--episodes",
            str(stage.episodes),
            "--difficulty",
            stage.difficulty,
            "--run_name",
            stage_run_name,
        ]
    )
    for key, value in stage.overrides.items():
        if key in reserved_override_keys:
            raise ValueError(f"curriculum stage overrides cannot set reserved key: {key}")
        _append_arg(command, str(key), value)
    if resume_path is not None:
        command.extend(["--resume_path", str(resume_path)])
        _append_arg(command, "reset_epsilon_on_resume", resume_transition.reset_epsilon_on_resume)
        _append_arg(command, "resume_epsilon_value", resume_transition.resume_epsilon_value)
        _append_arg(command, "reset_optimizer_on_resume", resume_transition.reset_optimizer_on_resume)
        _append_arg(command, "reset_global_step_on_resume", resume_transition.reset_global_step_on_resume)
        _append_arg(command, "reset_learn_step_on_resume", resume_transition.reset_learn_step_on_resume)
    return command


def run_curriculum_pipeline(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    script_path: str | Path,
    project_root: str | Path,
    profile: str,
    default_run_name: str,
) -> dict[str, Any] | None:
    stages = resolve_curriculum_plan(args, config)
    if stages is None:
        return None

    if getattr(args, "resume_path", None):
        raise ValueError("--resume_path cannot be combined with curriculum mode")

    base_run_name = args.run_name or build_curriculum_run_name(default_run_name)
    output_root = Path(args.output_dir)
    curriculum_root = ensure_dir(output_root / base_run_name)
    resume_transition = resolve_resume_transition_policy(args, config)

    plan_payload = {
        "run_name": base_run_name,
        "profile": profile,
        "seed": int(args.seed),
        "transition_policy": {
            "reset_epsilon_on_resume": resume_transition.reset_epsilon_on_resume,
            "resume_epsilon_value": resume_transition.resume_epsilon_value,
            "reset_optimizer_on_resume": resume_transition.reset_optimizer_on_resume,
            "reset_global_step_on_resume": resume_transition.reset_global_step_on_resume,
            "reset_learn_step_on_resume": resume_transition.reset_learn_step_on_resume,
        },
        "stages": [
            {
                "index": stage.index,
                "difficulty": stage.difficulty,
                "episodes": stage.episodes,
                "overrides": stage.overrides,
                "run_name": build_stage_run_name(base_run_name, stage),
            }
            for stage in stages
        ],
    }
    write_json(curriculum_root / "curriculum_config.json", plan_payload)

    stage_records: list[dict[str, Any]] = []
    previous_final_checkpoint: Path | None = None

    for stage in stages:
        stage_run_name = build_stage_run_name(base_run_name, stage)
        command = build_stage_command(
            script_path=script_path,
            args=args,
            stage=stage,
            stage_run_name=stage_run_name,
            resume_path=previous_final_checkpoint,
            resume_transition=resume_transition,
        )
        print("=" * 90)
        print(
            f"[CURRICULUM] Stage {stage.index}/{len(stages)} | "
            f"difficulty={stage.difficulty} | episodes={stage.episodes}"
        )
        if previous_final_checkpoint is not None:
            print(f"[CURRICULUM] Resume from: {previous_final_checkpoint}")
        print(f"[CURRICULUM] Command: {' '.join(command)}")

        completed = subprocess.run(
            command,
            cwd=str(project_root),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"curriculum stage {stage.index} failed with exit code {completed.returncode}"
            )

        stage_dir = output_root / stage_run_name
        stage_summary_path = stage_dir / "summary.json"
        stage_config_path = stage_dir / "train_config.json"
        stage_final_checkpoint = stage_dir / "checkpoints" / "final.pt"
        stage_best_eval_checkpoint = stage_dir / "checkpoints" / "best_eval.pt"

        if not stage_summary_path.exists():
            raise RuntimeError(f"missing stage summary: {stage_summary_path}")
        if not stage_config_path.exists():
            raise RuntimeError(f"missing stage config: {stage_config_path}")
        if not stage_final_checkpoint.exists():
            raise RuntimeError(f"missing stage final checkpoint: {stage_final_checkpoint}")

        stage_summary = json.loads(stage_summary_path.read_text(encoding="utf-8"))
        stage_config = json.loads(stage_config_path.read_text(encoding="utf-8"))
        stage_record = {
            "index": stage.index,
            "difficulty": stage.difficulty,
            "episodes": stage.episodes,
            "overrides": stage.overrides,
            "run_name": stage_run_name,
            "run_dir": str(stage_dir),
            "resume_path": None if previous_final_checkpoint is None else str(previous_final_checkpoint),
            "resume_transition": (
                None
                if previous_final_checkpoint is None
                else {
                    "reset_epsilon_on_resume": resume_transition.reset_epsilon_on_resume,
                    "resume_epsilon_value": resume_transition.resume_epsilon_value,
                    "reset_optimizer_on_resume": resume_transition.reset_optimizer_on_resume,
                    "reset_global_step_on_resume": resume_transition.reset_global_step_on_resume,
                    "reset_learn_step_on_resume": resume_transition.reset_learn_step_on_resume,
                }
            ),
            "summary_path": str(stage_summary_path),
            "config_path": str(stage_config_path),
            "final_checkpoint": str(stage_final_checkpoint),
            "best_eval_checkpoint": (
                str(stage_best_eval_checkpoint) if stage_best_eval_checkpoint.exists() else None
            ),
            "best_eval_return_mean": stage_summary.get("best_eval_return_mean"),
            "best_eval_success_rate": stage_summary.get("best_eval_success_rate"),
            "global_step": stage_summary.get("global_step"),
            "final_epsilon": stage_summary.get("final_epsilon"),
            "total_env_steps": stage_summary.get("total_env_steps"),
            "total_gradient_updates": stage_summary.get("total_gradient_updates"),
            "train_config": stage_config,
        }
        stage_records.append(stage_record)
        previous_final_checkpoint = stage_final_checkpoint

    assert previous_final_checkpoint is not None
    final_stage = stage_records[-1]
    curriculum_final_checkpoint = curriculum_root / "curriculum_final.pt"
    shutil.copy2(previous_final_checkpoint, curriculum_final_checkpoint)

    curriculum_best_eval_checkpoint: str | None = None
    final_stage_best = final_stage.get("best_eval_checkpoint")
    if final_stage_best:
        copied_best = curriculum_root / "curriculum_best_eval.pt"
        shutil.copy2(Path(final_stage_best), copied_best)
        curriculum_best_eval_checkpoint = str(copied_best)

    summary = {
        "type": "curriculum",
        "run_name": base_run_name,
        "profile": profile,
        "seed": int(args.seed),
        "stage_count": len(stage_records),
        "transition_policy": {
            "reset_epsilon_on_resume": resume_transition.reset_epsilon_on_resume,
            "resume_epsilon_value": resume_transition.resume_epsilon_value,
            "reset_optimizer_on_resume": resume_transition.reset_optimizer_on_resume,
            "reset_global_step_on_resume": resume_transition.reset_global_step_on_resume,
            "reset_learn_step_on_resume": resume_transition.reset_learn_step_on_resume,
        },
        "stages": stage_records,
        "final_stage_difficulty": final_stage["difficulty"],
        "final_stage_run_name": final_stage["run_name"],
        "final_checkpoint": str(curriculum_final_checkpoint),
        "best_eval_checkpoint": curriculum_best_eval_checkpoint,
        "best_stage_eval_return_mean": max(
            float(stage.get("best_eval_return_mean", float("-inf"))) for stage in stage_records
        ),
        "best_stage_eval_success_rate": max(
            float(stage.get("best_eval_success_rate", 0.0)) for stage in stage_records
        ),
        "total_env_steps": sum(
            int(stage.get("total_env_steps", 0) or 0) for stage in stage_records
        ),
        "total_gradient_updates": sum(
            int(stage.get("total_gradient_updates", 0) or 0) for stage in stage_records
        ),
    }
    write_json(curriculum_root / "curriculum_summary.json", summary)

    print("=" * 90)
    print(f"[DONE] Curriculum finished. Summary: {curriculum_root / 'curriculum_summary.json'}")
    print(f"[DONE] Final checkpoint: {curriculum_final_checkpoint}")
    if curriculum_best_eval_checkpoint is not None:
        print(f"[DONE] Final-stage best eval checkpoint: {curriculum_best_eval_checkpoint}")
    return summary
