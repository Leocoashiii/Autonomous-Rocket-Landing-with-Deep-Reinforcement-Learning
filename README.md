# Adaptive Landing Site Selection and Control for Autonomous Rocket Landing

## Overview

This project focuses on autonomous rocket landing in a static multi-pad environment. Each episode contains several candidate landing pads with different geometry and risk, and the controller must land on the designated target pad safely.

Core assumptions:

- actual candidate pads default to `3-5` per episode
- observation encoding uses `5` pad slots by default
- `38`-dimensional observation: `8` rocket features + `5 x 6` pad features
- `6` discrete actions
- standard difficulties: `baseline`, `level1`, `level2`, `level3`
- optional warm-start difficulty: `train_easy`

## Repository Layout

Core repository files:

- `.gitignore`: ignores caches, notebook checkpoints, and generated outputs while preserving `outputs/.gitkeep`
- `README.md`: project overview, repository map, and command entrypoints
- `requirements.txt`: Python dependencies for environment simulation, training, evaluation, and plotting
- `config.yaml`: shared runtime configuration for rewards, DQN defaults, curriculum settings, environment defaults, evaluation defaults, output paths, seeds, and training profiles

Agent package:

- `agents/__init__.py`: public exports for the agent package
- `agents/dqn_agent.py`: DQN / Double DQN model, replay buffer, checkpointing, and agent construction helpers

Environment package:

- `env/__init__.py`: public exports for environment-facing modules
- `env/multi_pad_lander.py`: main Gymnasium-style Box2D rocket-landing environment, including terrain generation, pad sampling, stepping, rendering, reward logic, and trace capture
- `env/reward_utils.py`: shared action labels, observation feature names, reward component names, and outcome-classification helpers
- `env/show_env.py`: command-line environment visualizer for random or fixed-action rollouts without loading a checkpoint

Training package:

- `training/curriculum.py`: shared helpers for multi-stage curriculum training and checkpoint handoff between stages
- `training/logger.py`: run-directory creation plus JSON, CSV, and JSONL artifact writing helpers
- `training/train_single.py`: CUDA-oriented single-policy training entrypoint
- `training/train_single_noGPU.py`: CPU-oriented single-policy training entrypoint

Evaluation package:

- `evaluation/eval.py`: checkpoint loading, deterministic evaluation, metric aggregation, and evaluation CLI
- `evaluation/visualize.py`: plotting entrypoint that delegates to the shared plotting helpers
- `evaluation/watch_best.py`: rendered playback for a trained checkpoint in the environment

Utilities:

- `utils/plotting.py`: reusable plotting helpers for training and evaluation outputs
- `utils/project_config.py`: config loading, config section helpers, environment kwargs assembly, and output-root resolution
- `utils/seeds.py`: global seed setup and per-episode seed helpers

Notebook and outputs:

- `notebooks/ablation_analysis.ipynb`: analysis notebook for experiment comparison and report preparation
- `outputs/.gitkeep`: keeps the output directory in version control when no generated files are present
- `outputs/`: generated run directories containing checkpoints, logs, figures, summaries, and saved training configurations

## Environment Launch Modes

Run all commands from the repository root.

Environment defaults:

- training and evaluation use the runtime config in `config.yaml`, where `difficulty=baseline`, `render_mode=None`, `min_pads=3`, `max_pads=5`, and `max_encoded_pads=5`
- this means training/evaluation episodes sample an actual pad count between `3` and `5`
- the observation always reserves `5` pad feature slots
- the dedicated visualizer `env/show_env.py` defaults to a fixed `4` displayed pads through `--pad-count 4`, which is useful for consistent demos

Accepted difficulty values for all environment-backed scripts:

- canonical multi-pad presets: `train_easy`, `baseline`, `level1`, `level2`, `level3`
- canonical single-pad presets: `single_pad_easy`, `single_pad_level1`, `single_pad_level2`, `single_pad_level3`
- additional aliases accepted by the environment: `easy`, `traineasy`, `warmup`, `warm_start`, `singlepad`, `singlepadeasy`, `widepadeasy`, `wide_pad_easy`, `singlepadlevel1`, `singlepad1`, `widepadlevel1`, `singlepadlevel2`, `singlepad2`, `widepadlevel2`, `singlepadlevel3`, `singlepad3`, `widepadlevel3`, `0`, `1`, `2`, `3`, `none`, `nofwind`, `weakwind`, `weakwind+gust`, `weakwindgust`, `strongwind`

Accepted render modes:

- `none`: headless stepping
- `human`: opens a live pygame window
- `rgb_array`: renders into arrays without opening a live window

Environment display commands:

- `python3 env/show_env.py`
- `python3 env/show_env.py --difficulty baseline --pad-count 4 --policy random`
- `python3 env/show_env.py --difficulty baseline --min-pads 3 --max-pads 5 --policy random`
- `python3 env/show_env.py --difficulty level3 --wind-enabled --policy random`
- `python3 env/show_env.py --difficulty single_pad_level3 --policy main_engine`
- `python3 env/show_env.py --render-mode rgb_array --trace-dir /tmp/env_traces`
- `python3 env/multi_pad_lander.py`
- `python3 evaluation/watch_best.py --checkpoint outputs/single_pad_level3_probe_retrain/checkpoints/best_eval.pt --difficulty baseline`
- `python3 evaluation/watch_best.py --checkpoint outputs/single_pad_level3_probe_retrain/checkpoints/best_eval.pt --difficulty baseline --episodes 5 --device cpu --touchdown_hold_seconds 2.0`

## Training, Evaluation, and Plotting Commands

Command help:

- `python3 env/show_env.py --help`
- `python3 training/train_single.py --help`
- `python3 training/train_single_noGPU.py --help`
- `python3 evaluation/eval.py --help`
- `python3 evaluation/watch_best.py --help`
- `python3 evaluation/visualize.py --help`

CPU training:

- `python3 training/train_single_noGPU.py --difficulty baseline`
- `python3 training/train_single_noGPU.py --difficulty level2 --use_double_dqn --run_name cpu_level2_run`
- `python3 training/train_single_noGPU.py --resume_path outputs/<run_name>/checkpoints/final.pt --difficulty baseline`
- `python3 training/train_single_noGPU.py --curriculum --use_double_dqn`
- `python3 training/train_single_noGPU.py --curriculum --curriculum_stages train_easy,level1,baseline --curriculum_stage_episodes 3000,2500,3500`

CUDA training:

- `python3 training/train_single.py --difficulty baseline --device cuda`
- `python3 training/train_single.py --difficulty baseline --device cuda --use_double_dqn --use_amp --use_compile`
- `python3 training/train_single.py --resume_path outputs/<run_name>/checkpoints/final.pt --difficulty baseline --device cuda`
- `python3 training/train_single.py --curriculum --use_double_dqn --device cuda`

Checkpoint evaluation:

- `python3 evaluation/eval.py --checkpoint outputs/baseline_v4_probe_5k/checkpoints/final.pt --difficulty baseline --episodes 20`
- `python3 evaluation/eval.py --checkpoint outputs/single_pad_level3_probe_retrain/checkpoints/best_eval.pt --difficulty level2 --episodes 50 --device cpu --output-json outputs/single_pad_level3_probe_retrain/manual_eval.json`

Playback:

- `python3 evaluation/watch_best.py --checkpoint outputs/single_pad_level3_probe_retrain/checkpoints/best_eval.pt --difficulty baseline`
- `python3 evaluation/watch_best.py --checkpoint outputs/single_pad_level3_probe_retrain/checkpoints/best_eval.pt --difficulty baseline --episodes 5 --device cpu --touchdown_hold_seconds 2.0`

Plotting:

- `python3 evaluation/visualize.py --run_dir outputs/single_pad_level3_probe_retrain`
- `python3 evaluation/visualize.py --run_dir outputs/baseline_v4_probe_5k --ma_window 50`

## CLI Parameter Reference

`env/multi_pad_lander.py`

- no command-line parameters
- purpose: legacy hard-coded random visual demo using `render_mode="human"`, `difficulty_level="baseline"`, `min_pads=3`, `max_pads=5`, `max_encoded_pads=5`, and `seed=111`

`env/show_env.py`

- `--episodes <int>`: number of episodes to display, default `1`
- `--difficulty <str>`: environment preset name or alias, default `baseline`
- `--seed <int>`: base seed, default `111`
- `--render-mode {human,rgb_array,none}`: rendering mode, default `human`
- `--policy {random,noop,left_thruster,main_engine,right_thruster,main_engine_left,main_engine_right}`: action policy used during display, default `random`
- `--pad-count <int>`: fixed pad count used only when `--min-pads` and `--max-pads` are both omitted, default `4`
- `--min-pads <int>`: optional lower bound for sampled pad count
- `--max-pads <int>`: optional upper bound for sampled pad count
- `--max-encoded-pads <int>`: observation encoding slots for pads, default `5`
- `--world-width <float>`: environment width, default `42.0`
- `--world-height <float>`: environment height, default `28.0`
- `--pad-width-min <float>`: minimum half-width sampling bound, default `1.2`
- `--pad-width-max <float>`: maximum half-width sampling bound, default `2.4`
- `--pad-slope-min <float>`: minimum slope sampling bound, default `-0.02`
- `--pad-slope-max <float>`: maximum slope sampling bound, default `0.02`
- `--pad-hazard-min <float>`: minimum hazard sampling bound, default `0.0`
- `--pad-hazard-max <float>`: maximum hazard sampling bound, default `1.0`
- `--min-pad-gap <float>`: minimum spacing between pads, default `2.5`
- `--terrain-segments <int>`: number of terrain segments, default `55`
- `--base-ground-y <float>`: nominal ground height, default `5.0`
- `--terrain-height-jitter <float>`: terrain roughness scale, default `6.0`
- `--water-level-offset <float>`: water height offset from base ground, default `2.0`
- `--pad-water-clearance <float>`: minimum pad clearance above water, default `1.0`
- `--max-episode-steps <int>`: truncation limit per episode, default `1000`
- `--wind-enabled` / `--no-wind-enabled`: optional override for wind enablement; default is preset-controlled
- `--wind-strength <float>`: optional override for base wind magnitude
- `--gust-strength <float>`: optional override for gust strength
- `--gust-decay <float>`: optional override for gust decay
- `--engine-dispersion-scale <float>`: engine noise scale, default `1.0`
- `--torque-disturbance <float>`: additional torque disturbance, default `0.0`
- `--hard-landing-speed <float>`: hard-landing speed threshold, default `1.75`
- `--hard-landing-vertical-speed <float>`: hard vertical landing threshold, default `1.25`
- `--trace-dir <path>`: optional directory for saved episode traces
- `--include-layout-in-trace` / `--no-include-layout-in-trace`: include terrain and pad layout in trace JSON, default enabled
- `--print-layout` / `--no-print-layout`: print pad layout summary to terminal, default enabled

`training/train_single_noGPU.py`

- `--episodes <int>`: training episode count, default from `config.yaml` CPU profile
- `--max_steps_per_episode <int>`: optional per-episode step limit override
- `--difficulty <str>`: environment preset name or alias, default from `config.yaml`
- `--seed <int>`: training seed, default from `config.yaml`
- `--device <str>`: requested device; the script forces CPU even if another value is passed
- `--learning_rate <float>`: optimizer learning rate
- `--gamma <float>`: discount factor
- `--batch_size <int>`: replay batch size
- `--replay_capacity <int>`: replay buffer capacity
- `--min_replay_size <int>`: minimum replay size before updates begin
- `--epsilon_start <float>`: initial epsilon
- `--epsilon_end <float>`: final epsilon
- `--epsilon_decay_steps <int>`: epsilon decay horizon
- `--target_update_freq <int>`: target network sync frequency
- `--grad_clip_norm <float>`: gradient clipping norm
- `--use_double_dqn`: enable Double DQN
- `--updates_per_step <int>`: number of gradient updates per environment step
- `--eval_every <int>`: evaluation interval in episodes
- `--eval_episodes <int>`: evaluation episode count
- `--save_every <int>`: checkpoint save interval in episodes
- `--log_every <int>`: terminal logging interval in episodes
- `--run_name <str>`: optional explicit run directory name
- `--output_dir <path>`: output root directory
- `--resume_path <path>`: checkpoint path for resume
- `--reset_epsilon_on_resume` / `--no-reset_epsilon_on_resume`: whether to reset epsilon after loading a checkpoint
- `--resume_epsilon_value <float>`: explicit epsilon value used when resetting epsilon on resume
- `--reset_optimizer_on_resume` / `--no-reset_optimizer_on_resume`: whether to recreate optimizer state on resume
- `--reset_global_step_on_resume` / `--no-reset_global_step_on_resume`: whether to reset global step counter on resume
- `--reset_learn_step_on_resume` / `--no-reset_learn_step_on_resume`: whether to reset gradient-update counter on resume
- `--curriculum`: switch from single-stage training to curriculum training
- `--curriculum_stages <csv>`: comma-separated difficulty list for curriculum mode
- `--curriculum_stage_episodes <csv>`: comma-separated episode counts aligned with `--curriculum_stages`

`training/train_single.py`

- includes every parameter from `training/train_single_noGPU.py`
- `--device <str>`: default is the CUDA profile device from `config.yaml`, typically `cuda`
- `--use_amp`: enable automatic mixed precision during training
- `--use_compile`: enable `torch.compile` when supported by the installed PyTorch version

`evaluation/eval.py`

- `--checkpoint <path>`: checkpoint to evaluate, required
- `--config <path>`: config file path, default `config.yaml`
- `--difficulty <str>`: evaluation difficulty override; defaults to the evaluation config when omitted
- `--episodes <int>`: number of evaluation episodes; defaults to the evaluation config when omitted
- `--seed <int>`: evaluation seed base, default `42`
- `--device <str>`: checkpoint loading device, default `cpu`
- `--output-json <path>`: optional JSON path for saved evaluation metrics

`evaluation/watch_best.py`

- `--checkpoint <path>`: checkpoint to replay, required
- `--config <path>`: config file path, default `config.yaml`
- `--difficulty <str>`: playback difficulty, default `baseline`
- `--seed <int>`: base seed for playback episodes, default `42`
- `--episodes <int>`: number of rendered episodes, default `3`
- `--device <str>`: checkpoint loading device, default `cpu`
- `--touchdown_hold_seconds <float>`: how long to hold the final touchdown frame, default `1.0`

`evaluation/visualize.py`

- `--run_dir <path>`: run directory containing `logs/episode_metrics.csv`, required
- `--ma_window <int>`: moving-average window used in return plotting, default `20`

## Notes

- The environment exposes reward breakdowns through `info["reward_breakdown"]`.
- Training outputs are written under `outputs/` with checkpoints, logs, summary JSON, and figures grouped by run name.
- Curriculum mode chains multiple difficulties by reusing the final checkpoint from the previous stage.
