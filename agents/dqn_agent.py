import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Agent parameter
@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int = 6

    # Network / optimization
    hidden_dims: Tuple[int, int, int] = (256, 128, 64)
    learning_rate: float = 1e-3
    gamma: float = 0.99

    # Replay buffer
    replay_capacity: int = 100_000
    batch_size: int = 64
    min_replay_size: int = 2_000

    # Epsilon-greedy
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 50_000

    # Target network
    target_update_freq: int = 100

    # Training stability
    grad_clip_norm: float = 10.0
    use_double_dqn: bool = True
    reward_clip_value: Optional[float] = 20.0
    preserve_terminal_reward_scale: bool = True
    target_q_clip_value: Optional[float] = 50.0
    target_value_clip_value: Optional[float] = 75.0
    q_value_hard_limit: float = 1_000.0
    td_error_hard_limit: float = 1_000.0
    loss_hard_limit: float = 10_000.0

    # Device / checkpointing
    device: str = "cpu"


# the network we build
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, int, int]):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# store the experience Q(s, a, r, s', done)
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "states": torch.as_tensor(self.states[idxs], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[idxs], dtype=torch.long, device=device),
            "rewards": torch.as_tensor(self.rewards[idxs], dtype=torch.float32, device=device),
            "next_states": torch.as_tensor(self.next_states[idxs], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[idxs], dtype=torch.float32, device=device),
        }
        return batch

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    """
    Single-policy DQN / Double DQN agent for MultiPadLander.

    The observation size depends on the environment configuration, so state_dim
    should be inferred from env.observation_space.shape[0] or from a reset()
    sample rather than hard-coded.
    """

    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.q_net = QNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.replay_buffer = ReplayBuffer(config.replay_capacity, config.state_dim)

        self.global_step = 0
        self.learn_step = 0
        self.epsilon = config.epsilon_start
        self.epsilon_decay_start_step = 0

    def _update_epsilon(self) -> None:
        effective_step = max(0, self.global_step - self.epsilon_decay_start_step)
        frac = min(1.0, effective_step / max(1, self.config.epsilon_decay_steps))
        self.epsilon = self.config.epsilon_start + frac * (
                self.config.epsilon_end - self.config.epsilon_start
        )

    @staticmethod
    def _maybe_clamp(values: torch.Tensor, clip_value: Optional[float]) -> torch.Tensor:
        if clip_value is None:
            return values
        return torch.clamp(values, min=-float(clip_value), max=float(clip_value))

    def _prepare_rewards_for_targets(
            self,
            rewards: torch.Tensor,
            dones: torch.Tensor,
    ) -> torch.Tensor:
        clipped_rewards = self._maybe_clamp(rewards, self.config.reward_clip_value)
        if (
            self.config.reward_clip_value is None
            or not self.config.preserve_terminal_reward_scale
        ):
            return clipped_rewards

        # Keep large terminal rewards intact so the agent can still strongly
        # distinguish safe touchdowns from hard crashes.
        terminal_mask = dones > 0.5
        return torch.where(terminal_mask, rewards, clipped_rewards)

    def _assert_finite_tensor(self, name: str, values: torch.Tensor) -> None:
        if not torch.isfinite(values).all():
            raise RuntimeError(f"non-finite values detected in {name}")

    def _assert_stable_learning_signal(
            self,
            *,
            loss_value: float,
            td_error_value: float,
            max_abs_q_value: float,
    ) -> None:
        if max_abs_q_value > float(self.config.q_value_hard_limit):
            raise RuntimeError(
                f"q-value instability detected: max_abs_q={max_abs_q_value:.4f} "
                f"exceeds limit={self.config.q_value_hard_limit:.4f}"
            )
        if td_error_value > float(self.config.td_error_hard_limit):
            raise RuntimeError(
                f"td-error instability detected: td_error={td_error_value:.4f} "
                f"exceeds limit={self.config.td_error_hard_limit:.4f}"
            )
        if loss_value > float(self.config.loss_hard_limit):
            raise RuntimeError(
                f"loss instability detected: loss={loss_value:.4f} "
                f"exceeds limit={self.config.loss_hard_limit:.4f}"
            )

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        state = np.asarray(state, dtype=np.float32)

        if (not greedy) and (random.random() < self.epsilon):
            return random.randrange(self.config.action_dim)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def store_transition(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def can_learn(self) -> bool:
        return len(self.replay_buffer) >= max(self.config.batch_size, self.config.min_replay_size)

    def learn(self) -> Optional[Dict[str, float]]:
        if not self.can_learn():
            return None

        batch = self.replay_buffer.sample(self.config.batch_size, self.device)

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        rewards_for_targets = self._prepare_rewards_for_targets(rewards, dones)

        # Q(s, a)
        q_values = self.q_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        self._assert_finite_tensor("q_sa", q_sa)

        with torch.no_grad():
            if self.config.use_double_dqn:
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1).values

            next_q = self._maybe_clamp(next_q, self.config.target_q_clip_value)
            targets = rewards_for_targets + self.config.gamma * (1.0 - dones) * next_q
            targets = self._maybe_clamp(targets, self.config.target_value_clip_value)

        self._assert_finite_tensor("targets", targets)

        loss = nn.functional.smooth_l1_loss(q_sa, targets)
        self._assert_finite_tensor("loss", loss.unsqueeze(0))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        self.learn_step += 1

        if self.learn_step % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        with torch.no_grad():
            td_error = (targets - q_sa).abs().mean().item()
            mean_q = q_sa.mean().item()
            max_abs_q = q_sa.abs().max().item()
            target_mean = targets.mean().item()
            target_max_abs = targets.abs().max().item()

        loss_value = float(loss.item())
        self._assert_stable_learning_signal(
            loss_value=loss_value,
            td_error_value=float(td_error),
            max_abs_q_value=float(max_abs_q),
        )

        return {
            "loss": loss_value,
            "td_error": float(td_error),
            "mean_q": float(mean_q),
            "max_abs_q": float(max_abs_q),
            "target_mean": float(target_mean),
            "target_max_abs": float(target_max_abs),
            "epsilon": float(self.epsilon),
            "buffer_size": float(len(self.replay_buffer)),
        }

    def step_env_transition(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            terminated: bool,
            truncated: bool,
    ) -> Optional[Dict[str, float]]:
        done = bool(terminated or truncated)

        self.store_transition(state, action, reward, next_state, done)
        self.global_step += 1
        self._update_epsilon()
        return self.learn()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "config": self.config.__dict__,
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "learn_step": self.learn_step,
            "epsilon": self.epsilon,
            "epsilon_decay_start_step": self.epsilon_decay_start_step,
        }
        torch.save(payload, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)

        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.global_step = int(checkpoint.get("global_step", 0))
        self.learn_step = int(checkpoint.get("learn_step", 0))
        self.epsilon = float(checkpoint.get("epsilon", self.config.epsilon_end))
        self.epsilon_decay_start_step = int(checkpoint.get("epsilon_decay_start_step", 0))

    def set_eval_mode(self) -> None:
        self.q_net.eval()
        self.target_net.eval()

    def set_train_mode(self) -> None:
        self.q_net.train()
        self.target_net.eval()

    def reset_optimizer(self) -> None:
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.learning_rate)

    def reset_global_step(self, value: int = 0) -> None:
        self.global_step = int(value)
        self.epsilon_decay_start_step = min(self.epsilon_decay_start_step, self.global_step)
        self._update_epsilon()

    def reset_learn_step(self, value: int = 0) -> None:
        self.learn_step = int(value)

    def reset_epsilon_schedule(
            self,
            *,
            epsilon: Optional[float] = None,
            decay_anchor_step: Optional[int] = None,
    ) -> None:
        if decay_anchor_step is None:
            decay_anchor_step = self.global_step
        self.epsilon_decay_start_step = int(decay_anchor_step)
        self.epsilon = float(self.config.epsilon_start if epsilon is None else epsilon)

    def training_state_snapshot(self) -> Dict[str, float]:
        return {
            "global_step": float(self.global_step),
            "learn_step": float(self.learn_step),
            "epsilon": float(self.epsilon),
            "epsilon_decay_start_step": float(self.epsilon_decay_start_step),
            "replay_buffer_size": float(len(self.replay_buffer)),
        }


def build_dqn_agent_from_env(env, device: str = "cpu") -> DQNAgent:
    """
    Convenience helper for building an agent with dimensions inferred directly
    from the environment.
    """
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )
    return DQNAgent(config)
