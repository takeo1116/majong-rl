"""Stage 1 Learner: shard データから PPO / 模倣学習で学習"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mahjong_rl.shard import ShardReader


class Learner:
    """shard データを読み込んで policy-value model を更新する

    モード:
        ppo: PPO clipped objective + MSE value loss + entropy bonus
        imitation: action ラベルへの cross-entropy loss
    """

    def __init__(self, config: dict, model: nn.Module, run_dir: Path,
                 device: torch.device | None = None):
        self._config = config
        self._run_dir = Path(run_dir)
        self._device = device or torch.device("cpu")
        self._model = model.to(self._device)

        tc = config.get("training", {})
        self._mode = tc.get("algorithm", "ppo")
        self._lr = tc.get("lr", 3e-4)
        self._batch_size = tc.get("batch_size", 256)
        self._epochs = tc.get("epochs", 4)
        self._gamma = tc.get("gamma", 0.99)
        self._gae_lambda = tc.get("gae_lambda", 0.95)
        self._clip_epsilon = tc.get("clip_epsilon", 0.2)
        self._value_loss_coef = tc.get("value_loss_coef", 0.5)
        self._entropy_coef = tc.get("entropy_coef", 0.01)
        self._max_grad_norm = tc.get("max_grad_norm", 0.5)

        self._optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)

    @property
    def mode(self) -> str:
        return self._mode

    def train(
        self,
        shard_dir: Path,
        num_epochs: int | None = None,
        filter_actor_type: str | None = None,
        imitation_filter: dict | None = None,
    ) -> dict:
        """shard データを読み込んで学習を実行する

        Args:
            filter_actor_type: 指定時、該当 actor_type のサンプルのみ学習に使う
            imitation_filter: imitation 用品質フィルタ設定
                - min_legal_actions: legal action 数の最小値
                - enabled: フィルタの有効/無効 (デフォルト True)

        Returns:
            metrics dict: mode, policy_loss, value_loss, entropy, total_steps
        """
        epochs = num_epochs if num_epochs is not None else self._epochs

        reader = ShardReader(shard_dir)
        data = reader.read_as_tensors(filter_actor_type=filter_actor_type)

        observations = torch.from_numpy(data["observations"]).to(self._device)
        legal_masks = torch.from_numpy(data["legal_masks"]).to(self._device)
        actions = torch.from_numpy(data["actions"]).long().to(self._device)
        rewards = torch.from_numpy(data["rewards"]).to(self._device)
        old_log_probs = torch.from_numpy(data["log_probs"]).to(self._device)
        old_values = torch.from_numpy(data["values"]).to(self._device)
        terminateds = torch.from_numpy(data["terminateds"]).to(self._device)

        n_before_filter = len(observations)

        # imitation 品質フィルタ適用
        filter_stats = None
        if self._mode == "imitation" and imitation_filter is not None:
            enabled = imitation_filter.get("enabled", True)
            if enabled:
                keep = self._apply_imitation_filter(legal_masks, imitation_filter)
                filter_stats = {
                    "before": n_before_filter,
                    "after": int(keep.sum()),
                    "removed": n_before_filter - int(keep.sum()),
                }
                observations = observations[keep]
                legal_masks = legal_masks[keep]
                actions = actions[keep]
                rewards = rewards[keep]
                old_log_probs = old_log_probs[keep]
                old_values = old_values[keep]
                terminateds = terminateds[keep]

        n = len(observations)
        if n == 0:
            result = {"mode": self._mode, "policy_loss": 0.0, "value_loss": 0.0,
                      "entropy": 0.0, "total_steps": 0, "num_updates": 0,
                      "device": str(self._device)}
            if filter_stats is not None:
                result["filter_stats"] = filter_stats
            return result

        if self._mode == "imitation":
            metrics = self._train_imitation(
                observations, legal_masks, actions, n, epochs)
        else:
            metrics = self._train_ppo(
                observations, legal_masks, actions, rewards,
                old_log_probs, old_values, terminateds, n, epochs)

        metrics["mode"] = self._mode
        metrics["total_steps"] = n
        metrics["device"] = str(self._device)
        if filter_stats is not None:
            metrics["filter_stats"] = filter_stats

        # metrics 保存
        metrics_dir = self._run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    @staticmethod
    def _apply_imitation_filter(
        legal_masks: torch.Tensor,
        filter_config: dict,
    ) -> torch.Tensor:
        """imitation 用品質フィルタを適用し、保持するインデックスの bool mask を返す

        Args:
            legal_masks: (N, 34) の legal mask テンソル
            filter_config: フィルタ設定 dict

        Returns:
            (N,) の bool テンソル — True のサンプルを保持
        """
        n = len(legal_masks)
        keep = torch.ones(n, dtype=torch.bool)

        # legal action 数フィルタ
        min_legal = filter_config.get("min_legal_actions", 0)
        if min_legal > 0:
            legal_counts = (legal_masks > 0).sum(dim=1)
            keep &= legal_counts >= min_legal

        return keep

    def _train_ppo(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        terminateds: torch.Tensor,
        n: int,
        epochs: int,
    ) -> dict:
        """PPO 学習"""
        advantages, returns = self._compute_gae(rewards, old_values, terminateds)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        num_updates = 0

        for _ in range(epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self._batch_size):
                end = min(start + self._batch_size, n)
                idx = indices[start:end]

                batch_obs = observations[idx]
                batch_masks = legal_masks[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                output = self._model(batch_obs, batch_masks)

                # ポリシーロス (PPO clipped)
                log_probs = torch.log_softmax(output.logits, dim=-1)
                action_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                ratio = torch.exp(action_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self._clip_epsilon, 1 + self._clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # バリューロス
                value = list(output.values.values())[0].squeeze(-1)
                value_loss = nn.functional.mse_loss(value, batch_returns)

                # エントロピーボーナス
                probs = torch.softmax(output.logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()

                # 合計ロス
                loss = policy_loss + self._value_loss_coef * value_loss - self._entropy_coef * entropy

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
                self._optimizer.step()
                num_updates += 1

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(entropy.item())

        return {
            "policy_loss": float(np.mean(all_policy_losses)) if all_policy_losses else 0.0,
            "value_loss": float(np.mean(all_value_losses)) if all_value_losses else 0.0,
            "entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
            "num_updates": num_updates,
        }

    def _train_imitation(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor,
        actions: torch.Tensor,
        n: int,
        epochs: int,
    ) -> dict:
        """模倣学習 (cross-entropy loss)"""
        all_policy_losses = []
        all_entropies = []
        num_updates = 0

        for _ in range(epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self._batch_size):
                end = min(start + self._batch_size, n)
                idx = indices[start:end]

                batch_obs = observations[idx]
                batch_masks = legal_masks[idx]
                batch_actions = actions[idx]

                output = self._model(batch_obs, batch_masks)

                # cross-entropy loss (legal mask 適用済みロジットに対して)
                policy_loss = nn.functional.cross_entropy(output.logits, batch_actions)

                # エントロピー（モニタリング用）
                log_probs = torch.log_softmax(output.logits, dim=-1)
                probs = torch.softmax(output.logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()

                loss = policy_loss - self._entropy_coef * entropy

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
                self._optimizer.step()
                num_updates += 1

                all_policy_losses.append(policy_loss.item())
                all_entropies.append(entropy.item())

        return {
            "policy_loss": float(np.mean(all_policy_losses)) if all_policy_losses else 0.0,
            "value_loss": 0.0,
            "entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
            "num_updates": num_updates,
        }

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        terminateds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE (Generalized Advantage Estimation) を計算する

        逐次計算のため CPU 上で行い、結果を self._device に移す。
        """
        # CPU 上で逐次計算
        r_cpu = rewards.cpu()
        v_cpu = values.cpu()
        t_cpu = terminateds.cpu()
        n = len(r_cpu)
        advantages = torch.zeros(n)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1 or t_cpu[t]:
                next_value = 0.0
            else:
                next_value = v_cpu[t + 1].item()

            delta = r_cpu[t] + self._gamma * next_value - v_cpu[t]
            if t_cpu[t]:
                last_gae = 0.0
            advantages[t] = last_gae = delta + self._gamma * self._gae_lambda * last_gae

        returns = advantages + v_cpu
        return advantages.to(self._device), returns.to(self._device)

    def save_checkpoint(self, tag: str = "") -> Path:
        """checkpoint を保存する"""
        ckpt_dir = self._run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        filename = f"checkpoint_{tag}.pt" if tag else "checkpoint.pt"
        path = ckpt_dir / filename
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
        }, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        """checkpoint を読み込む"""
        ckpt = torch.load(path, weights_only=False, map_location=self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
