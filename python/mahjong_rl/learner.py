"""Stage 1 Learner: shard データから PPO / 模倣学習で学習"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mahjong_rl.profiler import Profiler
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

        # CQ-0130: imitation loss mode
        self._imitation_loss_mode = tc.get("imitation_loss_mode", "strict_top1")
        _VALID_LOSS_MODES = {"strict_top1", "tie_aware_best_set"}
        if self._imitation_loss_mode not in _VALID_LOSS_MODES:
            raise ValueError(
                f"Unknown imitation_loss_mode: {self._imitation_loss_mode!r}. "
                f"有効値: {_VALID_LOSS_MODES}")

        # CQ-0150: imitation value warm start
        ivw = tc.get("imitation_value_warmstart", {})
        self._imi_value_enabled = ivw.get("enabled", False)
        self._imi_value_coef = ivw.get("coef", 0.5)

        # CQ-0164: 立直後打牌除外
        eprd = tc.get("exclude_post_riichi_discards", {})
        self._exclude_post_riichi = eprd.get("enabled", False)

        self._optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def imitation_loss_mode(self) -> str:
        return self._imitation_loss_mode

    def train(
        self,
        shard_dir: Path,
        num_epochs: int | None = None,
        filter_actor_type: str | None = None,
        imitation_filter: dict | None = None,
        profiler: Profiler | None = None,
    ) -> dict:
        """shard データを読み込んで学習を実行する

        Args:
            filter_actor_type: 指定時、該当 actor_type のサンプルのみ学習に使う
            imitation_filter: imitation 用品質フィルタ設定
                - min_legal_actions: legal action 数の最小値
                - enabled: フィルタの有効/無効 (デフォルト True)
            profiler: 簡易プロファイラ (CQ-0102)

        Returns:
            metrics dict: mode, policy_loss, value_loss, entropy, total_steps
        """
        if profiler is None:
            profiler = Profiler(enabled=False)
        epochs = num_epochs if num_epochs is not None else self._epochs

        profiler.start("shard_read")
        reader = ShardReader(shard_dir)
        data = reader.read_as_tensors(filter_actor_type=filter_actor_type)

        observations = torch.from_numpy(data["observations"]).to(self._device)
        legal_masks = torch.from_numpy(data["legal_masks"]).to(self._device)
        actions = torch.from_numpy(data["actions"]).long().to(self._device)
        rewards = torch.from_numpy(data["rewards"]).to(self._device)
        old_log_probs = torch.from_numpy(data["log_probs"]).to(self._device)
        old_values = torch.from_numpy(data["values"]).to(self._device)
        terminateds = torch.from_numpy(data["terminateds"]).to(self._device)
        profiler.stop("shard_read")

        n_before_filter = len(observations)

        # CQ-0145: shanten_deltas
        raw_shanten_deltas = data.get("shanten_deltas")

        # CQ-0151: current_shantens (value head 補助特徴)
        raw_current_shantens = data.get("current_shantens")

        # CQ-0156: turn_numbers (巡目診断用)
        raw_turn_numbers = data.get("turn_numbers")

        # CQ-0160: reward components (診断用)
        raw_point_delta_rewards = data.get("point_delta_rewards")
        raw_shanten_delta_rewards = data.get("shanten_delta_rewards")

        # CQ-0163: is_post_riichi_discards
        raw_is_post_riichi_discards = data.get("is_post_riichi_discards")

        # teacher_best_masks (CQ-0125, CQ-0128)
        raw_teacher_best_masks = data.get("teacher_best_masks")
        tbm_shard_info = data.get("teacher_best_mask_shard_info", {})

        # imitation 品質フィルタ適用
        filter_stats = None
        keep = None
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

        # CQ-0164: 立直後打牌除外
        post_riichi_exclusion_stats = None
        if self._exclude_post_riichi and raw_is_post_riichi_discards is not None:
            pr_mask = raw_is_post_riichi_discards
            if keep is not None:
                pr_mask = pr_mask[keep.cpu().numpy() if isinstance(keep, torch.Tensor) else keep]
            exclude_mask = torch.from_numpy(~pr_mask).to(observations.device)
            n_excluded = int(pr_mask.sum())
            post_riichi_exclusion_stats = {
                "total_before_exclusion": len(observations),
                "excluded_post_riichi_discards": n_excluded,
                "used_samples": len(observations) - n_excluded,
            }
            observations = observations[exclude_mask]
            legal_masks = legal_masks[exclude_mask]
            actions = actions[exclude_mask]
            rewards = rewards[exclude_mask]
            old_log_probs = old_log_probs[exclude_mask]
            old_values = old_values[exclude_mask]
            terminateds = terminateds[exclude_mask]
            # numpy optional arrays も除外
            exclude_np = exclude_mask.cpu().numpy()
            if raw_shanten_deltas is not None:
                raw_shanten_deltas = raw_shanten_deltas[exclude_np]
            if raw_current_shantens is not None:
                raw_current_shantens = raw_current_shantens[exclude_np]
            if raw_turn_numbers is not None:
                raw_turn_numbers = raw_turn_numbers[exclude_np]
            if raw_point_delta_rewards is not None:
                raw_point_delta_rewards = raw_point_delta_rewards[exclude_np]
            if raw_shanten_delta_rewards is not None:
                raw_shanten_delta_rewards = raw_shanten_delta_rewards[exclude_np]
            if raw_is_post_riichi_discards is not None:
                raw_is_post_riichi_discards = raw_is_post_riichi_discards[exclude_np]
            if raw_teacher_best_masks is not None:
                raw_teacher_best_masks = raw_teacher_best_masks[exclude_np]

        n = len(observations)
        if n == 0:
            result = {"mode": self._mode, "policy_loss": 0.0, "value_loss": 0.0,
                      "entropy": 0.0, "total_steps": 0, "num_updates": 0,
                      "device": str(self._device)}
            if filter_stats is not None:
                result["filter_stats"] = filter_stats
            if post_riichi_exclusion_stats is not None:
                result["post_riichi_exclusion"] = post_riichi_exclusion_stats
            return result

        # CQ-0151: current_shantens を value_aux_features 用テンソルに変換
        current_shantens_t = None
        if raw_current_shantens is not None:
            cs_arr = raw_current_shantens
            if keep is not None:
                cs_arr = cs_arr[keep]
            # 正規化: shanten 0-8 → /8.0 で [0,1] 範囲
            current_shantens_t = torch.from_numpy(
                cs_arr.astype(np.float32) / 8.0).unsqueeze(-1).to(self._device)

        profiler.start("model_forward")
        if self._mode == "imitation":
            # teacher_best_masks 準備 (CQ-0125, CQ-0128, CQ-0130)
            teacher_best_masks_t = None
            teacher_best_set_status = "missing"
            if raw_teacher_best_masks is not None:
                teacher_best_masks_t = torch.from_numpy(
                    raw_teacher_best_masks).to(self._device)
                if keep is not None:
                    teacher_best_masks_t = teacher_best_masks_t[keep]
                teacher_best_set_status = "available"
            else:
                avail = tbm_shard_info.get("available", 0)
                total = tbm_shard_info.get("total", 0)
                if avail > 0 and avail < total:
                    teacher_best_set_status = "mixed"

            metrics = self._train_imitation(
                observations, legal_masks, actions, n, epochs,
                teacher_best_masks=teacher_best_masks_t,
                rewards=rewards, old_values=old_values,
                terminateds=terminateds,
                value_aux_features=current_shantens_t)
            metrics["teacher_best_set_status"] = teacher_best_set_status
            if tbm_shard_info:
                metrics["teacher_best_mask_shard_info"] = tbm_shard_info
            repro = self._compute_imitation_metrics(
                observations, legal_masks, actions, teacher_best_masks_t,
                value_aux_features=current_shantens_t)
            metrics.update(repro)
        else:
            metrics = self._train_ppo(
                observations, legal_masks, actions, rewards,
                old_log_probs, old_values, terminateds, n, epochs,
                shanten_deltas=raw_shanten_deltas,
                value_aux_features=current_shantens_t,
                turn_numbers=raw_turn_numbers,
                point_delta_rewards=raw_point_delta_rewards,
                shanten_delta_rewards=raw_shanten_delta_rewards,
                is_post_riichi_discards=raw_is_post_riichi_discards)
        profiler.stop("model_forward")

        metrics["mode"] = self._mode
        metrics["total_steps"] = n
        metrics["device"] = str(self._device)
        if filter_stats is not None:
            metrics["filter_stats"] = filter_stats
        if post_riichi_exclusion_stats is not None:
            metrics["post_riichi_exclusion"] = post_riichi_exclusion_stats

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
        shanten_deltas: np.ndarray | None = None,
        value_aux_features: torch.Tensor | None = None,
        turn_numbers: np.ndarray | None = None,
        point_delta_rewards: np.ndarray | None = None,
        shanten_delta_rewards: np.ndarray | None = None,
        is_post_riichi_discards: np.ndarray | None = None,
    ) -> dict:
        """PPO 学習"""
        advantages, returns = self._compute_gae(rewards, old_values, terminateds)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        # CQ-0135: 診断統計用テンソル収集
        all_ratios = []
        all_new_values = []
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
                batch_vaux = value_aux_features[idx] if value_aux_features is not None else None

                output = self._model(batch_obs, batch_masks, value_aux_features=batch_vaux)

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
                # CQ-0135: バッチ単位の ratio / new_value を記録
                all_ratios.append(ratio.detach().cpu())
                all_new_values.append(value.detach().cpu())

        metrics = {
            "policy_loss": float(np.mean(all_policy_losses)) if all_policy_losses else 0.0,
            "value_loss": float(np.mean(all_value_losses)) if all_value_losses else 0.0,
            "entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
            "num_updates": num_updates,
        }

        # CQ-0135: PPO 診断統計
        if num_updates > 0:
            metrics["ppo_diag"] = self._compute_ppo_diagnostics(
                advantages, returns, old_values,
                all_ratios, all_new_values,
                clip_epsilon=self._clip_epsilon,
            )

            # CQ-0155: per-sample final new_values (学習完了後の value 予測)
            final_new_values_parts: list[torch.Tensor] = []
            self._model.eval()
            with torch.no_grad():
                for s in range(0, n, self._batch_size):
                    e = min(s + self._batch_size, n)
                    bvaux = value_aux_features[s:e] if value_aux_features is not None else None
                    out = self._model(observations[s:e], legal_masks[s:e], value_aux_features=bvaux)
                    final_new_values_parts.append(list(out.values.values())[0].squeeze(-1).cpu())
            self._model.train()
            final_new_values_t = torch.cat(final_new_values_parts)

            # CQ-0145/CQ-0155/CQ-0160/CQ-0163: shanten 変化別 advantage 診断
            if shanten_deltas is not None:
                metrics["ppo_diag"]["shanten_diag"] = (
                    self._compute_shanten_conditioned_diag(
                        advantages, returns, old_values, shanten_deltas,
                        new_values=final_new_values_t,
                        rewards=rewards, terminateds=terminateds,
                        gamma=self._gamma,
                        point_delta_rewards=point_delta_rewards,
                        shanten_delta_rewards=shanten_delta_rewards,
                        is_post_riichi_discards=is_post_riichi_discards))

            # CQ-0156: 巡目バケット別診断
            if turn_numbers is not None:
                metrics["ppo_diag"]["turn_diag"] = (
                    self._compute_turn_conditioned_diag(
                        advantages, returns, old_values, turn_numbers,
                        new_values=final_new_values_t))

        return metrics

    def _train_imitation(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor,
        actions: torch.Tensor,
        n: int,
        epochs: int,
        teacher_best_masks: torch.Tensor | None = None,
        rewards: torch.Tensor | None = None,
        old_values: torch.Tensor | None = None,
        terminateds: torch.Tensor | None = None,
        value_aux_features: torch.Tensor | None = None,
    ) -> dict:
        """模倣学習 (CQ-0130: loss mode 切替, CQ-0150: joint value warm start)

        imitation_loss_mode:
            strict_top1: 教師 action への cross-entropy loss
            tie_aware_best_set: -log(sum_{a in best_set} pi(a))
        """
        if self._imitation_loss_mode == "tie_aware_best_set" and teacher_best_masks is None:
            raise ValueError(
                "imitation_loss_mode='tie_aware_best_set' requires teacher_best_masks, "
                "but none were found in shard data")

        # CQ-0150: joint value warm start — returns 計算
        returns = None
        if self._imi_value_enabled and rewards is not None and old_values is not None and terminateds is not None:
            _, returns = self._compute_gae(rewards, old_values, terminateds)

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
                batch_vaux = value_aux_features[idx] if value_aux_features is not None else None

                output = self._model(batch_obs, batch_masks, value_aux_features=batch_vaux)

                # loss 計算 (CQ-0130)
                if self._imitation_loss_mode == "strict_top1":
                    policy_loss = nn.functional.cross_entropy(output.logits, batch_actions)
                else:  # tie_aware_best_set
                    batch_tbm = teacher_best_masks[idx]
                    probs = torch.softmax(output.logits, dim=-1)
                    best_set_prob = (probs * batch_tbm).sum(dim=-1)
                    policy_loss = -torch.log(best_set_prob + 1e-8).mean()

                # エントロピー（モニタリング用）
                log_probs = torch.log_softmax(output.logits, dim=-1)
                probs_for_ent = torch.softmax(output.logits, dim=-1)
                entropy = -(probs_for_ent * log_probs).sum(dim=-1).mean()

                # CQ-0150: value loss
                value_loss_t = torch.tensor(0.0, device=self._device)
                if returns is not None:
                    batch_returns = returns[idx]
                    value = list(output.values.values())[0].squeeze(-1)
                    value_loss_t = nn.functional.mse_loss(value, batch_returns)

                loss = policy_loss + self._imi_value_coef * value_loss_t - self._entropy_coef * entropy

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
                self._optimizer.step()
                num_updates += 1

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss_t.item())
                all_entropies.append(entropy.item())

        return {
            "policy_loss": float(np.mean(all_policy_losses)) if all_policy_losses else 0.0,
            "value_loss": float(np.mean(all_value_losses)) if all_value_losses else 0.0,
            "entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
            "num_updates": num_updates,
            "imitation_loss_mode": self._imitation_loss_mode,
            "imitation_value_warmstart": {
                "enabled": self._imi_value_enabled,
                "coef": self._imi_value_coef,
            },
        }

    @torch.no_grad()
    def _compute_imitation_metrics(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor,
        actions: torch.Tensor,
        teacher_best_masks: torch.Tensor | None,
        value_aux_features: torch.Tensor | None = None,
    ) -> dict:
        """学習後モデルの教師再現メトリクスを計算する (CQ-0125)

        Args:
            observations: (N, obs_dim)
            legal_masks: (N, 34)
            actions: (N,) 教師 action
            teacher_best_masks: (N, 34) or None
            value_aux_features: (N, aux_dim) or None (CQ-0153)

        Returns:
            teacher_top1_match_rate と（mask 存在時）teacher_best_set_hit_rate
        """
        self._model.eval()
        n = len(observations)
        if n == 0:
            self._model.train()
            return {}

        all_preds = []
        for start in range(0, n, self._batch_size):
            end = min(start + self._batch_size, n)
            batch_vaux = value_aux_features[start:end] if value_aux_features is not None else None
            output = self._model(
                observations[start:end], legal_masks[start:end],
                value_aux_features=batch_vaux)
            all_preds.append(output.logits.argmax(dim=-1))
        pred_actions = torch.cat(all_preds)

        metrics: dict = {}
        top1 = (pred_actions == actions).float().mean().item()
        metrics["teacher_top1_match_rate"] = float(top1)

        if teacher_best_masks is not None:
            hits = teacher_best_masks.gather(
                1, pred_actions.unsqueeze(1)).squeeze(1)
            metrics["teacher_best_set_hit_rate"] = float(hits.mean().item())

        self._model.train()
        return metrics

    @staticmethod
    def _compute_ppo_diagnostics(
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        all_ratios: list[torch.Tensor],
        all_new_values: list[torch.Tensor],
        clip_epsilon: float | None = None,
    ) -> dict:
        """PPO 診断統計を計算する (CQ-0135)

        既に計算済みのテンソルから統計量のみを抽出する。
        生テンソルは保存しない。
        """
        adv_np = advantages.cpu().numpy().astype(np.float64)
        ret_np = returns.cpu().numpy().astype(np.float64)
        old_val_np = old_values.cpu().numpy().astype(np.float64)
        ratio_np = np.concatenate([r.numpy().astype(np.float64) for r in all_ratios])
        new_val_np = np.concatenate([v.numpy().astype(np.float64) for v in all_new_values])

        def _stats(arr: np.ndarray, prefix: str) -> dict:
            return {
                f"{prefix}_mean": float(np.mean(arr)),
                f"{prefix}_std": float(np.std(arr, ddof=0)),
                f"{prefix}_p50": float(np.percentile(arr, 50)),
                f"{prefix}_p90": float(np.percentile(arr, 90)),
                f"{prefix}_p99": float(np.percentile(arr, 99)),
            }

        diag: dict = {}

        # advantage 統計
        diag.update(_stats(adv_np, "advantage"))
        n_pos = int(np.sum(adv_np > 0))
        n_neg = int(np.sum(adv_np < 0))
        n_total = len(adv_np)
        diag["advantage_positive_ratio"] = n_pos / n_total if n_total > 0 else 0.0
        diag["advantage_negative_ratio"] = n_neg / n_total if n_total > 0 else 0.0

        # return 統計
        diag.update(_stats(ret_np, "return"))

        # old_value 統計
        diag["old_value_mean"] = float(np.mean(old_val_np))
        diag["old_value_std"] = float(np.std(old_val_np, ddof=0))

        # new_value 統計（全 epoch/batch の集約）
        diag["new_value_mean"] = float(np.mean(new_val_np))
        diag["new_value_std"] = float(np.std(new_val_np, ddof=0))

        # value_error = old_value - return（GAE 前提の予測誤差）
        val_err = old_val_np - ret_np
        diag.update(_stats(val_err, "value_error"))

        # ratio 統計
        diag.update(_stats(ratio_np, "ratio"))

        # clip_fraction: |ratio - 1| > clip_epsilon の割合
        if clip_epsilon is None:
            clip_epsilon = 0.2
        clipped = np.abs(ratio_np - 1.0) > clip_epsilon
        diag["clip_fraction"] = float(np.mean(clipped))

        return diag

    @staticmethod
    def _compute_shanten_conditioned_diag(
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        shanten_deltas: np.ndarray,
        new_values: torch.Tensor | None = None,
        rewards: torch.Tensor | None = None,
        terminateds: torch.Tensor | None = None,
        gamma: float = 0.99,
        point_delta_rewards: np.ndarray | None = None,
        shanten_delta_rewards: np.ndarray | None = None,
        is_post_riichi_discards: np.ndarray | None = None,
    ) -> dict:
        """shanten 変化別の advantage/return/value_error 診断統計 (CQ-0145, CQ-0146, CQ-0160, CQ-0163)

        3群 (improve/same/worsen) に分割して統計量を計算する。
        NaN の shanten_delta は unavailable として除外し、same 群に混入させない。
        CQ-0160: reward/point_delta_reward/shanten_delta_reward/delta_t を追加。
        CQ-0163: is_post_riichi_discards による立直後打牌統計を追加。
        """
        adv_np = advantages.cpu().numpy().astype(np.float64)
        ret_np = returns.cpu().numpy().astype(np.float64)
        old_val_np = old_values.cpu().numpy().astype(np.float64)
        val_err_np = old_val_np - ret_np
        new_val_np = new_values.cpu().numpy().astype(np.float64) if new_values is not None else None
        sd = shanten_deltas.astype(np.float64)

        # CQ-0160: reward components
        reward_np = rewards.cpu().numpy().astype(np.float64) if rewards is not None else None
        pdr_np = point_delta_rewards.astype(np.float64) if point_delta_rewards is not None else None
        sdr_np = shanten_delta_rewards.astype(np.float64) if shanten_delta_rewards is not None else None

        # CQ-0160: delta_t = reward_t + gamma * next_value_t - old_value_t
        delta_t_np = None
        if reward_np is not None and terminateds is not None:
            t_cpu = terminateds.cpu().numpy()
            n = len(reward_np)
            delta_t_np = np.zeros(n, dtype=np.float64)
            for t in range(n):
                if t == n - 1 or t_cpu[t]:
                    next_val = 0.0
                else:
                    next_val = old_val_np[t + 1]
                delta_t_np[t] = reward_np[t] + gamma * next_val - old_val_np[t]

        # CQ-0146: NaN は unavailable として除外
        available_mask = ~np.isnan(sd)
        n_total = len(sd)
        n_available = int(np.sum(available_mask))
        n_unavailable = n_total - n_available

        # 状態判定
        if n_available == n_total:
            status = "complete"
        elif n_available == 0:
            status = "unavailable"
        else:
            status = "partial"

        groups = {
            "improve": available_mask & (sd > 0),
            "same": available_mask & (sd == 0),
            "worsen": available_mask & (sd < 0),
        }

        def _group_stats(arr: np.ndarray) -> dict:
            """基本統計量を計算"""
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
            }

        # CQ-0163: 立直後打牌フラグ
        pr_np = is_post_riichi_discards if is_post_riichi_discards is not None else None

        result: dict = {
            "status": status,
            "total_samples": n_total,
            "available_samples": n_available,
            "unavailable_samples": n_unavailable,
        }

        # CQ-0163: top-level post_riichi stats
        if pr_np is not None:
            total_pr = int(np.sum(pr_np))
            result["total_post_riichi_discards"] = total_pr
            avail_pr = int(np.sum(pr_np[available_mask])) if n_available > 0 else 0
            result["available_post_riichi_discards"] = avail_pr
        else:
            result["total_post_riichi_discards"] = None
            result["available_post_riichi_discards"] = None

        for name, mask in groups.items():
            count = int(np.sum(mask))
            if count == 0:
                result[name] = {"count": 0}
                continue
            entry: dict = {"count": count}
            # CQ-0163: 立直後打牌統計（群ごと）
            if pr_np is not None:
                pr_count = int(np.sum(pr_np[mask]))
                entry["post_riichi_discard_count"] = pr_count
                entry["post_riichi_discard_ratio"] = pr_count / count
            # advantage
            adv_g = adv_np[mask]
            adv_stats = _group_stats(adv_g)
            n_pos = int(np.sum(adv_g > 0))
            n_neg = int(np.sum(adv_g < 0))
            adv_stats["positive_ratio"] = n_pos / count
            adv_stats["negative_ratio"] = n_neg / count
            entry["advantage"] = adv_stats
            # return
            entry["return"] = _group_stats(ret_np[mask])
            # CQ-0155: old_value
            entry["old_value"] = _group_stats(old_val_np[mask])
            # value_error
            entry["value_error"] = _group_stats(val_err_np[mask])
            # CQ-0155: new_value / value_update_delta
            if new_val_np is not None:
                entry["new_value"] = _group_stats(new_val_np[mask])
                delta = new_val_np[mask] - old_val_np[mask]
                entry["value_update_delta"] = _group_stats(delta)
            # CQ-0160: reward / point_delta_reward / shanten_delta_reward / delta_t
            if reward_np is not None:
                entry["reward"] = _group_stats(reward_np[mask])
            if pdr_np is not None:
                entry["point_delta_reward"] = _group_stats(pdr_np[mask])
            if sdr_np is not None:
                entry["shanten_delta_reward"] = _group_stats(sdr_np[mask])
            if delta_t_np is not None:
                entry["delta_t"] = _group_stats(delta_t_np[mask])
            result[name] = entry
        return result

    @staticmethod
    def _compute_turn_conditioned_diag(
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        turn_numbers: np.ndarray,
        new_values: torch.Tensor | None = None,
    ) -> dict:
        """巡目バケット別の診断統計 (CQ-0156)

        バケット: early (0-5), mid (6-11), late (12+)
        """
        adv_np = advantages.cpu().numpy().astype(np.float64)
        ret_np = returns.cpu().numpy().astype(np.float64)
        old_val_np = old_values.cpu().numpy().astype(np.float64)
        new_val_np = new_values.cpu().numpy().astype(np.float64) if new_values is not None else None
        tn = turn_numbers.astype(np.int32)

        buckets = {
            "early": (0, 5),
            "mid": (6, 11),
            "late": (12, None),
        }

        def _group_stats(arr: np.ndarray) -> dict:
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
            }

        result: dict = {"total_samples": len(tn)}
        for name, (lo, hi) in buckets.items():
            if hi is not None:
                mask = (tn >= lo) & (tn <= hi)
            else:
                mask = tn >= lo
            count = int(np.sum(mask))
            if count == 0:
                result[name] = {"count": 0}
                continue
            entry: dict = {"count": count}
            # advantage
            adv_g = adv_np[mask]
            adv_stats = _group_stats(adv_g)
            n_pos = int(np.sum(adv_g > 0))
            n_neg = int(np.sum(adv_g < 0))
            adv_stats["positive_ratio"] = n_pos / count
            adv_stats["negative_ratio"] = n_neg / count
            entry["advantage"] = adv_stats
            # return
            entry["return"] = _group_stats(ret_np[mask])
            # old_value
            entry["old_value"] = _group_stats(old_val_np[mask])
            # value_error
            val_err = old_val_np[mask] - ret_np[mask]
            entry["value_error"] = _group_stats(val_err)
            # new_value / value_update_delta
            if new_val_np is not None:
                entry["new_value"] = _group_stats(new_val_np[mask])
                delta = new_val_np[mask] - old_val_np[mask]
                entry["value_update_delta"] = _group_stats(delta)
            result[name] = entry
        return result

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
