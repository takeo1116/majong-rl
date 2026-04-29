"""CQ-0226: Stage2a learner — discard + call の imitation / PPO 学習

Stage1 の Learner とは独立。DecisionSample shard を読み、
discard / call それぞれの loss を計算して更新する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mahjong_rl.call_shard import DecisionShardReader, DecisionSample
from mahjong_rl.models.stage2a_model import Stage2aModel
from mahjong_rl.candidate_encoding import (
    encode_candidates_batch as _encode_candidates_batch_impl,
    ACTION_TYPE_MAP as _ACTION_TYPE_MAP,
    MAX_CONSUMED as _MAX_CONSUMED,
)


def _encode_candidates_batch(
    samples: list[DecisionSample],
    max_cands: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """call sample 群の candidate encoding (shared helper 委譲)"""
    return _encode_candidates_batch_impl(samples, max_cands)


class Stage2aLearner:
    """Stage2a learner (imitation / PPO)"""

    def __init__(
        self,
        config: dict,
        model: Stage2aModel,
        run_dir: Path,
        device: torch.device | None = None,
    ):
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)

        tc = config.get("training", {})
        # CQ-0255: CUDA memory debug
        cmd = tc.get("cuda_memory_debug", {})
        self._cuda_mem_debug = cmd.get("enabled", False) and self._device.type == "cuda"
        self._mode = tc.get("algorithm", "imitation")
        self._lr = tc.get("lr", 3e-4)
        self._batch_size = tc.get("batch_size", 256)
        self._epochs = tc.get("epochs", 4)
        self._value_loss_coef = tc.get("value_loss_coef", 0.5)
        self._entropy_coef = tc.get("entropy_coef", 0.01)
        self._clip_epsilon = tc.get("clip_epsilon", 0.2)
        self._max_grad_norm = tc.get("max_grad_norm", 0.5)
        # CQ-0237: GAE / value loss / advantage stabilization
        self._gamma = tc.get("gamma", 0.99)
        self._gae_lambda = tc.get("gae_lambda", 0.95)
        vl_cfg = tc.get("value_loss", {})
        self._value_loss_type = vl_cfg.get("type", "mse")
        self._huber_delta = vl_cfg.get("huber_delta", 1.0)
        adv_stab = tc.get("advantage_stabilization", {})
        self._advantage_clip = adv_stab.get("clip", None)
        # CQ-0239: imitation loss mode
        self._imitation_loss_mode = tc.get("imitation_loss_mode", "strict_top1")
        # CQ-0241: value warmstart
        ivw = tc.get("imitation_value_warmstart", {})
        self._imi_value_enabled = ivw.get("enabled", False)
        self._imi_value_coef = ivw.get("coef", 0.5)
        # CQ-0240: policy anchor
        pa = tc.get("policy_anchor", {})
        self._anchor_enabled = pa.get("enabled", False)
        self._anchor_coef = pa.get("coef", 0.0)
        self._anchor_reference = pa.get("reference", "imitation_fixed")
        self._anchor_model: Stage2aModel | None = None
        # CQ-0249: mixed PPO weighting
        rml = tc.get("rule_mix_learner", {})
        self._baseline_sample_weight = rml.get("baseline_sample_weight", 1.0)
        self._mixed_ppo_mode = rml.get("ppo_mode", "separated")
        # CQ-0256: semantic aux
        sa = tc.get("semantic_aux", {})
        self._semantic_aux_enabled = sa.get("enabled", False)
        self._terminal_loss_coef = sa.get("terminal_loss_coef", 0.2)
        self._yaku_loss_coef = sa.get("yaku_loss_coef", 0.1)

        self._optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)

    @staticmethod
    def _compute_terminal_weights(episode_ids, round_ids, player_ids, n: int,
                                   device: torch.device) -> torch.Tensor:
        """CQ-0268: player-round 正規化重みを計算する

        同じ (episode_id, round_id, player_id) に属する row の weight 合計が 1.0 になる。
        """
        from collections import Counter
        keys = []
        for i in range(n):
            eid = episode_ids[i] if isinstance(episode_ids[i], str) else str(episode_ids[i])
            rid = int(round_ids[i])
            pid = int(player_ids[i])
            keys.append((eid, rid, pid))
        counts = Counter(keys)
        weights = torch.zeros(n, dtype=torch.float32, device=device)
        for i in range(n):
            weights[i] = 1.0 / counts[keys[i]]
        return weights

    def _log_cuda_memory(self, label: str) -> dict | None:
        """CQ-0255: CUDA memory debug ログ (enabled 時のみ)"""
        if not self._cuda_mem_debug:
            return None
        import logging
        _logger = logging.getLogger(__name__)
        alloc = torch.cuda.memory_allocated(self._device)
        reserved = torch.cuda.memory_reserved(self._device)
        max_alloc = torch.cuda.max_memory_allocated(self._device)
        max_reserved = torch.cuda.max_memory_reserved(self._device)
        snap = {
            "allocated_mb": round(alloc / 1048576, 1),
            "reserved_mb": round(reserved / 1048576, 1),
            "max_allocated_mb": round(max_alloc / 1048576, 1),
            "max_reserved_mb": round(max_reserved / 1048576, 1),
        }
        _logger.info(f"[CUDA mem] {label}: {snap}")
        return snap

    def train(self, shard_dir: Path, num_epochs: int | None = None,
              filter_actor_type: str | None = None) -> dict:
        """CQ-0251: tensor ベースで shard を読み込んで学習"""
        reader = DecisionShardReader(shard_dir)
        data = reader.read_as_tensors(filter_actor_type=filter_actor_type)

        d = data["discard"]
        c = data["call"]
        nd = d["n"] if d else 0
        nc = c["n"] if c else 0

        # CQ-0250: unsafe mixed PPO guard
        if (self._mode == "ppo" and self._mixed_ppo_mode == "mixed"
                and not filter_actor_type):
            all_at = (d["actor_types"] if d else []) + (c["actor_types"] if c else [])
            n_policy = sum(1 for a in all_at if a == "policy")
            if n_policy == 0:
                raise ValueError(
                    "Stage02a mixed PPO: policy sample が 0 件です。"
                    " training.rule_mix.policy_ratio <= 0.0 の pure-baseline mixed PPO は"
                    " PPO 発散の主因になるため unsupported です。"
                    " policy_ratio > 0 に設定するか、ppo_mode='separated' を使ってください。"
                    f" (rule_mix_learner.ppo_mode={self._mixed_ppo_mode!r})"
                )

        if nd + nc == 0:
            return {"mode": self._mode, "total_steps": 0,
                    "policy_loss": 0.0, "value_loss": 0.0, "num_updates": 0}

        epochs = num_epochs or self._epochs

        if self._mode == "imitation":
            return self._train_imitation_tensor(d, c, epochs)
        else:
            # PPO は sample list path を安全に使う (candidate semantics 維持)
            all_samples = reader.read_all()
            if filter_actor_type:
                all_samples = [s for s in all_samples
                               if s.actor_type == filter_actor_type]
            d_samples = [s for s in all_samples if s.decision_type == "discard"]
            c_samples = [s for s in all_samples if s.decision_type == "call"]
            return self._train_ppo(d_samples, c_samples, epochs)

    def _compute_grouped_returns_numpy(self, data_d, data_c):
        """tensor data から grouped return を計算 (imitation value warmstart 用)"""
        from collections import defaultdict
        # 全 sample を step_id 順に並べて grouped return 計算
        items = []  # (step_id, branch, branch_idx, sample_data)
        if data_d:
            for i in range(data_d["n"]):
                items.append((int(data_d["step_ids"][i]), "d", i, {
                    "reward": float(data_d["rewards"][i]),
                    "value": float(data_d["values"][i]),
                    "terminated": float(data_d["terminateds"][i]),
                    "episode_id": data_d["episode_ids"][i],
                    "player_id": int(data_d["player_ids"][i]),
                }))
        if data_c:
            for i in range(data_c["n"]):
                items.append((int(data_c["step_ids"][i]), "c", i, {
                    "reward": float(data_c["rewards"][i]),
                    "value": float(data_c["values"][i]),
                    "terminated": float(data_c["terminateds"][i]),
                    "episode_id": data_c["episode_ids"][i],
                    "player_id": int(data_c["player_ids"][i]),
                }))
        items.sort(key=lambda x: x[0])

        # grouped GAE → return
        groups: dict[tuple, list[int]] = defaultdict(list)
        for idx, (_, _, _, info) in enumerate(items):
            groups[(info["episode_id"], info["player_id"])].append(idx)

        n = len(items)
        returns = np.zeros(n, dtype=np.float64)
        for key, indices in groups.items():
            g = len(indices)
            for t in reversed(range(g)):
                ii = indices[t]
                r = items[ii][3]["reward"]
                v = items[ii][3]["value"]
                term = items[ii][3]["terminated"]
                if t == g - 1 or term:
                    nv = 0.0
                    last_gae = 0.0
                else:
                    nv = items[indices[t + 1]][3]["value"]
                delta = r + self._gamma * nv - v
                last_gae = delta + self._gamma * self._gae_lambda * last_gae
                returns[ii] = last_gae + v

        # split back
        d_returns = np.zeros(data_d["n"] if data_d else 0, dtype=np.float32)
        c_returns = np.zeros(data_c["n"] if data_c else 0, dtype=np.float32)
        for idx, (_, branch, bi, _) in enumerate(items):
            if branch == "d":
                d_returns[bi] = returns[idx]
            else:
                c_returns[bi] = returns[idx]
        return d_returns, c_returns

    def _train_imitation_tensor(self, d, c, epochs):
        """CQ-0251/0252/0253: tensor ベース imitation"""
        import time as _time
        train_start = _time.perf_counter()
        self._model.train()
        all_losses, d_losses, c_losses, vl_list = [], [], [], []
        num_updates = 0
        nd = d["n"] if d else 0
        nc = c["n"] if c else 0

        # grouped returns for value warmstart
        d_ret_np, c_ret_np = None, None
        if self._imi_value_enabled:
            d_ret_np, c_ret_np = self._compute_grouped_returns_numpy(d, c)

        # pre-move to device
        d_obs = d_masks = d_actions = d_teacher_top1 = d_tbm = d_returns_t = None
        d_term_cls = d_yaku_mh = d_is_winner = d_term_weights = None
        if d:
            d_obs = torch.tensor(d["observations"], dtype=torch.float32, device=self._device)
            d_masks = torch.tensor(d["legal_masks"], dtype=torch.float32, device=self._device)
            d_actions = torch.tensor(d["actions"], dtype=torch.long, device=self._device)
            d_teacher_top1 = torch.tensor(d["teacher_top1"], dtype=torch.long, device=self._device)
            if d["teacher_best_mask"] is not None:
                d_tbm = torch.tensor(d["teacher_best_mask"], dtype=torch.float32, device=self._device)
            if d_ret_np is not None:
                d_returns_t = torch.tensor(d_ret_np, dtype=torch.float32, device=self._device)
            # CQ-0256: semantic targets
            d_term_cls = torch.tensor(d["terminal_classes"], dtype=torch.long, device=self._device) if "terminal_classes" in d else None
            d_yaku_mh = torch.tensor(d["yaku_multihot"], dtype=torch.float32, device=self._device) if "yaku_multihot" in d else None
            d_is_winner = torch.tensor(d["is_winner"], dtype=torch.float32, device=self._device) if "is_winner" in d else None
            # CQ-0268: terminal player-round weights
            d_term_weights = self._compute_terminal_weights(
                d["episode_ids"], d.get("round_ids", np.zeros(nd, dtype=np.int64)),
                d["player_ids"], nd, self._device) if self._semantic_aux_enabled else None

        c_obs = c_cf = c_cm = c_rc = c_targets = c_returns_t = None
        c_rewards = c_term_cls = c_yaku_mh = c_is_winner = c_term_weights = None
        if c:
            c_obs = torch.tensor(c["observations"], dtype=torch.float32, device=self._device)
            c_cf = torch.tensor(c["cand_feats"], dtype=torch.long, device=self._device)
            c_cm = torch.tensor(c["cand_mask"], dtype=torch.float32, device=self._device)
            c_rc = torch.tensor(c["response_context"], dtype=torch.float32, device=self._device)
            c_targets = torch.tensor(c["teacher_top1"], dtype=torch.long, device=self._device)
            sel = torch.tensor(c["selected_idx"], dtype=torch.long, device=self._device)
            neg = c_targets < 0
            c_targets[neg] = sel[neg]
            if c_ret_np is not None:
                c_returns_t = torch.tensor(c_ret_np, dtype=torch.float32, device=self._device)
            c_rewards = torch.tensor(c["rewards"], dtype=torch.float32, device=self._device)
            c_term_cls = torch.tensor(c["terminal_classes"], dtype=torch.long, device=self._device) if "terminal_classes" in c else None
            c_yaku_mh = torch.tensor(c["yaku_multihot"], dtype=torch.float32, device=self._device) if "yaku_multihot" in c else None
            c_is_winner = torch.tensor(c["is_winner"], dtype=torch.float32, device=self._device) if "is_winner" in c else None
            # CQ-0268: terminal player-round weights
            c_term_weights = self._compute_terminal_weights(
                c["episode_ids"], c.get("round_ids", np.zeros(nc, dtype=np.int64)),
                c["player_ids"], nc, self._device) if self._semantic_aux_enabled else None

        self._log_cuda_memory("imitation_preload")
        sem_tl_list, sem_yl_list = [], []

        for _ in range(epochs):
            # discard imitation (CQ-0252: vectorized best-set)
            if d and nd > 0:
                indices = torch.randperm(nd, device=self._device)
                for start in range(0, nd, self._batch_size):
                    end = min(start + self._batch_size, nd)
                    idx = indices[start:end]
                    out = self._model.forward_discard(d_obs[idx], d_masks[idx])
                    log_p = F.log_softmax(out.discard_logits, dim=-1)

                    if self._imitation_loss_mode == "tie_aware_best_set" and d_tbm is not None:
                        # CQ-0252: vectorized best-set loss + row-wise fallback
                        batch_tbm = d_tbm[idx]
                        has_best = batch_tbm.sum(dim=-1) > 0  # (B,)
                        probs = log_p.exp()
                        # best-set rows
                        best_set_prob = (probs * batch_tbm).sum(dim=-1)
                        best_loss = -torch.log(best_set_prob.clamp(min=1e-10))
                        # fallback rows: teacher_top1 or action
                        fallback_targets = d_teacher_top1[idx].clone()
                        neg = fallback_targets < 0
                        fallback_targets[neg] = d_actions[idx][neg]
                        fallback_loss = F.nll_loss(log_p, fallback_targets, reduction="none")
                        # combine
                        per_sample = torch.where(has_best, best_loss, fallback_loss)
                        ploss = per_sample.mean()
                    else:
                        targets = d_teacher_top1[idx].clone()
                        neg = targets < 0
                        targets[neg] = d_actions[idx][neg]
                        ploss = F.nll_loss(log_p, targets)

                    total = ploss
                    if self._imi_value_enabled and d_returns_t is not None:
                        v = out.values["round_delta"].squeeze(-1)
                        vl = self._compute_value_loss(v, d_returns_t[idx])
                        total = total + self._imi_value_coef * vl
                        vl_list.append(vl.item())
                    # CQ-0256/0268: semantic aux loss with terminal weights
                    if self._semantic_aux_enabled and d_term_cls is not None:
                        tw = d_term_weights[idx] if d_term_weights is not None else None
                        sa_loss, tl_v, yl_v = self._compute_semantic_aux_loss(
                            out.semantic, d_term_cls[idx], d_yaku_mh[idx], d_is_winner[idx],
                            terminal_weights=tw)
                        total = total + sa_loss
                        sem_tl_list.append(tl_v)
                        sem_yl_list.append(yl_v)

                    self._optimizer.zero_grad()
                    total.backward()
                    nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
                    self._optimizer.step()
                    d_losses.append(ploss.item())
                    all_losses.append(ploss.item())
                    num_updates += 1

            # optional imitation (CQ-0253: precomputed tensors)
            if c and nc > 0:
                indices = torch.randperm(nc, device=self._device)
                for start in range(0, nc, self._batch_size):
                    end = min(start + self._batch_size, nc)
                    idx = indices[start:end]
                    out = self._model.forward_optional(
                        c_obs[idx], c_cf[idx], c_cm[idx],
                        response_context=c_rc[idx])
                    log_p = F.log_softmax(out.optional_scores, dim=-1)
                    ploss = F.nll_loss(log_p, c_targets[idx])

                    total = ploss
                    if self._imi_value_enabled and c_returns_t is not None:
                        v = out.values["round_delta"].squeeze(-1)
                        vl = self._compute_value_loss(v, c_returns_t[idx])
                        total = total + self._imi_value_coef * vl
                        vl_list.append(vl.item())
                    # CQ-0256/0268: semantic aux loss with terminal weights
                    if self._semantic_aux_enabled and c_term_cls is not None:
                        tw = c_term_weights[idx] if c_term_weights is not None else None
                        sa_loss, tl_v, yl_v = self._compute_semantic_aux_loss(
                            out.semantic, c_term_cls[idx], c_yaku_mh[idx], c_is_winner[idx],
                            terminal_weights=tw)
                        total = total + sa_loss
                        sem_tl_list.append(tl_v)
                        sem_yl_list.append(yl_v)

                    self._optimizer.zero_grad()
                    total.backward()
                    nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
                    self._optimizer.step()
                    c_losses.append(ploss.item())
                    all_losses.append(ploss.item())
                    num_updates += 1

        train_end = _time.perf_counter()
        self._log_cuda_memory("imitation_train_end")

        # CQ-0255: free training-only GPU tensors before diagnostics
        # Keep d_obs, d_masks, c_obs, c_cf, c_cm, c_rc for diagnostics reuse
        d_actions = d_teacher_top1 = d_returns_t = d_tbm = None
        d_term_cls = d_yaku_mh = d_is_winner = None
        c_targets = c_returns_t = c_rewards = None
        c_term_cls = c_yaku_mh = c_is_winner = None

        self._log_cuda_memory("imitation_pre_diag")
        diag_t0 = _time.perf_counter()
        diag = self._compute_imitation_diagnostics_preloaded(
            d, d_obs, d_masks, c, c_obs, c_cf, c_cm, c_rc)
        diag_sec = _time.perf_counter() - diag_t0

        # CQ-0255: free remaining GPU tensors
        del d_obs, d_masks, c_obs, c_cf, c_cm, c_rc
        self._log_cuda_memory("imitation_post_diag")

        return {
            "mode": "imitation",
            "imitation_loss_mode": self._imitation_loss_mode,
            "total_steps": nd + nc,
            "discard_count": nd,
            "call_count": nc,
            "policy_loss": float(np.mean(all_losses)) if all_losses else 0.0,
            "discard_loss": float(np.mean(d_losses)) if d_losses else 0.0,
            "call_loss": float(np.mean(c_losses)) if c_losses else 0.0,
            "value_loss": float(np.mean(vl_list)) if vl_list else 0.0,
            "num_updates": num_updates,
            "imitation_value_warmstart": self._imi_value_enabled,
            "semantic_aux_enabled": self._semantic_aux_enabled,
            "terminal_loss": float(np.mean(sem_tl_list)) if sem_tl_list else None,
            "yaku_loss": float(np.mean(sem_yl_list)) if sem_yl_list else None,
            "timing": {
                "train_sec": round(train_end - train_start, 3),
                "diagnostics_sec": round(diag_sec, 3),
            },
            **diag,
        }

    def _compute_imitation_diagnostics_preloaded(
        self, d, d_obs, d_masks, c, c_obs, c_cf, c_cm, c_rc,
    ):
        """CQ-0255: diagnostics using preloaded GPU tensors (no re-allocation)"""
        diag: dict = {}
        self._model.eval()
        with torch.inference_mode():
            if d and d["n"] > 0 and d_obs is not None:
                out = self._model.forward_discard(d_obs, d_masks, compute_value=False)
                pred = out.discard_logits.argmax(dim=-1).cpu().numpy()
                t1 = d["teacher_top1"]
                valid = t1 >= 0
                if valid.sum() > 0:
                    diag["teacher_top1_match_rate_discard"] = float(
                        (pred[valid] == t1[valid]).mean())
                if d["teacher_best_mask"] is not None:
                    tbm = d["teacher_best_mask"]
                    has_best = tbm.sum(axis=1) > 0
                    if has_best.sum() > 0:
                        hits = tbm[has_best, pred[has_best]]
                        diag["teacher_best_set_hit_rate_discard"] = float(
                            (hits > 0).mean())

            if c and c["n"] > 0 and c_obs is not None:
                bs = min(self._batch_size, c["n"])
                preds = []
                for start in range(0, c["n"], bs):
                    end = min(start + bs, c["n"])
                    out = self._model.forward_optional(
                        c_obs[start:end], c_cf[start:end], c_cm[start:end],
                        response_context=c_rc[start:end], compute_value=False)
                    preds.append(out.optional_scores.argmax(dim=-1).cpu().numpy())
                pred = np.concatenate(preds)
                t1 = c["teacher_top1"]
                valid = t1 >= 0
                if valid.sum() > 0:
                    diag["teacher_top1_match_rate_optional"] = float(
                        (pred[valid] == t1[valid]).mean())

        self._model.train()
        return diag

    def _train_imitation(
        self,
        discard_samples: list[DecisionSample],
        call_samples: list[DecisionSample],
        epochs: int,
    ) -> dict:
        """imitation 学習 (teacher-aware CE + optional value warmstart)"""
        import time as _time
        train_start = _time.perf_counter()
        self._model.train()
        all_losses = []
        discard_losses = []
        call_losses = []
        all_value_losses: list[float] = []
        num_updates = 0

        # CQ-0241: grouped returns for value warmstart target
        d_returns = None
        c_returns = None
        if self._imi_value_enabled:
            all_imi = discard_samples + call_samples
            # step_id 順に並べて grouped return 計算
            indexed = [(s.step_id, i, s) for i, s in enumerate(all_imi)]
            indexed.sort(key=lambda x: x[0])
            sorted_samples = [s for _, _, s in indexed]
            all_ret, _ = self._compute_returns_advantages(sorted_samples)
            # 元の順序に戻す
            ret_by_orig = {}
            for j, (_, orig_i, _) in enumerate(indexed):
                ret_by_orig[orig_i] = all_ret[j].item()
            nd = len(discard_samples)
            d_returns = [ret_by_orig[i] for i in range(nd)] if nd > 0 else None
            c_returns = [ret_by_orig[nd + i] for i in range(len(call_samples))] if call_samples else None

        for _ in range(epochs):
            if discard_samples:
                dl, dvl = self._imitation_discard_epoch(discard_samples, d_returns)
                discard_losses.extend(dl)
                all_losses.extend(dl)
                all_value_losses.extend(dvl)
                num_updates += len(dl)

            if call_samples:
                cl, cvl = self._imitation_call_epoch(call_samples, c_returns)
                call_losses.extend(cl)
                all_losses.extend(cl)
                all_value_losses.extend(cvl)
                num_updates += len(cl)

        import time as _time
        train_end = _time.perf_counter()

        # CQ-0239: teacher diagnostics
        diag_t0 = _time.perf_counter()
        diag = self._compute_imitation_diagnostics(discard_samples, call_samples)
        diag_sec = _time.perf_counter() - diag_t0

        return {
            "mode": "imitation",
            "imitation_loss_mode": self._imitation_loss_mode,
            "total_steps": len(discard_samples) + len(call_samples),
            "discard_count": len(discard_samples),
            "call_count": len(call_samples),
            "policy_loss": float(np.mean(all_losses)) if all_losses else 0.0,
            "discard_loss": float(np.mean(discard_losses)) if discard_losses else 0.0,
            "call_loss": float(np.mean(call_losses)) if call_losses else 0.0,
            "value_loss": float(np.mean(all_value_losses)) if all_value_losses else 0.0,
            "num_updates": num_updates,
            "imitation_value_warmstart": self._imi_value_enabled,
            "timing": {
                "train_sec": round(train_end - train_start, 3),
                "diagnostics_sec": round(diag_sec, 3),
            },
            **diag,
        }

    def _imitation_discard_epoch(self, samples: list[DecisionSample],
                                 returns: list[float] | None = None,
                                 ) -> tuple[list[float], list[float]]:
        """discard imitation の 1 epoch"""
        losses = []
        value_losses: list[float] = []
        indices = np.random.permutation(len(samples))
        for start in range(0, len(samples), self._batch_size):
            end = min(start + self._batch_size, len(samples))
            batch_idx = indices[start:end]
            batch = [samples[i] for i in batch_idx]

            obs = torch.tensor(
                np.stack([s.observation for s in batch]),
                dtype=torch.float32, device=self._device)
            masks = torch.tensor(
                np.stack([s.legal_mask for s in batch]),
                dtype=torch.float32, device=self._device)
            actions = torch.tensor(
                [s.action for s in batch],
                dtype=torch.long, device=self._device)

            out = self._model.forward_discard(obs, masks)
            log_probs = F.log_softmax(out.discard_logits, dim=-1)

            # CQ-0239: teacher-aware imitation
            if self._imitation_loss_mode == "tie_aware_best_set":
                # best-set loss: maximize sum of log_probs for best indices
                policy_loss = torch.tensor(0.0, device=self._device)
                for bi, s in enumerate(batch):
                    if s.teacher_best_indices:
                        best_lp = log_probs[bi, s.teacher_best_indices]
                        policy_loss -= torch.logsumexp(best_lp, dim=0)
                    else:
                        policy_loss -= log_probs[bi, actions[bi]]
                policy_loss = policy_loss / len(batch)
            else:
                # strict_top1: teacher_top1_index を使う
                teacher_targets = torch.tensor(
                    [s.teacher_top1_index if s.teacher_top1_index >= 0 else s.action
                     for s in batch],
                    dtype=torch.long, device=self._device)
                policy_loss = F.nll_loss(log_probs, teacher_targets)

            # CQ-0241: value warmstart (grouped return target)
            total_loss = policy_loss
            vl_val = 0.0
            if self._imi_value_enabled and returns is not None:
                value = out.values["round_delta"].squeeze(-1)
                target_vals = [returns[batch_idx[j]] for j in range(len(batch))]
                targets_t = torch.tensor(target_vals,
                                          dtype=torch.float32, device=self._device)
                vl = self._compute_value_loss(value, targets_t)
                total_loss = total_loss + self._imi_value_coef * vl
                vl_val = vl.item()

            self._optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()
            losses.append(policy_loss.item())
            if self._imi_value_enabled:
                value_losses.append(vl_val)

        return losses, value_losses

    def _imitation_call_epoch(self, samples: list[DecisionSample],
                              returns: list[float] | None = None,
                              ) -> tuple[list[float], list[float]]:
        """optional imitation の 1 epoch"""
        losses = []
        value_losses: list[float] = []
        max_cands = max(s.candidate_count for s in samples)
        indices = np.random.permutation(len(samples))

        for start in range(0, len(samples), self._batch_size):
            end = min(start + self._batch_size, len(samples))
            batch_idx = indices[start:end]
            batch = [samples[i] for i in batch_idx]

            obs = torch.tensor(
                np.stack([s.observation for s in batch]),
                dtype=torch.float32, device=self._device)
            targets = torch.tensor(
                [s.selected_candidate_index for s in batch],
                dtype=torch.long, device=self._device)

            cand_feats, cand_mask = _encode_candidates_batch(batch, max_cands)
            cand_feats = cand_feats.to(self._device)
            cand_mask = cand_mask.to(self._device)

            # CQ-0242: response_context from shard
            rc_list = [s.response_context if s.response_context is not None
                       else np.zeros(3, dtype=np.float32)
                       for s in batch]
            rc_t = torch.tensor(np.stack(rc_list),
                                dtype=torch.float32, device=self._device)

            out = self._model.forward_optional(
                obs, cand_feats, cand_mask, response_context=rc_t)
            log_probs = F.log_softmax(out.optional_scores, dim=-1)

            # CQ-0239: teacher top1 for optional
            teacher_targets = torch.tensor(
                [s.teacher_top1_index
                 if s.teacher_top1_index >= 0
                 else s.selected_candidate_index
                 for s in batch],
                dtype=torch.long, device=self._device)
            policy_loss = F.nll_loss(log_probs, teacher_targets)

            # CQ-0241: value warmstart (grouped return target)
            total_loss = policy_loss
            vl_val = 0.0
            if self._imi_value_enabled and returns is not None:
                value = out.values["round_delta"].squeeze(-1)
                target_vals = [returns[batch_idx[j]] for j in range(len(batch))]
                targets_t = torch.tensor(target_vals,
                                          dtype=torch.float32, device=self._device)
                vl = self._compute_value_loss(value, targets_t)
                total_loss = total_loss + self._imi_value_coef * vl
                vl_val = vl.item()

            self._optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()
            losses.append(policy_loss.item())
            if self._imi_value_enabled:
                value_losses.append(vl_val)

        return losses, value_losses

    def _train_ppo_tensor(self, d, c, epochs):
        """PPO は当面 sample list path を使う (candidate semantics の安全を優先)"""
        # tensor 変換は imitation のみ。PPO は read_all() から sample list を直接使う
        # _shard_dir は train() が持っていないため、d/c の raw sample は使えない。
        # 代わりに d/c の numpy 配列から安全に sample を再構築する
        d_samples = self._rebuild_discard_samples(d)
        c_samples = self._rebuild_call_samples(c)
        return self._train_ppo(d_samples, c_samples, epochs)

    @staticmethod
    def _rebuild_discard_samples(d):
        if not d or d["n"] == 0:
            return []
        samples = []
        for i in range(d["n"]):
            samples.append(DecisionSample(
                decision_type="discard",
                observation=d["observations"][i],
                legal_mask=d["legal_masks"][i],
                action=int(d["actions"][i]),
                reward=float(d["rewards"][i]),
                log_prob=float(d["log_probs"][i]),
                value=float(d["values"][i]),
                terminated=bool(d["terminateds"][i]),
                round_over=False,
                step_id=int(d["step_ids"][i]),
                actor_type=d["actor_types"][i],
                episode_id=d["episode_ids"][i],
                player_id=int(d["player_ids"][i]),
                response_context=d["response_context"][i],
            ))
        return samples

    @staticmethod
    def _rebuild_call_samples(c):
        """call sample を raw shard 表現から安全に再構築 (candidate encoding を逆変換しない)"""
        if not c or c["n"] == 0:
            return []
        # NOTE: cand_feats は learner encoding 済み。CandidateRecord の raw 値とは異なる。
        # PPO epoch は _encode_candidates_batch を内部で呼ぶため、
        # ここでは encoding 前の raw 情報が必要。
        # read_as_tensors の call dict には raw candidate が入っていないため、
        # PPO path 用に空 candidates の sample を返す。
        # _train_ppo 内の _ppo_call_epoch は cand_feats を内部で再計算するが、
        # sample.candidates が空なので encode は全 padding になる。
        # → PPO call branch は当面 read_all() で正しく動かす必要がある。
        # この制限は CQ-0251 実装メモに記載済み。
        #
        # 安全のため、PPO tensor path の call branch は当面スキップし、
        # caller が read_all() 経由で call_samples を渡す形を推奨する。
        return []  # call PPO は旧 path に fallback

    def _train_ppo(
        self,
        discard_samples: list[DecisionSample],
        call_samples: list[DecisionSample],
        epochs: int,
    ) -> dict:
        """PPO 学習 (mixed-trajectory grouped GAE + entropy + diagnostics)"""
        self._model.train()
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_ratios = []
        all_anchor_kl_discard: list[float] = []
        all_anchor_kl_optional: list[float] = []
        num_updates = 0

        # CQ-0275: branch 元順を保つように scatter する
        is_mixed = self._mixed_ppo_mode == "mixed"
        ppo_targets = self._compute_ppo_branch_targets(
            discard_samples, call_samples, is_mixed=is_mixed)
        d_ret = ppo_targets["d_ret"]
        d_adv = ppo_targets["d_adv"]
        d_weights = ppo_targets["d_weights"]
        c_ret = ppo_targets["c_ret"]
        c_adv = ppo_targets["c_adv"]
        c_weights = ppo_targets["c_weights"]
        all_sorted = ppo_targets["all_sorted"]

        # CQ-0256: semantic aux targets for PPO
        d_term_cls = d_yaku_mh = d_is_winner = None
        c_term_cls = c_yaku_mh = c_is_winner = None
        if self._semantic_aux_enabled:
            from mahjong_rl.outcome_vocab import (
                terminal_label_to_class, yaku_ids_to_multihot, NUM_YAKU)
            if discard_samples:
                d_term_cls = torch.tensor(
                    [terminal_label_to_class(s.round_terminal_label)
                     for s in discard_samples],
                    dtype=torch.long, device=self._device)
                d_yaku_mh = torch.tensor(
                    [yaku_ids_to_multihot(s.eventual_win_yaku_ids)
                     for s in discard_samples],
                    dtype=torch.float32, device=self._device)
                d_is_winner = torch.tensor(
                    [1.0 if s.round_terminal_label in
                     ("win", "win_menzen", "win_called") else 0.0
                     for s in discard_samples],
                    dtype=torch.float32, device=self._device)
            if call_samples:
                c_term_cls = torch.tensor(
                    [terminal_label_to_class(s.round_terminal_label)
                     for s in call_samples],
                    dtype=torch.long, device=self._device)
                c_yaku_mh = torch.tensor(
                    [yaku_ids_to_multihot(s.eventual_win_yaku_ids)
                     for s in call_samples],
                    dtype=torch.float32, device=self._device)
                c_is_winner = torch.tensor(
                    [1.0 if s.round_terminal_label in
                     ("win", "win_menzen", "win_called") else 0.0
                     for s in call_samples],
                    dtype=torch.float32, device=self._device)

        # CQ-0268: terminal player-round weights for PPO
        d_term_weights = c_term_weights = None
        if self._semantic_aux_enabled:
            if discard_samples:
                d_term_weights = self._compute_terminal_weights(
                    [s.episode_id for s in discard_samples],
                    [s.round_id for s in discard_samples],
                    [s.player_id for s in discard_samples],
                    len(discard_samples), self._device)
            if call_samples:
                c_term_weights = self._compute_terminal_weights(
                    [s.episode_id for s in call_samples],
                    [s.round_id for s in call_samples],
                    [s.player_id for s in call_samples],
                    len(call_samples), self._device)

        all_sem_tl: list[float] = []
        all_sem_yl: list[float] = []

        for _ in range(epochs):
            if discard_samples:
                pl, vl, ent, rats, akl, stl, syl = self._ppo_discard_epoch(
                    discard_samples, d_ret, d_adv, d_weights,
                    terminal_classes_t=d_term_cls,
                    yaku_multihot_t=d_yaku_mh,
                    is_winner_t=d_is_winner,
                    terminal_weights_t=d_term_weights)
                all_policy_losses.extend(pl)
                all_value_losses.extend(vl)
                all_entropies.extend(ent)
                all_ratios.extend(rats)
                all_anchor_kl_discard.extend(akl)
                all_sem_tl.extend(stl)
                all_sem_yl.extend(syl)
                num_updates += len(pl)

            if call_samples:
                pl, vl, ent, rats, akl, stl, syl = self._ppo_call_epoch(
                    call_samples, c_ret, c_adv, c_weights,
                    terminal_classes_t=c_term_cls,
                    yaku_multihot_t=c_yaku_mh,
                    is_winner_t=c_is_winner,
                    terminal_weights_t=c_term_weights)
                all_policy_losses.extend(pl)
                all_value_losses.extend(vl)
                all_entropies.extend(ent)
                all_ratios.extend(rats)
                all_anchor_kl_optional.extend(akl)
                all_sem_tl.extend(stl)
                all_sem_yl.extend(syl)
                num_updates += len(pl)

        # diagnostics
        ppo_diag: dict = {}
        if all_ratios:
            ratios_cat = torch.cat(all_ratios)
            ratio_np = ratios_cat.numpy().astype(np.float64)
            ppo_diag["ratio_mean"] = float(np.mean(ratio_np))
            ppo_diag["ratio_std"] = float(np.std(ratio_np))
            ppo_diag["clip_fraction"] = float(
                np.mean(np.abs(ratio_np - 1.0) > self._clip_epsilon))
        # overall advantage (discard + optional)
        adv_parts = []
        if d_adv is not None:
            adv_parts.append(d_adv.cpu().numpy())
        if c_adv is not None:
            adv_parts.append(c_adv.cpu().numpy())
        if adv_parts:
            all_adv_np = np.concatenate(adv_parts).astype(np.float64)
            ppo_diag["advantage_mean"] = float(np.mean(all_adv_np))
            ppo_diag["advantage_std"] = float(np.std(all_adv_np))
        # CQ-0240: anchor KL diagnostics
        if all_anchor_kl_discard:
            ppo_diag["anchor_kl_discard"] = float(np.mean(all_anchor_kl_discard))
        if all_anchor_kl_optional:
            ppo_diag["anchor_kl_optional"] = float(np.mean(all_anchor_kl_optional))
        # CQ-0249: mixed PPO diagnostics
        if is_mixed:
            n_policy = sum(1 for s in all_sorted if s.actor_type == "policy")
            n_baseline = sum(1 for s in all_sorted if s.actor_type == "baseline")
            ppo_diag["mixed_ppo"] = {
                "mixed_ppo_enabled": True,
                "baseline_sample_weight": self._baseline_sample_weight,
                "num_policy_samples": n_policy,
                "num_baseline_samples": n_baseline,
                "effective_weight_sum_policy": float(n_policy),
                "effective_weight_sum_baseline": float(n_baseline * self._baseline_sample_weight),
            }

        return {
            "mode": "ppo",
            "total_steps": len(discard_samples) + len(call_samples),
            "discard_count": len(discard_samples),
            "call_count": len(call_samples),
            "policy_loss": float(np.mean(all_policy_losses)) if all_policy_losses else 0.0,
            "value_loss": float(np.mean(all_value_losses)) if all_value_losses else 0.0,
            "entropy": float(np.mean(all_entropies)) if all_entropies else 0.0,
            "num_updates": num_updates,
            "semantic_aux_enabled": self._semantic_aux_enabled,
            "terminal_loss": float(np.mean(all_sem_tl)) if all_sem_tl else None,
            "yaku_loss": float(np.mean(all_sem_yl)) if all_sem_yl else None,
            "ppo_diag": ppo_diag,
        }

    def load_anchor(self, checkpoint_path: str) -> None:
        """CQ-0240: anchor model をロードする"""
        import copy
        self._anchor_model = copy.deepcopy(self._model)
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self._anchor_model.load_state_dict(sd)
        self._anchor_model.to(self._device)
        self._anchor_model.eval()
        for p in self._anchor_model.parameters():
            p.requires_grad_(False)

    def update_lagged_anchor(self) -> None:
        """CQ-0240: lagged policy anchor を現在の model で更新"""
        if self._anchor_model is not None:
            self._anchor_model.load_state_dict(self._model.state_dict())

    def _compute_anchor_kl_discard(
        self, features, masks, out,
    ) -> tuple[torch.Tensor, float]:
        """discard branch の anchor KL"""
        if self._anchor_model is None:
            return torch.tensor(0.0, device=self._device), 0.0
        with torch.no_grad():
            ref_out = self._anchor_model.forward_discard(features, masks)
        # KL(current || ref) on legal mask
        cur_lp = F.log_softmax(out.discard_logits, dim=-1)
        ref_lp = F.log_softmax(ref_out.discard_logits, dim=-1)
        cur_p = cur_lp.exp()
        kl = (cur_p * (cur_lp - ref_lp)).sum(dim=-1).mean()
        return kl, float(kl.item())

    def _compute_anchor_kl_optional(
        self, features, cand_feats, cand_mask, out,
        response_context=None,
    ) -> tuple[torch.Tensor, float]:
        """optional branch の anchor KL"""
        if self._anchor_model is None:
            return torch.tensor(0.0, device=self._device), 0.0
        with torch.no_grad():
            ref_out = self._anchor_model.forward_optional(
                features, cand_feats, cand_mask,
                response_context=response_context)
        cur_lp = F.log_softmax(out.optional_scores, dim=-1)
        ref_lp = F.log_softmax(ref_out.optional_scores, dim=-1)
        cur_p = cur_lp.exp()
        kl = (cur_p * (cur_lp - ref_lp)).sum(dim=-1).mean()
        return kl, float(kl.item())

    def _compute_imitation_diagnostics(
        self, discard_samples, call_samples,
    ) -> dict:
        """CQ-0239: teacher match diagnostics (post-training)"""
        diag: dict = {}
        self._model.eval()
        with torch.no_grad():
            # discard
            if discard_samples:
                obs = torch.tensor(
                    np.stack([s.observation for s in discard_samples]),
                    dtype=torch.float32, device=self._device)
                masks = torch.tensor(
                    np.stack([s.legal_mask for s in discard_samples]),
                    dtype=torch.float32, device=self._device)
                out = self._model.forward_discard(obs, masks)
                pred = out.discard_logits.argmax(dim=-1).cpu().numpy()
                top1_match = sum(
                    1 for i, s in enumerate(discard_samples)
                    if s.teacher_top1_index >= 0 and pred[i] == s.teacher_top1_index
                )
                n_with_teacher = sum(1 for s in discard_samples if s.teacher_top1_index >= 0)
                diag["teacher_top1_match_rate_discard"] = (
                    top1_match / n_with_teacher if n_with_teacher > 0 else None)
                # best-set hit rate
                best_hit = 0
                n_best = 0
                for i, s in enumerate(discard_samples):
                    if s.teacher_best_indices:
                        n_best += 1
                        if pred[i] in s.teacher_best_indices:
                            best_hit += 1
                diag["teacher_best_set_hit_rate_discard"] = (
                    best_hit / n_best if n_best > 0 else None)

            # CQ-0247: optional diagnostics batched
            if call_samples:
                teacher_samples = [s for s in call_samples if s.teacher_top1_index >= 0]
                n_with_teacher = len(teacher_samples)
                top1_match = 0
                if n_with_teacher > 0:
                    max_cands = max(s.candidate_count for s in teacher_samples)
                    bs = min(self._batch_size, n_with_teacher)
                    for start in range(0, n_with_teacher, bs):
                        batch = teacher_samples[start:start + bs]
                        obs_t = torch.tensor(
                            np.stack([s.observation for s in batch]),
                            dtype=torch.float32, device=self._device)
                        cf, cm = _encode_candidates_batch(batch, max_cands)
                        cf = cf.to(self._device)
                        cm = cm.to(self._device)
                        rc_list = [s.response_context
                                   if s.response_context is not None
                                   else np.zeros(3, dtype=np.float32)
                                   for s in batch]
                        rc_t = torch.tensor(np.stack(rc_list),
                                            dtype=torch.float32, device=self._device)
                        out = self._model.forward_optional(
                            obs_t, cf, cm, response_context=rc_t,
                            compute_value=False)
                        preds = out.optional_scores.argmax(dim=-1).cpu().numpy()
                        for j, s in enumerate(batch):
                            if preds[j] == s.teacher_top1_index:
                                top1_match += 1
                diag["teacher_top1_match_rate_optional"] = (
                    top1_match / n_with_teacher if n_with_teacher > 0 else None)

        self._model.train()
        return diag

    def _compute_ppo_branch_targets(
        self,
        discard_samples: list[DecisionSample],
        call_samples: list[DecisionSample],
        is_mixed: bool = False,
    ) -> dict:
        """CQ-0275: PPO の advantage / return / weight を branch 元順で返す

        all_indexed を (step_id, branch, branch_idx, sample) で持ち、
        step_id 順 GAE 計算結果を branch 元 index 位置に scatter する。
        """
        nd = len(discard_samples)
        nc = len(call_samples)

        # all_indexed: (step_id, branch, branch_idx, sample)
        all_indexed: list[tuple[int, str, int, DecisionSample]] = []
        for i, s in enumerate(discard_samples):
            all_indexed.append((s.step_id, "discard", i, s))
        for i, s in enumerate(call_samples):
            all_indexed.append((s.step_id, "call", i, s))
        all_indexed.sort(key=lambda x: x[0])  # step_id 順

        all_sorted = [s for _, _, _, s in all_indexed]
        all_ret, all_adv = self._compute_returns_advantages(all_sorted)

        # CQ-0250: advantage を全体で一度だけ正規化
        if all_adv.numel() > 1:
            all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
        if self._advantage_clip is not None:
            all_adv = all_adv.clamp(-self._advantage_clip, self._advantage_clip)

        # branch 元順 tensor を初期化
        device = self._device
        d_ret_t = torch.zeros(nd, dtype=torch.float32, device=device) if nd > 0 else None
        d_adv_t = torch.zeros(nd, dtype=torch.float32, device=device) if nd > 0 else None
        d_weights_t = torch.ones(nd, dtype=torch.float32, device=device) if nd > 0 else None
        c_ret_t = torch.zeros(nc, dtype=torch.float32, device=device) if nc > 0 else None
        c_adv_t = torch.zeros(nc, dtype=torch.float32, device=device) if nc > 0 else None
        c_weights_t = torch.ones(nc, dtype=torch.float32, device=device) if nc > 0 else None

        # CQ-0275: sorted_idx の値を branch_idx 位置に scatter
        for sorted_idx, (_, branch, branch_idx, s) in enumerate(all_indexed):
            w = (self._baseline_sample_weight
                 if is_mixed and s.actor_type == "baseline" else 1.0)
            if branch == "discard":
                d_ret_t[branch_idx] = all_ret[sorted_idx]
                d_adv_t[branch_idx] = all_adv[sorted_idx]
                d_weights_t[branch_idx] = w
            else:
                c_ret_t[branch_idx] = all_ret[sorted_idx]
                c_adv_t[branch_idx] = all_adv[sorted_idx]
                c_weights_t[branch_idx] = w

        return {
            "d_ret": d_ret_t, "d_adv": d_adv_t, "d_weights": d_weights_t,
            "c_ret": c_ret_t, "c_adv": c_adv_t, "c_weights": c_weights_t,
            "all_adv": all_adv, "all_sorted": all_sorted,
        }

    def _compute_returns_advantages(self, samples):
        """CQ-0237: same-player grouped GAE"""
        from collections import defaultdict
        n = len(samples)
        advantages = np.zeros(n, dtype=np.float64)
        returns_arr = np.zeros(n, dtype=np.float64)

        # group by (episode_id, player_id) for same-player trajectory
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            groups[(s.episode_id, s.player_id)].append(i)

        for key, indices in groups.items():
            # indices は step_id 順（shard 書き出し順）
            grp_rewards = np.array([samples[i].reward for i in indices], dtype=np.float64)
            grp_values = np.array([samples[i].value for i in indices], dtype=np.float64)
            grp_term = np.array([samples[i].terminated for i in indices], dtype=np.float64)
            g = len(indices)
            last_gae = 0.0
            for t in reversed(range(g)):
                if t == g - 1 or grp_term[t]:
                    next_value = 0.0
                    last_gae = 0.0
                else:
                    next_value = grp_values[t + 1]
                delta = grp_rewards[t] + self._gamma * next_value - grp_values[t]
                last_gae = delta + self._gamma * self._gae_lambda * last_gae
                advantages[indices[t]] = last_gae
            for t in range(g):
                returns_arr[indices[t]] = advantages[indices[t]] + grp_values[t]

        return (torch.tensor(returns_arr, dtype=torch.float32, device=self._device),
                torch.tensor(advantages, dtype=torch.float32, device=self._device))

    def _normalize_advantage(self, advantage: torch.Tensor) -> torch.Tensor:
        # CQ-0250: advantage は _train_ppo で全体一括正規化済み。
        # minibatch ごとの再正規化はしない。
        return advantage

    def _compute_value_loss(self, value: torch.Tensor,
                            target: torch.Tensor) -> torch.Tensor:
        if self._value_loss_type == "huber":
            return F.huber_loss(value, target, delta=self._huber_delta)
        return F.mse_loss(value, target)

    def _compute_semantic_aux_loss(
        self, semantic: dict | None,
        terminal_targets: torch.Tensor,
        yaku_targets: torch.Tensor,
        is_winner: torch.Tensor,
        terminal_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float, float]:
        """CQ-0256/0268: semantic aux loss (terminal CE + winner-only yaku BCE)

        terminal_weights: CQ-0268 player-round 正規化重み (optional)
        """
        if semantic is None or not self._semantic_aux_enabled:
            z = torch.tensor(0.0, device=self._device)
            return z, 0.0, 0.0
        # CQ-0268: terminal CE with player-round weights
        tl_per = F.cross_entropy(
            semantic["terminal_logits"], terminal_targets, reduction="none")
        if terminal_weights is not None:
            tl = (tl_per * terminal_weights).sum()
        else:
            tl = tl_per.mean()
        # yaku: winner-only BCE (unchanged)
        yl_per = F.binary_cross_entropy_with_logits(
            semantic["yaku_logits"], yaku_targets, reduction="none")
        mask = is_winner.unsqueeze(-1)  # (B, 1)
        if mask.sum() > 0:
            yl = (yl_per * mask).sum() / mask.sum() / yl_per.size(-1)
        else:
            yl = torch.tensor(0.0, device=self._device)
        total = self._terminal_loss_coef * tl + self._yaku_loss_coef * yl
        return total, tl.item(), yl.item()

    def _compute_value_loss_per_sample(self, value: torch.Tensor,
                                        target: torch.Tensor) -> torch.Tensor:
        """per-sample value loss (reduction=none) for weighted mean"""
        if self._value_loss_type == "huber":
            return F.huber_loss(value, target, reduction="none",
                                delta=self._huber_delta)
        return (value - target).pow(2)

    def _ppo_discard_epoch(self, samples, returns_t, advantages_t,
                           weights_t=None,
                           terminal_classes_t=None, yaku_multihot_t=None,
                           is_winner_t=None, terminal_weights_t=None):
        """discard PPO の 1 epoch (CQ-0249: weighted mean)"""
        policy_losses, value_losses, entropies = [], [], []
        all_ratios = []
        _anchor_kls: list[float] = []
        _sem_tl: list[float] = []
        _sem_yl: list[float] = []
        n = len(samples)
        indices = np.random.permutation(n)

        for start in range(0, n, self._batch_size):
            end = min(start + self._batch_size, n)
            batch_idx = indices[start:end]
            batch = [samples[i] for i in batch_idx]
            idx_t = torch.tensor(batch_idx, dtype=torch.long)

            obs = torch.tensor(
                np.stack([s.observation for s in batch]),
                dtype=torch.float32, device=self._device)
            masks = torch.tensor(
                np.stack([s.legal_mask for s in batch]),
                dtype=torch.float32, device=self._device)
            actions = torch.tensor(
                [s.action for s in batch],
                dtype=torch.long, device=self._device)
            old_log_probs = torch.tensor(
                [s.log_prob for s in batch],
                dtype=torch.float32, device=self._device)
            batch_returns = returns_t[idx_t]
            batch_advantages = self._normalize_advantage(advantages_t[idx_t])

            out = self._model.forward_discard(obs, masks)
            log_probs = F.log_softmax(out.discard_logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            # CQ-0249: per-sample weighted mean
            batch_w = (weights_t[idx_t] if weights_t is not None
                       else torch.ones(len(batch), device=self._device))
            w_sum = batch_w.sum().clamp(min=1e-8)

            probs = torch.softmax(out.discard_logits, dim=-1)
            ent_per = -(probs * log_probs).sum(dim=-1)
            entropy = (ent_per * batch_w).sum() / w_sum

            ratio = torch.exp(action_log_probs - old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self._clip_epsilon,
                                1 + self._clip_epsilon) * batch_advantages
            surr_min = -torch.min(surr1, surr2)
            policy_loss = (surr_min * batch_w).sum() / w_sum

            value = out.values["round_delta"].squeeze(-1)
            vl_per = self._compute_value_loss_per_sample(value, batch_returns)
            value_loss = (vl_per * batch_w).sum() / w_sum

            loss = (policy_loss
                    + self._value_loss_coef * value_loss
                    - self._entropy_coef * entropy)

            akl_val = 0.0
            if self._anchor_enabled and self._anchor_model is not None:
                anchor_kl, akl_val = self._compute_anchor_kl_discard(obs, masks, out)
                loss = loss + self._anchor_coef * anchor_kl

            # CQ-0256/0268: semantic aux loss with terminal weights
            if self._semantic_aux_enabled and terminal_classes_t is not None:
                tw = terminal_weights_t[idx_t] if terminal_weights_t is not None else None
                sa_loss, tl_v, yl_v = self._compute_semantic_aux_loss(
                    out.semantic, terminal_classes_t[idx_t],
                    yaku_multihot_t[idx_t], is_winner_t[idx_t],
                    terminal_weights=tw)
                loss = loss + sa_loss
                _sem_tl.append(tl_v)
                _sem_yl.append(yl_v)

            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            all_ratios.append(ratio.detach().cpu())
            if self._anchor_enabled and self._anchor_model is not None:
                _anchor_kls.append(akl_val)

        return (policy_losses, value_losses, entropies, all_ratios,
                _anchor_kls, _sem_tl, _sem_yl)

    def _ppo_call_epoch(self, samples, returns_t, advantages_t,
                        weights_t=None,
                        terminal_classes_t=None, yaku_multihot_t=None,
                        is_winner_t=None, terminal_weights_t=None):
        """optional PPO の 1 epoch (CQ-0249: weighted mean)"""
        policy_losses, value_losses, entropies = [], [], []
        all_ratios = []
        _anchor_kls: list[float] = []
        _sem_tl: list[float] = []
        _sem_yl: list[float] = []
        max_cands = max(s.candidate_count for s in samples)
        n = len(samples)
        indices = np.random.permutation(n)

        for start in range(0, n, self._batch_size):
            end = min(start + self._batch_size, n)
            batch_idx = indices[start:end]
            batch = [samples[i] for i in batch_idx]
            idx_t = torch.tensor(batch_idx, dtype=torch.long)

            obs = torch.tensor(
                np.stack([s.observation for s in batch]),
                dtype=torch.float32, device=self._device)
            targets = torch.tensor(
                [s.selected_candidate_index for s in batch],
                dtype=torch.long, device=self._device)
            old_log_probs = torch.tensor(
                [s.log_prob for s in batch],
                dtype=torch.float32, device=self._device)
            batch_returns = returns_t[idx_t]
            batch_advantages = self._normalize_advantage(advantages_t[idx_t])

            cand_feats, cand_mask = _encode_candidates_batch(batch, max_cands)
            cand_feats = cand_feats.to(self._device)
            cand_mask = cand_mask.to(self._device)

            # CQ-0242: response_context from shard
            rc_list = [s.response_context if s.response_context is not None
                       else np.zeros(3, dtype=np.float32)
                       for s in batch]
            rc_t = torch.tensor(np.stack(rc_list),
                                dtype=torch.float32, device=self._device)

            out = self._model.forward_optional(
                obs, cand_feats, cand_mask, response_context=rc_t)
            log_probs = F.log_softmax(out.optional_scores, dim=-1)
            action_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            # CQ-0249: per-sample weighted mean
            batch_w = (weights_t[idx_t] if weights_t is not None
                       else torch.ones(len(batch), device=self._device))
            w_sum = batch_w.sum().clamp(min=1e-8)

            probs = torch.softmax(out.optional_scores, dim=-1)
            ent_per = -(probs * log_probs).sum(dim=-1)
            entropy = (ent_per * batch_w).sum() / w_sum

            ratio = torch.exp(action_log_probs - old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self._clip_epsilon,
                                1 + self._clip_epsilon) * batch_advantages
            surr_min = -torch.min(surr1, surr2)
            policy_loss = (surr_min * batch_w).sum() / w_sum

            value = out.values["round_delta"].squeeze(-1)
            vl_per = self._compute_value_loss_per_sample(value, batch_returns)
            value_loss = (vl_per * batch_w).sum() / w_sum

            loss = (policy_loss
                    + self._value_loss_coef * value_loss
                    - self._entropy_coef * entropy)

            akl_val = 0.0
            if self._anchor_enabled and self._anchor_model is not None:
                anchor_kl, akl_val = self._compute_anchor_kl_optional(
                    obs, cand_feats, cand_mask, out,
                    response_context=rc_t)
                loss = loss + self._anchor_coef * anchor_kl

            # CQ-0256/0268: semantic aux loss with terminal weights
            if self._semantic_aux_enabled and terminal_classes_t is not None:
                tw = terminal_weights_t[idx_t] if terminal_weights_t is not None else None
                sa_loss, tl_v, yl_v = self._compute_semantic_aux_loss(
                    out.semantic, terminal_classes_t[idx_t],
                    yaku_multihot_t[idx_t], is_winner_t[idx_t],
                    terminal_weights=tw)
                loss = loss + sa_loss
                _sem_tl.append(tl_v)
                _sem_yl.append(yl_v)

            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            all_ratios.append(ratio.detach().cpu())
            if self._anchor_enabled and self._anchor_model is not None:
                _anchor_kls.append(akl_val)

        return (policy_losses, value_losses, entropies, all_ratios,
                _anchor_kls, _sem_tl, _sem_yl)
