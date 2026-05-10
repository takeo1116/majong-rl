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
        # CQ-0282: mixed PPO は baseline actor sample が PPO ratio を off-policy 化
        # しうるため、明示 opt-in が無いと train() で fail-fast する。
        self._allow_mixed_offpolicy_baseline = bool(
            rml.get("allow_mixed_offpolicy_baseline", False))
        # CQ-0256: semantic aux
        sa = tc.get("semantic_aux", {})
        self._semantic_aux_enabled = sa.get("enabled", False)
        self._terminal_loss_coef = sa.get("terminal_loss_coef", 0.2)
        self._yaku_loss_coef = sa.get("yaku_loss_coef", 0.1)
        # CQ-0284: gradient norm diagnostics (default off)
        gn = tc.get("diagnostics", {}).get("gradient_norms", {})
        self._gn_enabled = bool(gn.get("enabled", False))
        self._gn_max_batches_per_epoch = int(gn.get("max_batches_per_epoch", 4))
        self._gn_every_n_epochs = max(1, int(gn.get("every_n_epochs", 1)))
        # CQ-0287: PPO target KL early stop (default off)
        # 各 minibatch の forward 後に ((ratio-1) - log_ratio).mean() で
        # approx KL を計算し、`target * stop_multiplier` を超えたら
        # その branch epoch の残り minibatch を early stop する。
        # default off では既存学習挙動と完全互換。
        tk = tc.get("ppo_target_kl", {}) or {}
        self._tk_enabled = bool(tk.get("enabled", False))
        self._tk_target = float(tk.get("target", 0.03))
        self._tk_stop_multiplier = float(tk.get("stop_multiplier", 1.5))
        self._tk_skip_on_exceed = bool(tk.get("skip_minibatch_on_exceed", True))
        self._tk_threshold = self._tk_target * self._tk_stop_multiplier

        # CQ-0286: optimizer parameter groups (policy / value_semantic) lr 分離
        # default off: 既存と完全互換 (single group, lr=self._lr)
        # CQ-0289: `apply_to` 指定で algorithm scope (ppo / imitation) を分離
        lrg = tc.get("lr_groups", {}) or {}
        self._lr_groups_enabled = bool(lrg.get("enabled", False))
        self._lr_groups_apply_to = self._validate_lr_groups_apply_to(
            lrg.get("apply_to", ["ppo", "imitation"]))
        # 現在の algorithm で lr_groups が active か
        self._lr_groups_active = (
            self._lr_groups_enabled
            and self._mode in self._lr_groups_apply_to
        )
        self._optimizer, self._lr_groups_info = self._build_optimizer(
            model, base_lr=self._lr, lr_groups_cfg=lrg,
            active=self._lr_groups_active,
            requested_enabled=self._lr_groups_enabled,
            apply_to=self._lr_groups_apply_to,
            algorithm=self._mode)

    # CQ-0289: lr_groups の適用 scope (algorithm 名)
    _LR_GROUPS_KNOWN_ALGORITHMS: tuple[str, ...] = ("ppo", "imitation")

    @classmethod
    def _validate_lr_groups_apply_to(cls, value) -> list[str]:
        """CQ-0289: apply_to を validate して list[str] で返す。

        - list / tuple 以外は ValueError
        - 空 list は ValueError
        - 未知の algorithm 名は ValueError
        """
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"CQ-0289: training.lr_groups.apply_to は list である必要が"
                f" あります (got {type(value).__name__}: {value!r})")
        items = [str(x) for x in value]
        if not items:
            raise ValueError(
                "CQ-0289: training.lr_groups.apply_to が空 list です。"
                " 適用したい algorithm を 1 つ以上指定するか、"
                " lr_groups.enabled=false にしてください。")
        unknown = [x for x in items if x not in cls._LR_GROUPS_KNOWN_ALGORITHMS]
        if unknown:
            raise ValueError(
                f"CQ-0289: training.lr_groups.apply_to に未知の algorithm が"
                f" 含まれます: {unknown}."
                f" 受け入れる値: {list(cls._LR_GROUPS_KNOWN_ALGORITHMS)}")
        return items

    def _build_optimizer(self, model, base_lr: float, lr_groups_cfg: dict,
                          *,
                          active: bool | None = None,
                          requested_enabled: bool | None = None,
                          apply_to: list[str] | None = None,
                          algorithm: str | None = None,
                         ) -> tuple["torch.optim.Optimizer", dict]:
        """CQ-0286 / CQ-0289: optimizer 構築。`lr_groups.enabled=True` かつ
        現在の algorithm が `apply_to` に含まれるとき、policy / value_semantic
        の lr を分離する。

        default off (既存互換):
            Adam(model.parameters(), lr=base_lr) — 1 group のみ
        active:
            Adam([{"params": policy, "lr": policy_lr},
                  {"params": value_semantic, "lr": value_semantic_lr},
                  {"params": default, "lr": default_lr}], lr=base_lr)
            (default group は unknown trainable parameter があるときのみ
             optimizer に追加される)

        ``active`` が False のときは `lr_groups.enabled=true` 指定でも
        single group optimizer を返す (CQ-0289: scope 外 algorithm の場合)。

        重複チェック (`id`) と取りこぼしチェックを行い、不一致なら
        ValueError で fail-fast する。
        """
        cfg_enabled = bool(lr_groups_cfg.get("enabled", False))
        # 後方互換: active が明示されない場合は cfg_enabled を使う
        if active is None:
            active = cfg_enabled
        if requested_enabled is None:
            requested_enabled = cfg_enabled

        info: dict = {
            "enabled": bool(active),
            # CQ-0289: 追加 diagnostics keys
            "requested_enabled": bool(requested_enabled),
            "active_for_algorithm": bool(active),
            "apply_to": list(apply_to) if apply_to is not None else None,
            "algorithm": algorithm,
            "groups": {},
        }

        if not active:
            opt = torch.optim.Adam(model.parameters(), lr=base_lr)
            # 既存挙動互換のため、info も最小限
            info["groups"]["all"] = {
                "lr": float(base_lr),
                "param_count": int(sum(p.numel() for p in model.parameters()
                                       if p.requires_grad)),
                "tensor_count": int(sum(1 for p in model.parameters()
                                        if p.requires_grad)),
            }
            return opt, info

        policy_lr = float(lr_groups_cfg.get("policy", base_lr))
        value_semantic_lr = float(lr_groups_cfg.get("value_semantic", base_lr))
        default_lr = float(lr_groups_cfg.get("default", base_lr))

        named_trainable = [(n, p) for n, p in model.named_parameters()
                           if p.requires_grad]
        classify = self._classify_param_groups(named_trainable)
        policy_named = classify["policy"]
        vs_named = classify["value_semantic"]
        default_named = classify["default"]

        # 重複チェック (id ベース): 各 named param は厳密に 1 group
        all_ids: set[int] = set()
        for label, lst in (("policy", policy_named),
                           ("value_semantic", vs_named),
                           ("default", default_named)):
            for n, p in lst:
                pid = id(p)
                if pid in all_ids:
                    raise ValueError(
                        f"CQ-0286: parameter {n!r} が複数の group に重複"
                        f" 分類されました ({label})")
                all_ids.add(pid)
        # 取りこぼしチェック: trainable で classifier に拾われなかった
        # parameter がないこと
        expected_ids = {id(p) for _, p in named_trainable}
        missing = expected_ids - all_ids
        if missing:
            missing_names = [n for n, p in named_trainable
                             if id(p) in missing]
            raise ValueError(
                f"CQ-0286: trainable parameter が optimizer group に "
                f"取りこぼされました: {missing_names}")
        extra = all_ids - expected_ids
        if extra:
            raise ValueError(
                f"CQ-0286: classifier が trainable でない parameter を "
                f"group に入れました (id 数={len(extra)})")

        param_groups: list[dict] = []
        if policy_named:
            param_groups.append({
                "params": [p for _, p in policy_named],
                "lr": policy_lr,
                "name": "policy",
            })
        if vs_named:
            param_groups.append({
                "params": [p for _, p in vs_named],
                "lr": value_semantic_lr,
                "name": "value_semantic",
            })
        if default_named:
            param_groups.append({
                "params": [p for _, p in default_named],
                "lr": default_lr,
                "name": "default",
            })

        if not param_groups:
            raise ValueError(
                "CQ-0286: optimizer に追加する param group がありません "
                "(trainable parameter が 0 件)")

        opt = torch.optim.Adam(param_groups, lr=base_lr)

        # diagnostics info
        def _gstats(named_list):
            return {
                "param_count": int(sum(p.numel() for _, p in named_list)),
                "tensor_count": int(len(named_list)),
            }
        info["groups"]["policy"] = {
            "lr": policy_lr, **_gstats(policy_named)}
        info["groups"]["value_semantic"] = {
            "lr": value_semantic_lr, **_gstats(vs_named)}
        if default_named:
            info["groups"]["default"] = {
                "lr": default_lr,
                **_gstats(default_named),
                "names": [n for n, _ in default_named],
            }
        else:
            info["groups"]["default"] = {
                "lr": default_lr,
                "param_count": 0,
                "tensor_count": 0,
                "names": [],
            }
        return opt, info

    # CQ-0286: parameter group 分類規則 (Stage2aModel の module 階層に基づく)
    _LR_POLICY_PREFIXES: tuple[str, ...] = (
        "discard_trunk", "discard_head", "optional_trunk",
        "candidate_encoder", "optional_scorer",
        "_tile_embedding", "_local_scorer", "_context_gate",
    )
    _LR_VALUE_SEMANTIC_PREFIXES: tuple[str, ...] = (
        # CQ-0288: semantic_proj は削除済み
        "value_trunk", "value_head",
        "terminal_head", "yaku_head",
    )

    @classmethod
    def _classify_param_groups(
        cls, named_params: list[tuple[str, "nn.Parameter"]],
    ) -> dict[str, list[tuple[str, "nn.Parameter"]]]:
        """CQ-0286: named parameters を policy / value_semantic / default に
        分類する。

        Returns:
            {"policy": [...], "value_semantic": [...], "default": [...]}
            それぞれ (name, param) のリスト。default は分類規則のどれにも
            一致しなかった trainable parameter。
        """
        out: dict[str, list[tuple[str, "nn.Parameter"]]] = {
            "policy": [], "value_semantic": [], "default": [],
        }
        for name, p in named_params:
            top = name.split(".", 1)[0]
            if top in cls._LR_POLICY_PREFIXES:
                out["policy"].append((name, p))
            elif top in cls._LR_VALUE_SEMANTIC_PREFIXES:
                out["value_semantic"].append((name, p))
            else:
                out["default"].append((name, p))
        return out

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

    @staticmethod
    def _compute_terminal_weights_cross_branch(
        d_episode_ids, d_round_ids, d_player_ids,
        c_episode_ids, c_round_ids, c_player_ids,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """CQ-0277: discard/call 横断で player-round 正規化重みを計算する

        同じ (episode_id, round_id, player_id) に属する row が discard と call に
        分かれていても、合計 weight が 1.0 になるように、両 branch を結合した
        母集団でカウントしてから branch 元順で返す。

        片方 branch が空の場合は、もう片方の branch だけで CQ-0268 と同じ結果に
        なる (片 branch のみのケース後方互換)。

        Returns:
            (d_weights, c_weights): それぞれ branch 元順、空 branch は None
        """
        from collections import Counter
        nd = len(d_episode_ids) if d_episode_ids is not None else 0
        nc = len(c_episode_ids) if c_episode_ids is not None else 0

        def _key(eid, rid, pid):
            eid_s = eid if isinstance(eid, str) else str(eid)
            return (eid_s, int(rid), int(pid))

        d_keys = []
        for i in range(nd):
            d_keys.append(_key(d_episode_ids[i], d_round_ids[i], d_player_ids[i]))
        c_keys = []
        for i in range(nc):
            c_keys.append(_key(c_episode_ids[i], c_round_ids[i], c_player_ids[i]))

        # 横断 count
        counts = Counter(d_keys)
        counts.update(c_keys)

        d_w = None
        if nd > 0:
            d_w = torch.zeros(nd, dtype=torch.float32, device=device)
            for i, k in enumerate(d_keys):
                d_w[i] = 1.0 / counts[k]
        c_w = None
        if nc > 0:
            c_w = torch.zeros(nc, dtype=torch.float32, device=device)
            for i, k in enumerate(c_keys):
                c_w[i] = 1.0 / counts[k]
        return d_w, c_w

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

    REQUIRED_SAMPLE_SEMANTICS_VERSION = 3  # CQ-0279

    @classmethod
    def _check_sample_semantics_version(
        cls, branch_data: dict | None, branch_name: str,
    ) -> None:
        """CQ-0279: shard sample_semantics_version を検証する

        v3 未満が含まれていたら ValueError で fail-fast。
        CQ-0274 で reward semantics が変わったため、旧 shard を学習に使うと
        return / advantage の意味が崩れる。
        """
        if branch_data is None or branch_data.get("n", 0) == 0:
            return
        versions = branch_data.get("sample_semantics_versions")
        if versions is None or len(versions) == 0:
            return
        min_ver = int(np.asarray(versions).min())
        required = cls.REQUIRED_SAMPLE_SEMANTICS_VERSION
        if min_ver < required:
            raise ValueError(
                f"Stage2a {branch_name} shard に "
                f"sample_semantics_version={min_ver} の sample が含まれます "
                f"(required: {required})。"
                f" CQ-0274 前の旧 reward semantics shard である可能性が高く、"
                f" return / advantage の意味が崩れるため fail-fast します。"
                f" Stage2a selfplay で shard を再生成してください。"
            )

    def train(self, shard_dir: Path, num_epochs: int | None = None,
              filter_actor_type: str | None = None) -> dict:
        """CQ-0251: tensor ベースで shard を読み込んで学習"""
        reader = DecisionShardReader(shard_dir)
        data = reader.read_as_tensors(filter_actor_type=filter_actor_type)

        d = data["discard"]
        c = data["call"]
        nd = d["n"] if d else 0
        nc = c["n"] if c else 0

        # CQ-0279: legacy shard fail-fast (read_as_tensors 経路)
        self._check_sample_semantics_version(d, "discard")
        self._check_sample_semantics_version(c, "call")

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

        # CQ-0282: mixed PPO + baseline_sample_weight > 0 は明示 opt-in が無ければ
        # fail-fast。baseline actor sample は learned policy から sample されて
        # いないため、PPO ratio が off-policy になる (exp_022 collapse の主因候補)。
        if (self._mode == "ppo" and self._mixed_ppo_mode == "mixed"
                and self._baseline_sample_weight > 0.0
                and not filter_actor_type
                and not self._allow_mixed_offpolicy_baseline):
            raise ValueError(
                "Stage02a mixed PPO: baseline actor sample is not sampled from "
                "learned policy. PPO ratio may be off-policy and lead to "
                "ratio explosion / late-cycle collapse (see exp_022). "
                "Use ppo_mode='separated' unless explicitly intended. "
                "If you really want mixed PPO, set "
                "training.rule_mix_learner.allow_mixed_offpolicy_baseline=true. "
                f"(ppo_mode={self._mixed_ppo_mode!r}, "
                f"baseline_sample_weight={self._baseline_sample_weight})"
            )

        if nd + nc == 0:
            return {"mode": self._mode, "total_steps": 0,
                    "policy_loss": 0.0, "value_loss": 0.0, "num_updates": 0,
                    "ppo_mode": (
                        "separated" if filter_actor_type == "policy"
                        else self._mixed_ppo_mode),
                    "executed": False,
                    "used_policy_samples": 0,
                    "used_baseline_samples": 0,
                    "excluded_baseline_samples": 0}

        epochs = num_epochs or self._epochs

        if self._mode == "imitation":
            return self._train_imitation_tensor(d, c, epochs)
        else:
            # PPO は sample list path を安全に使う (candidate semantics 維持)
            all_samples = reader.read_all()
            # CQ-0279: read_all 経路でも fail-fast (filter 前)
            if all_samples:
                min_ver = min(s.sample_semantics_version for s in all_samples)
                required = self.REQUIRED_SAMPLE_SEMANTICS_VERSION
                if min_ver < required:
                    raise ValueError(
                        f"Stage2a shard (read_all) に "
                        f"sample_semantics_version={min_ver} の sample が含まれます "
                        f"(required: {required})。"
                        f" CQ-0274 前の旧 reward semantics shard である可能性が高く、"
                        f" return / advantage の意味が崩れるため fail-fast します。"
                        f" Stage2a selfplay で shard を再生成してください。"
                    )
            # CQ-0282: filter 前に actor_type 別件数を集計
            n_policy_total = sum(1 for s in all_samples
                                 if s.actor_type == "policy")
            n_baseline_total = sum(1 for s in all_samples
                                   if s.actor_type == "baseline")
            if filter_actor_type:
                all_samples = [s for s in all_samples
                               if s.actor_type == filter_actor_type]
            d_samples = [s for s in all_samples if s.decision_type == "discard"]
            c_samples = [s for s in all_samples if s.decision_type == "call"]
            result = self._train_ppo(d_samples, c_samples, epochs)
            # CQ-0282: separated/mixed の運用 metadata を summary に残す
            ppo_mode_eff = ("separated" if filter_actor_type == "policy"
                            else self._mixed_ppo_mode)
            result["ppo_mode"] = ppo_mode_eff
            result["executed"] = True
            if ppo_mode_eff == "separated":
                result["used_policy_samples"] = n_policy_total
                result["used_baseline_samples"] = 0
                result["excluded_baseline_samples"] = n_baseline_total
            else:
                # mixed: baseline も PPO に含まれる
                result["used_policy_samples"] = n_policy_total
                result["used_baseline_samples"] = n_baseline_total
                result["excluded_baseline_samples"] = 0
            return result

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

        # CQ-0277: terminal player-round weights (discard/call 横断)
        if self._semantic_aux_enabled:
            d_eps = d["episode_ids"] if d else None
            d_rds = (d.get("round_ids", np.zeros(nd, dtype=np.int64))
                     if d else None)
            d_pids = d["player_ids"] if d else None
            c_eps = c["episode_ids"] if c else None
            c_rds = (c.get("round_ids", np.zeros(nc, dtype=np.int64))
                     if c else None)
            c_pids = c["player_ids"] if c else None
            d_term_weights, c_term_weights = (
                self._compute_terminal_weights_cross_branch(
                    d_eps, d_rds, d_pids, c_eps, c_rds, c_pids, self._device))

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
                        sa_loss, tl_t, yl_t = self._compute_semantic_aux_loss(
                            out.semantic, d_term_cls[idx], d_yaku_mh[idx], d_is_winner[idx],
                            terminal_weights=tw)
                        total = total + sa_loss
                        sem_tl_list.append(float(tl_t.detach()))
                        sem_yl_list.append(float(yl_t.detach()))

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
                        sa_loss, tl_t, yl_t = self._compute_semantic_aux_loss(
                            out.semantic, c_term_cls[idx], c_yaku_mh[idx], c_is_winner[idx],
                            terminal_weights=tw)
                        total = total + sa_loss
                        sem_tl_list.append(float(tl_t.detach()))
                        sem_yl_list.append(float(yl_t.detach()))

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
            "optimizer_lr_groups": self._lr_groups_info,  # CQ-0286
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

        # CQ-0268/0277: terminal player-round weights (discard/call 横断)
        d_term_weights = c_term_weights = None
        if self._semantic_aux_enabled:
            d_eps = [s.episode_id for s in discard_samples] if discard_samples else None
            d_rds = [s.round_id for s in discard_samples] if discard_samples else None
            d_pids = [s.player_id for s in discard_samples] if discard_samples else None
            c_eps = [s.episode_id for s in call_samples] if call_samples else None
            c_rds = [s.round_id for s in call_samples] if call_samples else None
            c_pids = [s.player_id for s in call_samples] if call_samples else None
            d_term_weights, c_term_weights = (
                self._compute_terminal_weights_cross_branch(
                    d_eps, d_rds, d_pids, c_eps, c_rds, c_pids, self._device))

        all_sem_tl: list[float] = []
        all_sem_yl: list[float] = []

        # CQ-0281: branch別 diagnostics buffer
        d_diag_bufs: list[dict] = []
        c_diag_bufs: list[dict] = []

        for epoch_idx in range(epochs):
            if discard_samples:
                pl, vl, ent, rats, akl, stl, syl, dbuf = self._ppo_discard_epoch(
                    discard_samples, d_ret, d_adv, d_weights,
                    terminal_classes_t=d_term_cls,
                    yaku_multihot_t=d_yaku_mh,
                    is_winner_t=d_is_winner,
                    terminal_weights_t=d_term_weights,
                    epoch_idx=epoch_idx)
                all_policy_losses.extend(pl)
                all_value_losses.extend(vl)
                all_entropies.extend(ent)
                all_ratios.extend(rats)
                all_anchor_kl_discard.extend(akl)
                all_sem_tl.extend(stl)
                all_sem_yl.extend(syl)
                d_diag_bufs.append(dbuf)
                num_updates += len(pl)

            if call_samples:
                pl, vl, ent, rats, akl, stl, syl, dbuf = self._ppo_call_epoch(
                    call_samples, c_ret, c_adv, c_weights,
                    terminal_classes_t=c_term_cls,
                    yaku_multihot_t=c_yaku_mh,
                    is_winner_t=c_is_winner,
                    terminal_weights_t=c_term_weights,
                    epoch_idx=epoch_idx)
                all_policy_losses.extend(pl)
                all_value_losses.extend(vl)
                all_entropies.extend(ent)
                all_ratios.extend(rats)
                all_anchor_kl_optional.extend(akl)
                all_sem_tl.extend(stl)
                all_sem_yl.extend(syl)
                c_diag_bufs.append(dbuf)
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

        # CQ-0281: 拡張 diagnostics (log_ratio quantiles / advantage sign /
        # cross stats / max_prob quantiles / branch 別)
        def _flatten_buf(bufs: list[dict], key: str) -> np.ndarray | None:
            parts = []
            for b in bufs:
                for t in b.get(key, []):
                    parts.append(t.numpy().astype(np.float64))
            if not parts:
                return None
            return np.concatenate(parts)

        d_lr = _flatten_buf(d_diag_bufs, "log_ratios")
        d_adv_b = _flatten_buf(d_diag_bufs, "advantages")
        d_mp = _flatten_buf(d_diag_bufs, "max_probs")
        d_w = _flatten_buf(d_diag_bufs, "weights")
        c_lr = _flatten_buf(c_diag_bufs, "log_ratios")
        c_adv_b = _flatten_buf(c_diag_bufs, "advantages")
        c_mp = _flatten_buf(c_diag_bufs, "max_probs")
        c_w = _flatten_buf(c_diag_bufs, "weights")

        # branch 別
        if d_lr is not None and len(d_lr) > 0:
            ppo_diag["discard"] = self._compute_ppo_diag_stats(
                d_lr, d_adv_b, d_mp, d_w, self._clip_epsilon)
        if c_lr is not None and len(c_lr) > 0:
            ppo_diag["call"] = self._compute_ppo_diag_stats(
                c_lr, c_adv_b, c_mp, c_w, self._clip_epsilon)

        # top-level aggregate (discard + call 結合)
        all_lr_parts = [x for x in (d_lr, c_lr) if x is not None and len(x) > 0]
        all_adv_b_parts = [x for x in (d_adv_b, c_adv_b)
                           if x is not None and len(x) > 0]
        all_mp_parts = [x for x in (d_mp, c_mp) if x is not None and len(x) > 0]
        all_w_parts = [x for x in (d_w, c_w) if x is not None and len(x) > 0]
        if all_lr_parts or all_adv_b_parts or all_mp_parts:
            top_lr = np.concatenate(all_lr_parts) if all_lr_parts else None
            top_adv = (np.concatenate(all_adv_b_parts)
                       if all_adv_b_parts else None)
            top_mp = np.concatenate(all_mp_parts) if all_mp_parts else None
            top_w = (np.concatenate(all_w_parts)
                     if (all_w_parts and len(all_w_parts) == len(all_lr_parts))
                     else None)
            top_stats = self._compute_ppo_diag_stats(
                top_lr, top_adv, top_mp, top_w, self._clip_epsilon)
            # 既存 key (ratio_mean / ratio_std / clip_fraction /
            # advantage_mean / advantage_std) は上書きしない
            for k, v in top_stats.items():
                if k not in ppo_diag:
                    ppo_diag[k] = v
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
                # CQ-0282: mixed PPO は baseline action が learned policy
                # から sample されていないため、PPO ratio が off-policy
                # になりうる。allow_mixed_offpolicy_baseline=True で
                # 明示 opt-in した場合に限りここまで到達する。
                "allow_mixed_offpolicy_baseline": (
                    self._allow_mixed_offpolicy_baseline),
                "warning": (
                    "baseline actor sample is not sampled from learned "
                    "policy; PPO ratio may be off-policy. "
                    "Use ppo_mode='separated' unless explicitly intended."),
            }

        # CQ-0284: gradient norm diagnostics aggregate (default off → empty)
        if self._gn_enabled:
            d_gn_minibatches: list[dict] = []
            c_gn_minibatches: list[dict] = []
            for b in d_diag_bufs:
                d_gn_minibatches.extend(b.get("gradient_norms", []))
            for b in c_diag_bufs:
                c_gn_minibatches.extend(b.get("gradient_norms", []))
            agg_minibatches = d_gn_minibatches + c_gn_minibatches
            agg_stats = self._gn_summarize_buffer(agg_minibatches)
            d_stats = self._gn_summarize_buffer(d_gn_minibatches)
            c_stats = self._gn_summarize_buffer(c_gn_minibatches)
            ppo_diag["gradient_norms"] = {
                "config": {
                    "enabled": True,
                    "max_batches_per_epoch": self._gn_max_batches_per_epoch,
                    "every_n_epochs": self._gn_every_n_epochs,
                },
                "aggregate": {
                    **agg_stats,
                    "ratios": self._gn_compute_ratios(agg_stats),
                },
                "discard": d_stats,
                "call": c_stats,
            }

        # CQ-0287: target_kl early-stop diagnostics (top-level + branch 別)
        d_tk = self._tk_summarize(d_diag_bufs)
        c_tk = self._tk_summarize(c_diag_bufs)
        all_tk = self._tk_summarize(d_diag_bufs + c_diag_bufs)
        # branch 別: 該当 branch dict があるときに merge
        if "discard" in ppo_diag:
            ppo_diag["discard"].update(d_tk)
        if "call" in ppo_diag:
            ppo_diag["call"].update(c_tk)
        # top-level (既存 key を上書きしない方針)
        for k, v in all_tk.items():
            if k not in ppo_diag:
                ppo_diag[k] = v

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
            "optimizer_lr_groups": self._lr_groups_info,  # CQ-0286
            "ppo_diag": ppo_diag,
        }

    def load_anchor(self, checkpoint_path: str) -> None:
        """CQ-0240: anchor model をロードする"""
        import copy
        from mahjong_rl.models.stage2a_model import load_stage2a_state_dict
        self._anchor_model = copy.deepcopy(self._model)
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # CQ-0288: 旧 checkpoint の semantic_proj.* keys を互換的に drop
        load_stage2a_state_dict(self._anchor_model, sd)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CQ-0256/0268: semantic aux loss (terminal CE + winner-only yaku BCE)

        terminal_weights: CQ-0268 player-round 正規化重み (optional)

        Returns (total, tl, yl) - all torch.Tensor scalars (CQ-0284: tl/yl
        as tensors so caller can use them for autograd-based gradient norm
        diagnostics).
        """
        if semantic is None or not self._semantic_aux_enabled:
            z = torch.tensor(0.0, device=self._device)
            return z, z.detach().clone(), z.detach().clone()
        # CQ-0268: terminal CE with player-round weights
        # CQ-0285: weighted path を mean lossスケール (sum / weight_sum) に
        # 正規化する。player-round 重複補正は維持しつつ、batch内の group 数
        # に比例した scale 膨張を避ける。
        tl_per = F.cross_entropy(
            semantic["terminal_logits"], terminal_targets, reduction="none")
        if terminal_weights is not None:
            w_sum = terminal_weights.sum().clamp_min(1e-8)
            tl = (tl_per * terminal_weights).sum() / w_sum
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
        return total, tl, yl

    def _compute_value_loss_per_sample(self, value: torch.Tensor,
                                        target: torch.Tensor) -> torch.Tensor:
        """per-sample value loss (reduction=none) for weighted mean"""
        if self._value_loss_type == "huber":
            return F.huber_loss(value, target, reduction="none",
                                delta=self._huber_delta)
        return (value - target).pow(2)

    # ------------------------------------------------------------------
    # CQ-0284: gradient norm diagnostics (default off)
    # ------------------------------------------------------------------

    @staticmethod
    def _gn_module_param_indices(named_params: list[tuple[str, "nn.Parameter"]],
                                  module_name: str) -> list[int]:
        """`module_name` 配下の parameter index を named_params から拾う"""
        prefix = module_name + "."
        return [i for i, (n, _) in enumerate(named_params) if n.startswith(prefix)]

    def _gn_build_param_groups(
        self, named_params: list[tuple[str, "nn.Parameter"]],
    ) -> dict[str, list[int]]:
        """CQ-0284: parameter group → named_params 上の index 集合

        存在しない module は欠落させる (空 list の group は dict に入れない)。
        """
        n = len(named_params)
        groups: dict[str, list[int]] = {"all": list(range(n))}

        # individual modules — Stage2aModel 上にあれば拾う
        # CQ-0288: semantic_proj は削除済み
        atomic_modules = [
            "discard_trunk", "discard_head", "optional_trunk",
            "candidate_encoder", "optional_scorer",
            "value_trunk", "value_head",
            "terminal_head", "yaku_head",
            "_tile_embedding", "_local_scorer", "_context_gate",
        ]
        for k in atomic_modules:
            idxs = self._gn_module_param_indices(named_params, k)
            if idxs:
                groups[k] = idxs

        # composite groups
        policy_module_names = [
            "discard_trunk", "discard_head", "optional_trunk",
            "candidate_encoder", "optional_scorer",
            "_tile_embedding", "_local_scorer", "_context_gate",
        ]
        policy_idxs: list[int] = []
        for k in policy_module_names:
            policy_idxs.extend(groups.get(k, []))
        if policy_idxs:
            groups["policy"] = policy_idxs

        # CQ-0288: semantic_proj は削除済み
        value_semantic_module_names = [
            "value_trunk", "value_head",
            "terminal_head", "yaku_head",
        ]
        vs_idxs: list[int] = []
        for k in value_semantic_module_names:
            vs_idxs.extend(groups.get(k, []))
        if vs_idxs:
            groups["value_semantic"] = vs_idxs

        return groups

    def _gn_should_measure(self, *, batch_idx_in_epoch: int,
                           epoch_idx: int) -> bool:
        """CQ-0284: 計測 budget 内かどうか"""
        if not self._gn_enabled:
            return False
        if self._gn_max_batches_per_epoch <= 0:
            return False
        if (epoch_idx % self._gn_every_n_epochs) != 0:
            return False
        if batch_idx_in_epoch >= self._gn_max_batches_per_epoch:
            return False
        return True

    def _gn_compute_minibatch_norms(
        self, *,
        policy_loss: "torch.Tensor",
        value_loss: "torch.Tensor",
        sa_loss_total: "torch.Tensor | None",
        terminal_loss_t: "torch.Tensor | None",
        yaku_loss_t: "torch.Tensor | None",
        total_loss: "torch.Tensor",
    ) -> dict[str, dict[str, float]] | None:
        """CQ-0284: 1 minibatch ぶんの component × group 別 gradient norm

        ``torch.autograd.grad`` を使うため、optimizer 用の ``.grad`` は汚染
        しない。``retain_graph=True`` を使うので、後段の ``loss.backward()``
        は通常通り動く。

        Returns dict[component_name][group_name] -> float (norm)。空 None。
        """
        named_params = [
            (n, p) for n, p in self._model.named_parameters() if p.requires_grad
        ]
        if not named_params:
            return None
        all_params = [p for _, p in named_params]

        groups = self._gn_build_param_groups(named_params)
        if not groups:
            return None

        components: list[tuple[str, "torch.Tensor", float]] = []
        components.append(("policy_loss", policy_loss, 1.0))
        components.append(("value_loss", value_loss, 1.0))
        components.append(("weighted_value_loss", value_loss,
                           float(self._value_loss_coef)))
        if (self._semantic_aux_enabled
                and terminal_loss_t is not None
                and terminal_loss_t.requires_grad):
            components.append(("terminal_loss", terminal_loss_t, 1.0))
            components.append(("weighted_terminal_loss", terminal_loss_t,
                               float(self._terminal_loss_coef)))
        if (self._semantic_aux_enabled
                and yaku_loss_t is not None
                and yaku_loss_t.requires_grad):
            components.append(("yaku_loss", yaku_loss_t, 1.0))
            components.append(("weighted_yaku_loss", yaku_loss_t,
                               float(self._yaku_loss_coef)))
        if (self._semantic_aux_enabled
                and sa_loss_total is not None
                and sa_loss_total.requires_grad):
            components.append(("semantic_aux_loss", sa_loss_total, 1.0))
        components.append(("total_loss_pre_clip", total_loss, 1.0))

        result: dict[str, dict[str, float]] = {}
        for cname, loss_t, scale in components:
            if loss_t is None:
                continue
            if not isinstance(loss_t, torch.Tensor):
                continue
            if not loss_t.requires_grad:
                # 例: semantic disabled で zero tensor のまま渡された等
                continue
            try:
                grads = torch.autograd.grad(
                    loss_t, all_params,
                    retain_graph=True, allow_unused=True,
                    create_graph=False)
            except RuntimeError:
                continue
            # per-param squared norm (detached, scalar)
            per_param_sq: list[float] = []
            for g in grads:
                if g is None:
                    per_param_sq.append(0.0)
                else:
                    per_param_sq.append(float(g.detach().pow(2).sum().item()))
            comp_result: dict[str, float] = {}
            scale_abs = abs(float(scale))
            for group_name, indices in groups.items():
                if not indices:
                    continue
                sq_sum = 0.0
                for i in indices:
                    sq_sum += per_param_sq[i]
                norm = sq_sum ** 0.5
                if scale_abs != 1.0:
                    norm *= scale_abs
                comp_result[group_name] = norm
            if comp_result:
                result[cname] = comp_result
        return result if result else None

    @staticmethod
    def _gn_summarize_buffer(
        per_minibatch_norms: list[dict[str, dict[str, float]]],
    ) -> dict:
        """CQ-0284: minibatch ごとの norm dict 列を集計する

        Returns:
            {component: {group: {mean, p50, p90, max, count}}}
        """
        # collect per (component, group) → list[float]
        from collections import defaultdict
        bucket: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list))
        for mb in per_minibatch_norms:
            if not mb:
                continue
            for cname, gd in mb.items():
                for gname, val in gd.items():
                    bucket[cname][gname].append(float(val))

        out: dict[str, dict[str, dict]] = {}
        for cname, gdict in bucket.items():
            out[cname] = {}
            for gname, vals in gdict.items():
                if not vals:
                    out[cname][gname] = {
                        "mean": None, "p50": None, "p90": None,
                        "max": None, "count": 0,
                    }
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                out[cname][gname] = {
                    "mean": float(arr.mean()),
                    "p50": float(np.quantile(arr, 0.5)),
                    "p90": float(np.quantile(arr, 0.9)),
                    "max": float(arr.max()),
                    "count": int(arr.size),
                }
        return out

    @staticmethod
    def _gn_compute_ratios(stats: dict) -> dict:
        """CQ-0284: 係数調整に使う ratio (mean ベース)"""
        def _get_mean(comp: str, group: str = "value_semantic"):
            d = stats.get(comp, {}).get(group, {})
            v = d.get("mean")
            if v is None:
                return None
            return float(v)

        def _ratio(num, den):
            if num is None or den is None or den == 0.0:
                return None
            return float(num / den)

        t = _get_mean("terminal_loss")
        y = _get_mean("yaku_loss")
        wt = _get_mean("weighted_terminal_loss")
        wy = _get_mean("weighted_yaku_loss")
        wv = _get_mean("weighted_value_loss")
        return {
            "value_semantic_terminal_to_yaku": _ratio(t, y),
            "value_semantic_weighted_terminal_to_weighted_yaku":
                _ratio(wt, wy),
            "value_semantic_weighted_terminal_to_weighted_value":
                _ratio(wt, wv),
            "value_semantic_weighted_yaku_to_weighted_value":
                _ratio(wy, wv),
        }

    # ====================================================================
    # CQ-0281: PPO diagnostics helpers
    # ====================================================================

    @staticmethod
    def _safe_np_quantiles(values, quantiles):
        """空配列に安全な quantile 計算

        Args:
            values: 1D ndarray (np.float64 推奨) もしくは None / 空
            quantiles: 0..1 の percentile を 0..1 で

        Returns:
            dict: {f"p{int(q*100):02d}": float | None}
        """
        out: dict = {}
        if values is None or len(values) == 0:
            for q in quantiles:
                out[f"p{int(q * 100):02d}"] = None
            return out
        for q in quantiles:
            out[f"p{int(q * 100):02d}"] = float(np.quantile(values, q))
        return out

    @staticmethod
    def _weighted_mean(values, weights):
        """重み付き平均。空 / weight 合計 0 なら None"""
        if values is None or len(values) == 0:
            return None
        if weights is None:
            return float(np.mean(values))
        ws = float(np.sum(weights))
        if ws <= 0:
            return None
        return float(np.sum(values * weights) / ws)

    @staticmethod
    def _weighted_fraction(mask, weights):
        """重み付き fraction (mask は 0/1 or bool)"""
        if mask is None or len(mask) == 0:
            return None
        m = np.asarray(mask, dtype=np.float64)
        if weights is None:
            return float(m.mean())
        w = np.asarray(weights, dtype=np.float64)
        ws = float(w.sum())
        if ws <= 0:
            return None
        return float((m * w).sum() / ws)

    @classmethod
    def _compute_ppo_diag_stats(
        cls,
        log_ratios: np.ndarray | None,
        advantages: np.ndarray | None,
        max_probs: np.ndarray | None,
        weights: np.ndarray | None,
        clip_epsilon: float,
    ) -> dict:
        """CQ-0281: PPO diagnostics 統計を計算する

        log_ratio / advantage / max_prob から log_ratio quantile / advantage
        sign fraction / cross stats / max_prob quantile などを生成する。

        集計方針:
            - mean / std / fraction: weights があれば weighted、無ければ unweighted
            - quantile: unweighted (重み付き quantile は scope 外)
            - num_adv_pos / num_adv_neg は raw count
            - 空配列や片符号のみのケースでは安全に None を返す
        """
        d: dict = {}
        if log_ratios is not None and len(log_ratios) > 0:
            lr = np.asarray(log_ratios, dtype=np.float64)
            ratio = np.exp(lr)
            d["log_ratio_mean"] = cls._weighted_mean(lr, weights)
            d["log_ratio_std"] = float(np.std(lr))  # unweighted
            d["log_ratio_min"] = float(np.min(lr))
            d["log_ratio_max"] = float(np.max(lr))
            qs = cls._safe_np_quantiles(lr, [0.01, 0.05, 0.50, 0.95, 0.99])
            for k, v in qs.items():
                d[f"log_ratio_{k}"] = v
            qsr = cls._safe_np_quantiles(ratio, [0.01, 0.05, 0.50, 0.95, 0.99])
            for k, v in qsr.items():
                d[f"ratio_{k}"] = v
            d["ratio_max"] = float(np.max(ratio))
            d["clip_fraction"] = cls._weighted_fraction(
                np.abs(ratio - 1.0) > clip_epsilon, weights)

        if advantages is not None and len(advantages) > 0:
            a = np.asarray(advantages, dtype=np.float64)
            pos_mask = a > 0
            neg_mask = a < 0
            zero_mask = a == 0
            d["advantage_pos_frac"] = cls._weighted_fraction(pos_mask, weights)
            d["advantage_neg_frac"] = cls._weighted_fraction(neg_mask, weights)
            d["advantage_zero_frac"] = cls._weighted_fraction(zero_mask, weights)
            d["advantage_abs_mean"] = cls._weighted_mean(np.abs(a), weights)
            qsa = cls._safe_np_quantiles(a, [0.01, 0.05, 0.50, 0.95, 0.99])
            for k, v in qsa.items():
                d[f"advantage_{k}"] = v
            d["num_adv_pos"] = int(pos_mask.sum())
            d["num_adv_neg"] = int(neg_mask.sum())

            # cross stats: advantage × log_ratio
            if log_ratios is not None and len(log_ratios) == len(a):
                lr = np.asarray(log_ratios, dtype=np.float64)
                ratio = np.exp(lr)
                clipped = np.abs(ratio - 1.0) > clip_epsilon
                w_pos = (weights[pos_mask] if (weights is not None
                                                and pos_mask.any()) else None)
                w_neg = (weights[neg_mask] if (weights is not None
                                                and neg_mask.any()) else None)
                if pos_mask.any():
                    d["log_ratio_mean_adv_pos"] = cls._weighted_mean(
                        lr[pos_mask], w_pos)
                    d["ratio_mean_adv_pos"] = cls._weighted_mean(
                        ratio[pos_mask], w_pos)
                    d["clip_fraction_adv_pos"] = cls._weighted_fraction(
                        clipped[pos_mask], w_pos)
                else:
                    d["log_ratio_mean_adv_pos"] = None
                    d["ratio_mean_adv_pos"] = None
                    d["clip_fraction_adv_pos"] = None
                if neg_mask.any():
                    d["log_ratio_mean_adv_neg"] = cls._weighted_mean(
                        lr[neg_mask], w_neg)
                    d["ratio_mean_adv_neg"] = cls._weighted_mean(
                        ratio[neg_mask], w_neg)
                    d["clip_fraction_adv_neg"] = cls._weighted_fraction(
                        clipped[neg_mask], w_neg)
                else:
                    d["log_ratio_mean_adv_neg"] = None
                    d["ratio_mean_adv_neg"] = None
                    d["clip_fraction_adv_neg"] = None

        if max_probs is not None and len(max_probs) > 0:
            mp = np.asarray(max_probs, dtype=np.float64)
            d["max_prob_mean"] = cls._weighted_mean(mp, weights)
            qsm = cls._safe_np_quantiles(mp, [0.50, 0.90, 0.95, 0.99])
            d["max_prob_p50"] = qsm["p50"]
            d["max_prob_p90"] = qsm["p90"]
            d["max_prob_p95"] = qsm["p95"]
            d["max_prob_p99"] = qsm["p99"]
        return d

    def _tk_summarize(self, diag_bufs: list[dict]) -> dict:
        """CQ-0287: target_kl early-stop diagnostics を集計する

        diag_bufs は 1 cycle 中の epoch ごとの buffer の list (discard 側
        または call 側、もしくはその結合)。
        """
        approx_kls: list[float] = []
        skipped = 0
        stop_count = 0
        for b in diag_bufs:
            approx_kls.extend(b.get("tk_approx_kls", []) or [])
            skipped += int(b.get("tk_skipped_minibatches", 0))
            stop_count += int(b.get("tk_stop_count", 0))
        checked = len(approx_kls)
        out: dict = {
            "target_kl_enabled": bool(self._tk_enabled),
            "target_kl": float(self._tk_target),
            "target_kl_threshold": float(self._tk_threshold),
            "target_kl_skip_minibatch_on_exceed": bool(
                self._tk_skip_on_exceed),
            "target_kl_stop_count": int(stop_count),
            "target_kl_skipped_minibatches": int(skipped),
            "target_kl_checked_minibatches": int(checked),
        }
        if checked > 0:
            arr = np.asarray(approx_kls, dtype=np.float64)
            out["approx_kl_mean"] = float(arr.mean())
            out["approx_kl_max"] = float(arr.max())
        else:
            out["approx_kl_mean"] = None
            out["approx_kl_max"] = None
        return out

    def _ppo_discard_epoch(self, samples, returns_t, advantages_t,
                           weights_t=None,
                           terminal_classes_t=None, yaku_multihot_t=None,
                           is_winner_t=None, terminal_weights_t=None,
                           epoch_idx: int = 0):
        """discard PPO の 1 epoch (CQ-0249: weighted mean)"""
        policy_losses, value_losses, entropies = [], [], []
        all_ratios = []
        # CQ-0281: 追加 diagnostics 用バッファ
        all_log_ratios: list[torch.Tensor] = []
        all_batch_adv: list[torch.Tensor] = []
        all_max_probs: list[torch.Tensor] = []
        all_batch_w: list[torch.Tensor] = []
        _anchor_kls: list[float] = []
        _sem_tl: list[float] = []
        _sem_yl: list[float] = []
        # CQ-0284: gradient norm diagnostics buffer (1 entry per measured minibatch)
        gn_minibatches: list[dict] = []
        gn_measured = 0
        # CQ-0287: target_kl early-stop buffers
        tk_approx_kls: list[float] = []
        tk_skipped_minibatches = 0
        tk_stop_count = 0  # この epoch で early stop が発火したか (0 or 1)
        n = len(samples)
        indices = np.random.permutation(n)

        for batch_idx_in_epoch, start in enumerate(range(0, n, self._batch_size)):
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

            log_ratio = action_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self._clip_epsilon,
                                1 + self._clip_epsilon) * batch_advantages
            surr_min = -torch.min(surr1, surr2)
            policy_loss = (surr_min * batch_w).sum() / w_sum

            # CQ-0281: max_prob = legal mask 適用後 softmax の sample-wise max
            with torch.no_grad():
                masked_logits = out.discard_logits + (1.0 - masks) * (-1e9)
                masked_probs = torch.softmax(masked_logits, dim=-1)
                max_prob = masked_probs.max(dim=-1).values

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
            sa_loss_t = None
            tl_t = None
            yl_t = None
            if self._semantic_aux_enabled and terminal_classes_t is not None:
                tw = terminal_weights_t[idx_t] if terminal_weights_t is not None else None
                sa_loss, tl_t, yl_t = self._compute_semantic_aux_loss(
                    out.semantic, terminal_classes_t[idx_t],
                    yaku_multihot_t[idx_t], is_winner_t[idx_t],
                    terminal_weights=tw)
                loss = loss + sa_loss
                sa_loss_t = sa_loss
                _sem_tl.append(float(tl_t.detach()))
                _sem_yl.append(float(yl_t.detach()))

            # CQ-0284: gradient norm diagnostics (default off)
            # ``loss.backward()`` の前に ``torch.autograd.grad`` で計測。
            # ``retain_graph=True`` を使うので後段 backward は通常通り動く。
            if self._gn_should_measure(
                    batch_idx_in_epoch=gn_measured, epoch_idx=epoch_idx):
                gn_norms = self._gn_compute_minibatch_norms(
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    sa_loss_total=sa_loss_t,
                    terminal_loss_t=tl_t,
                    yaku_loss_t=yl_t,
                    total_loss=loss,
                )
                if gn_norms is not None:
                    gn_minibatches.append(gn_norms)
                    gn_measured += 1

            # CQ-0287: target_kl early stop. 学習挙動を変えないように、
            # default off では「approx_kl の記録のみ」、enabled かつ
            # threshold 超過のときだけ skip / break する。
            with torch.no_grad():
                approx_kl = float(
                    ((ratio.detach() - 1.0) - log_ratio.detach()).mean().item())
            tk_approx_kls.append(approx_kl)
            if self._tk_enabled and approx_kl > self._tk_threshold:
                tk_stop_count = 1  # この epoch は early-stop した
                if self._tk_skip_on_exceed:
                    # backward / optimizer step を呼ばずに break
                    tk_skipped_minibatches += 1
                    # diagnostics 用 tensor は記録 (skip 前 forward の結果)
                    all_ratios.append(ratio.detach().cpu())
                    all_log_ratios.append(log_ratio.detach().cpu())
                    all_batch_adv.append(batch_advantages.detach().cpu())
                    all_max_probs.append(max_prob.detach().cpu())
                    all_batch_w.append(batch_w.detach().cpu())
                    break
                # skip_on_exceed=False: 通常 step してから break する
                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(),
                                          self._max_grad_norm)
                self._optimizer.step()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                all_ratios.append(ratio.detach().cpu())
                all_log_ratios.append(log_ratio.detach().cpu())
                all_batch_adv.append(batch_advantages.detach().cpu())
                all_max_probs.append(max_prob.detach().cpu())
                all_batch_w.append(batch_w.detach().cpu())
                if self._anchor_enabled and self._anchor_model is not None:
                    _anchor_kls.append(akl_val)
                break

            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            # CQ-0281: detach + cpu で集計用に回収 (GPU tensor を長く保持しない)
            all_ratios.append(ratio.detach().cpu())
            all_log_ratios.append(log_ratio.detach().cpu())
            all_batch_adv.append(batch_advantages.detach().cpu())
            all_max_probs.append(max_prob.detach().cpu())
            all_batch_w.append(batch_w.detach().cpu())
            if self._anchor_enabled and self._anchor_model is not None:
                _anchor_kls.append(akl_val)

        diag_buffer = {
            "log_ratios": all_log_ratios,
            "advantages": all_batch_adv,
            "max_probs": all_max_probs,
            "weights": all_batch_w,
            # CQ-0284
            "gradient_norms": gn_minibatches,
            # CQ-0287
            "tk_approx_kls": tk_approx_kls,
            "tk_skipped_minibatches": tk_skipped_minibatches,
            "tk_stop_count": tk_stop_count,
        }
        return (policy_losses, value_losses, entropies, all_ratios,
                _anchor_kls, _sem_tl, _sem_yl, diag_buffer)

    def _ppo_call_epoch(self, samples, returns_t, advantages_t,
                        weights_t=None,
                        terminal_classes_t=None, yaku_multihot_t=None,
                        is_winner_t=None, terminal_weights_t=None,
                        epoch_idx: int = 0):
        """optional PPO の 1 epoch (CQ-0249: weighted mean)"""
        policy_losses, value_losses, entropies = [], [], []
        all_ratios = []
        # CQ-0281: 追加 diagnostics 用バッファ
        all_log_ratios: list[torch.Tensor] = []
        all_batch_adv: list[torch.Tensor] = []
        all_max_probs: list[torch.Tensor] = []
        all_batch_w: list[torch.Tensor] = []
        _anchor_kls: list[float] = []
        _sem_tl: list[float] = []
        _sem_yl: list[float] = []
        # CQ-0284: gradient norm diagnostics buffer
        gn_minibatches: list[dict] = []
        gn_measured = 0
        # CQ-0287: target_kl early-stop buffers
        tk_approx_kls: list[float] = []
        tk_skipped_minibatches = 0
        tk_stop_count = 0
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

            log_ratio = action_log_probs - old_log_probs
            ratio = torch.exp(log_ratio)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self._clip_epsilon,
                                1 + self._clip_epsilon) * batch_advantages
            surr_min = -torch.min(surr1, surr2)
            policy_loss = (surr_min * batch_w).sum() / w_sum

            # CQ-0281: max_prob = candidate mask 適用後 softmax の sample-wise max
            with torch.no_grad():
                masked_scores = out.optional_scores + (1.0 - cand_mask) * (-1e9)
                masked_probs = torch.softmax(masked_scores, dim=-1)
                max_prob = masked_probs.max(dim=-1).values

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
            sa_loss_t = None
            tl_t = None
            yl_t = None
            if self._semantic_aux_enabled and terminal_classes_t is not None:
                tw = terminal_weights_t[idx_t] if terminal_weights_t is not None else None
                sa_loss, tl_t, yl_t = self._compute_semantic_aux_loss(
                    out.semantic, terminal_classes_t[idx_t],
                    yaku_multihot_t[idx_t], is_winner_t[idx_t],
                    terminal_weights=tw)
                loss = loss + sa_loss
                sa_loss_t = sa_loss
                _sem_tl.append(float(tl_t.detach()))
                _sem_yl.append(float(yl_t.detach()))

            # CQ-0284: gradient norm diagnostics (default off)
            if self._gn_should_measure(
                    batch_idx_in_epoch=gn_measured, epoch_idx=epoch_idx):
                gn_norms = self._gn_compute_minibatch_norms(
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    sa_loss_total=sa_loss_t,
                    terminal_loss_t=tl_t,
                    yaku_loss_t=yl_t,
                    total_loss=loss,
                )
                if gn_norms is not None:
                    gn_minibatches.append(gn_norms)
                    gn_measured += 1

            # CQ-0287: target_kl early stop (call branch)
            with torch.no_grad():
                approx_kl = float(
                    ((ratio.detach() - 1.0) - log_ratio.detach()).mean().item())
            tk_approx_kls.append(approx_kl)
            if self._tk_enabled and approx_kl > self._tk_threshold:
                tk_stop_count = 1
                if self._tk_skip_on_exceed:
                    tk_skipped_minibatches += 1
                    all_ratios.append(ratio.detach().cpu())
                    all_log_ratios.append(log_ratio.detach().cpu())
                    all_batch_adv.append(batch_advantages.detach().cpu())
                    all_max_probs.append(max_prob.detach().cpu())
                    all_batch_w.append(batch_w.detach().cpu())
                    break
                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(),
                                          self._max_grad_norm)
                self._optimizer.step()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                all_ratios.append(ratio.detach().cpu())
                all_log_ratios.append(log_ratio.detach().cpu())
                all_batch_adv.append(batch_advantages.detach().cpu())
                all_max_probs.append(max_prob.detach().cpu())
                all_batch_w.append(batch_w.detach().cpu())
                if self._anchor_enabled and self._anchor_model is not None:
                    _anchor_kls.append(akl_val)
                break

            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            # CQ-0281: detach + cpu で集計用に回収
            all_ratios.append(ratio.detach().cpu())
            all_log_ratios.append(log_ratio.detach().cpu())
            all_batch_adv.append(batch_advantages.detach().cpu())
            all_max_probs.append(max_prob.detach().cpu())
            all_batch_w.append(batch_w.detach().cpu())
            if self._anchor_enabled and self._anchor_model is not None:
                _anchor_kls.append(akl_val)

        diag_buffer = {
            "log_ratios": all_log_ratios,
            "advantages": all_batch_adv,
            "max_probs": all_max_probs,
            "weights": all_batch_w,
            # CQ-0284
            "gradient_norms": gn_minibatches,
            # CQ-0287
            "tk_approx_kls": tk_approx_kls,
            "tk_skipped_minibatches": tk_skipped_minibatches,
            "tk_stop_count": tk_stop_count,
        }
        return (policy_losses, value_losses, entropies, all_ratios,
                _anchor_kls, _sem_tl, _sem_yl, diag_buffer)
