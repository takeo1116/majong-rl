"""CQ-0286: Stage2a optimizer parameter groups (policy / value_semantic) lr 分離

確認:
- default off (config 未指定 / enabled=false) で既存挙動と完全互換
  (single optimizer group, lr=training.lr)
- enabled=true で policy / value_semantic の 2 group が作られる
- group ごとに正しい lr が設定される
- module 振り分けが Stage2aModel の構造と一致する
- trainable parameter の重複/取りこぼしがない (id ベース)
- unknown parameter は default group に流れる (現 model では空)
- summary に lr_groups 情報が出力される
- gradient norm diagnostics と併用しても crash しない
- small PPO smoke で完走
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.smoke

from mahjong_rl.stage2a_learner import Stage2aLearner
from mahjong_rl.models.stage2a_model import Stage2aModel
from mahjong_rl.call_shard import (
    DecisionSample, CandidateRecord, DecisionShardWriter,
)


# ========== helpers ==========


def _make_model(*, semantic_aux: bool = False, direct_hint: bool = False):
    sa_cfg = {"enabled": True, "policy_projection_dim": 4} if semantic_aux else None
    direct_hint_ranges = (
        {"shanten_hint": (10, 14), "discard_ukeire_hint": (14, 18)}
        if direct_hint else None)
    return Stage2aModel(
        input_dim=50, discard_hidden_dims=[16],
        optional_hidden_dims=[16], value_hidden_dims=[16],
        candidate_dim=8, optional_scorer_hidden=8,
        semantic_aux_config=sa_cfg,
        direct_hint_ranges=direct_hint_ranges,
    )


def _make_learner(tmp_path, *, lr_groups: dict | None = None,
                   semantic_aux: bool = False, direct_hint: bool = False,
                   base_lr: float = 1e-4):
    cfg: dict = {
        "training": {
            "algorithm": "ppo", "epochs": 1, "batch_size": 4,
            "lr": base_lr,
        },
    }
    if semantic_aux:
        cfg["training"]["semantic_aux"] = {
            "enabled": True, "terminal_loss_coef": 0.1, "yaku_loss_coef": 0.05,
        }
    if lr_groups is not None:
        cfg["training"]["lr_groups"] = lr_groups
    model = _make_model(semantic_aux=semantic_aux, direct_hint=direct_hint)
    return Stage2aLearner(
        config=cfg, model=model, run_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )


def _mk_discard(step_id, version=3, **kw):
    base = dict(
        decision_type="discard",
        observation=np.zeros(50, dtype=np.float32),
        legal_mask=np.ones(34, dtype=np.float32),
        action=0, reward=0.0, log_prob=-0.5, value=0.0,
        terminated=False, round_over=False,
        player_id=0, episode_id="ep0", round_id=0,
        step_id=step_id, actor_type="policy",
        experiment_id="t", run_id="r", worker_id="w",
    )
    base.update(kw)
    s = DecisionSample(**base)
    s.sample_semantics_version = version
    return s


def _mk_call(step_id, version=3, **kw):
    base = dict(
        decision_type="call",
        observation=np.zeros(50, dtype=np.float32),
        reward=0.0, log_prob=-0.5, value=0.0,
        terminated=False, round_over=False,
        selected_candidate_index=0, candidate_count=2,
        candidates=[CandidateRecord(action_type=8),
                    CandidateRecord(action_type=4, tile_type=10,
                                     target_rel_seat=2)],
        player_id=0, episode_id="ep0", round_id=0,
        step_id=step_id, actor_type="policy",
        experiment_id="t", run_id="r", worker_id="w",
    )
    base.update(kw)
    s = DecisionSample(**base)
    s.sample_semantics_version = version
    return s


def _write_smoke_shard(shard_dir, n_d=8, n_c=6):
    writer = DecisionShardWriter(shard_dir, max_samples=100)
    rng = np.random.RandomState(42)
    for i in range(n_d):
        writer.add(_mk_discard(
            step_id=i,
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_d - 1),
        ))
    for i in range(n_c):
        writer.add(_mk_call(
            step_id=n_d + i,
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_c - 1),
        ))
    writer.close()


# ========== 1. default off compatibility ==========


class TestDefaultOffCompat:
    """`training.lr_groups` 未指定または enabled=false で既存挙動と互換"""

    def test_no_lr_groups_key_means_single_group(self, tmp_path):
        """training.lr_groups 自体が無いとき: optimizer は single param group"""
        learner = _make_learner(tmp_path, lr_groups=None, base_lr=1.5e-4)
        assert len(learner._optimizer.param_groups) == 1
        g = learner._optimizer.param_groups[0]
        assert g["lr"] == pytest.approx(1.5e-4)

    def test_enabled_false_means_single_group(self, tmp_path):
        """lr_groups.enabled=false で single group"""
        learner = _make_learner(
            tmp_path, lr_groups={"enabled": False, "policy": 9.9,
                                  "value_semantic": 9.9},
            base_lr=2e-4)
        assert len(learner._optimizer.param_groups) == 1
        g = learner._optimizer.param_groups[0]
        assert g["lr"] == pytest.approx(2e-4)
        # 同 group に全 trainable param が入っている
        ids = {id(p) for p in g["params"]}
        expected = {id(p) for p in learner._model.parameters() if p.requires_grad}
        assert ids == expected

    def test_disabled_lr_groups_info(self, tmp_path):
        """default off の info は最小限 ('all' のみ)"""
        learner = _make_learner(tmp_path, lr_groups=None, base_lr=1e-4)
        info = learner._lr_groups_info
        assert info["enabled"] is False
        assert "all" in info["groups"]
        assert info["groups"]["all"]["lr"] == pytest.approx(1e-4)

    def test_metrics_carry_lr_groups_info(self, tmp_path):
        """default off でも metrics に optimizer_lr_groups が出る"""
        shard_dir = tmp_path / "shards"
        _write_smoke_shard(shard_dir)
        learner = _make_learner(tmp_path)
        metrics = learner.train(shard_dir)
        assert "optimizer_lr_groups" in metrics
        assert metrics["optimizer_lr_groups"]["enabled"] is False


# ========== 2. enabled creates policy / value_semantic groups ==========


class TestEnabledCreatesGroups:
    """enabled=true で policy / value_semantic の 2 group が作られる"""

    def test_two_groups_with_correct_lr(self, tmp_path):
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True,
                       "policy": 1e-4,
                       "value_semantic": 3e-4},
            semantic_aux=True)
        groups = learner._optimizer.param_groups
        # policy + value_semantic は両方 non-empty
        names = sorted([g.get("name") for g in groups
                        if g.get("name") is not None])
        assert "policy" in names
        assert "value_semantic" in names
        # default は param 0 件のはず → optimizer に追加されない
        assert "default" not in names
        # 各 group の lr を確認
        by_name = {g["name"]: g for g in groups if g.get("name")}
        assert by_name["policy"]["lr"] == pytest.approx(1e-4)
        assert by_name["value_semantic"]["lr"] == pytest.approx(3e-4)

    def test_lr_groups_info_records_param_counts(self, tmp_path):
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True,
                       "policy": 5e-5,
                       "value_semantic": 5e-4},
            semantic_aux=True)
        info = learner._lr_groups_info
        assert info["enabled"] is True
        g = info["groups"]
        assert g["policy"]["lr"] == pytest.approx(5e-5)
        assert g["value_semantic"]["lr"] == pytest.approx(5e-4)
        assert g["policy"]["param_count"] > 0
        assert g["value_semantic"]["param_count"] > 0
        assert g["policy"]["tensor_count"] > 0
        assert g["value_semantic"]["tensor_count"] > 0

    def test_default_lr_falls_back_to_training_lr(self, tmp_path):
        """policy / value_semantic 未指定なら training.lr を使う"""
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True},
            base_lr=7e-5)
        groups = learner._optimizer.param_groups
        for g in groups:
            assert g["lr"] == pytest.approx(7e-5)


# ========== 3. module assignment ==========


class TestModuleAssignment:
    """policy / value_semantic への振り分けが Stage2aModel の構造に一致"""

    def _build_id_sets(self, learner):
        groups = learner._optimizer.param_groups
        by_name = {g["name"]: g for g in groups if g.get("name")}
        return {
            name: {id(p) for p in g["params"]}
            for name, g in by_name.items()
        }

    def test_policy_modules_in_policy_group(self, tmp_path):
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True, "policy": 1e-4,
                       "value_semantic": 1e-3},
            semantic_aux=True, direct_hint=False)
        id_sets = self._build_id_sets(learner)
        policy_ids = id_sets["policy"]
        m = learner._model
        # discard_trunk / discard_head / optional_trunk / candidate_encoder /
        # optional_scorer の params がすべて policy group に
        for mod_name in ("discard_trunk", "discard_head", "optional_trunk",
                         "candidate_encoder", "optional_scorer"):
            mod = getattr(m, mod_name)
            for p in mod.parameters():
                assert id(p) in policy_ids, (
                    f"{mod_name} の param が policy group に入っていない")

    def test_value_semantic_modules_in_value_semantic_group(self, tmp_path):
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True, "policy": 1e-4,
                       "value_semantic": 1e-3},
            semantic_aux=True)
        id_sets = self._build_id_sets(learner)
        vs_ids = id_sets["value_semantic"]
        m = learner._model
        # CQ-0288: semantic_proj は削除済み
        for mod_name in ("value_trunk", "value_head",
                         "terminal_head", "yaku_head"):
            mod = getattr(m, mod_name)
            for p in mod.parameters():
                assert id(p) in vs_ids, (
                    f"{mod_name} の param が value_semantic group に入っていない")
        # 削除確認
        assert not hasattr(m, "semantic_proj"), (
            "CQ-0288 で semantic_proj は削除されたはず")

    def test_direct_hint_modules_in_policy_group(self, tmp_path):
        """direct hint 有効時、_tile_embedding / _local_scorer / _context_gate
        が policy group に入る"""
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True, "policy": 1e-4,
                       "value_semantic": 1e-3},
            semantic_aux=False, direct_hint=True)
        id_sets = self._build_id_sets(learner)
        policy_ids = id_sets["policy"]
        m = learner._model
        for mod_name in ("_tile_embedding", "_local_scorer", "_context_gate"):
            mod = getattr(m, mod_name)
            for p in mod.parameters():
                assert id(p) in policy_ids, (
                    f"{mod_name} (direct hint) の param が policy group に "
                    f"入っていない")


# ========== 4. no duplicate / no missing ==========


class TestNoDuplicateNoMissing:
    """trainable params の id 集合と optimizer params の id 集合が一致"""

    def test_id_sets_match_no_duplicate(self, tmp_path):
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True, "policy": 1e-4,
                       "value_semantic": 1e-3},
            semantic_aux=True, direct_hint=True)
        # 期待 set: model の trainable param 全部
        expected_ids = {id(p) for p in learner._model.parameters()
                        if p.requires_grad}
        # optimizer に積まれた id (重複はあとで別途チェック)
        opt_ids: list[int] = []
        for g in learner._optimizer.param_groups:
            for p in g["params"]:
                opt_ids.append(id(p))
        # 重複がない
        assert len(opt_ids) == len(set(opt_ids)), (
            "optimizer に同 parameter が複数 group に入っている")
        # 集合一致
        assert set(opt_ids) == expected_ids, (
            "optimizer params と model trainable params が一致しない")

    def test_default_off_id_set_matches(self, tmp_path):
        """default off 経路でも同様"""
        learner = _make_learner(
            tmp_path, lr_groups=None, semantic_aux=True, direct_hint=True)
        expected_ids = {id(p) for p in learner._model.parameters()
                        if p.requires_grad}
        opt_ids = [id(p) for g in learner._optimizer.param_groups
                   for p in g["params"]]
        assert len(opt_ids) == len(set(opt_ids))
        assert set(opt_ids) == expected_ids

    def test_classifier_alone_partitions_strictly(self, tmp_path):
        """_classify_param_groups の出力が排他かつ網羅"""
        m = _make_model(semantic_aux=True, direct_hint=True)
        named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        out = Stage2aLearner._classify_param_groups(named)
        # union == named, pairwise disjoint
        all_ids: list[int] = []
        for k in ("policy", "value_semantic", "default"):
            all_ids.extend(id(p) for _, p in out[k])
        assert len(all_ids) == len(set(all_ids))  # disjoint
        expected = {id(p) for _, p in named}
        assert set(all_ids) == expected


# ========== 5. unknown / default behavior ==========


class TestUnknownDefaultBehavior:
    """現 Stage2aModel では unknown parameter なし"""

    @pytest.mark.parametrize("direct_hint,semantic_aux", [
        (False, False), (True, False), (False, True), (True, True),
    ])
    def test_no_unknown_params_in_current_model(self, direct_hint, semantic_aux):
        m = _make_model(semantic_aux=semantic_aux, direct_hint=direct_hint)
        named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        out = Stage2aLearner._classify_param_groups(named)
        assert out["default"] == [], (
            f"unknown parameter が見つかった: "
            f"{[n for n, _ in out['default']]}")

    def test_default_group_handles_unknown_synthetic(self):
        """合成テスト: 名前が unknown 始まりの param が default に入る"""
        from torch.nn import Parameter
        named = [
            ("discard_trunk.0.weight", Parameter(torch.zeros(2))),
            ("value_trunk.0.weight", Parameter(torch.zeros(2))),
            ("strange_module.weight", Parameter(torch.zeros(2))),
        ]
        out = Stage2aLearner._classify_param_groups(named)
        assert len(out["policy"]) == 1
        assert len(out["value_semantic"]) == 1
        assert len(out["default"]) == 1
        assert out["default"][0][0] == "strange_module.weight"


# ========== 6. PPO smoke ==========


class TestPpoSmoke:
    """small PPO smoke で enabled=true が完走 + diagnostics 併用"""

    def test_ppo_smoke_with_lr_groups(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_smoke_shard(shard_dir)
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True, "policy": 1e-4,
                       "value_semantic": 5e-4},
            semantic_aux=True)
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        assert metrics["num_updates"] > 0
        info = metrics["optimizer_lr_groups"]
        assert info["enabled"] is True
        assert info["groups"]["policy"]["lr"] == pytest.approx(1e-4)
        assert info["groups"]["value_semantic"]["lr"] == pytest.approx(5e-4)

    def test_ppo_smoke_with_lr_groups_and_gradient_norms(self, tmp_path):
        """gradient norm diagnostics と併用しても crash しない"""
        shard_dir = tmp_path / "shards"
        _write_smoke_shard(shard_dir)
        model = _make_model(semantic_aux=True)
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "ppo", "epochs": 1, "batch_size": 4,
                "lr": 1e-4,
                "lr_groups": {
                    "enabled": True,
                    "policy": 1e-4,
                    "value_semantic": 5e-4,
                },
                "semantic_aux": {
                    "enabled": True,
                    "terminal_loss_coef": 0.1, "yaku_loss_coef": 0.05,
                },
                "diagnostics": {"gradient_norms": {
                    "enabled": True, "max_batches_per_epoch": 2,
                }},
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        assert "gradient_norms" in metrics["ppo_diag"]
        assert metrics["optimizer_lr_groups"]["enabled"] is True

    def test_param_changes_obey_group_lr(self, tmp_path):
        """policy lr=0 / value_semantic lr=1e-2 で 1 step 後、
        policy params は不変、value_semantic params は変化"""
        shard_dir = tmp_path / "shards"
        _write_smoke_shard(shard_dir)
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True, "policy": 0.0,
                       "value_semantic": 1e-2},
            semantic_aux=True)
        # snapshot
        m = learner._model
        before_policy = {n: p.detach().clone()
                         for n, p in m.discard_trunk.named_parameters()}
        before_vs = {n: p.detach().clone()
                     for n, p in m.value_trunk.named_parameters()}
        learner.train(shard_dir)
        # policy 群は不変
        for n, p in m.discard_trunk.named_parameters():
            diff = (p.detach() - before_policy[n]).abs().max().item()
            assert diff == 0.0, (
                f"policy param {n} が lr=0 にも関わらず更新された "
                f"(max abs diff={diff})")
        # value_semantic 群はどこか変化
        any_changed = False
        for n, p in m.value_trunk.named_parameters():
            diff = (p.detach() - before_vs[n]).abs().max().item()
            if diff > 0.0:
                any_changed = True
                break
        assert any_changed, (
            "value_semantic 群 (value_trunk) が lr>0 にも関わらず "
            "1 sample も更新されなかった")


# ========== 7. CQ-0289: apply_to scope ==========


def _make_imitation_learner(tmp_path, *, lr_groups: dict | None = None,
                              semantic_aux: bool = False,
                              base_lr: float = 1e-4):
    """imitation algorithm の Stage2aLearner"""
    cfg: dict = {
        "training": {
            "algorithm": "imitation", "epochs": 1, "batch_size": 4,
            "lr": base_lr,
        },
    }
    if semantic_aux:
        cfg["training"]["semantic_aux"] = {
            "enabled": True, "terminal_loss_coef": 0.1, "yaku_loss_coef": 0.05,
        }
    if lr_groups is not None:
        cfg["training"]["lr_groups"] = lr_groups
    model = _make_model(semantic_aux=semantic_aux)
    return Stage2aLearner(
        config=cfg, model=model, run_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )


def _write_imitation_shard(shard_dir, n_d=8, n_c=4):
    """imitation 用 shard (teacher_top1_index 付き)"""
    writer = DecisionShardWriter(shard_dir, max_samples=100)
    rng = np.random.RandomState(42)
    for i in range(n_d):
        s = _mk_discard(
            step_id=i,
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_d - 1),
            teacher_top1_index=i % 34,
            teacher_source="rule_based",
        )
        writer.add(s)
    for i in range(n_c):
        s = _mk_call(
            step_id=n_d + i,
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_c - 1),
            teacher_top1_index=0,
            teacher_source="rule_based",
        )
        writer.add(s)
    writer.close()


class TestApplyToDefault:
    """default apply_to (= ["ppo", "imitation"]) で PPO/imitation とも active"""

    def test_default_apply_to_active_for_ppo(self, tmp_path):
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True, "policy": 1e-4,
                       "value_semantic": 1e-3})
        info = learner._lr_groups_info
        assert info["enabled"] is True
        assert info["requested_enabled"] is True
        assert info["active_for_algorithm"] is True
        assert info["apply_to"] == ["ppo", "imitation"]
        assert info["algorithm"] == "ppo"
        # 2 group が作られる
        names = sorted([g.get("name") for g in learner._optimizer.param_groups
                        if g.get("name")])
        assert "policy" in names
        assert "value_semantic" in names

    def test_default_apply_to_active_for_imitation(self, tmp_path):
        learner = _make_imitation_learner(
            tmp_path,
            lr_groups={"enabled": True, "policy": 1e-4,
                       "value_semantic": 1e-3})
        info = learner._lr_groups_info
        assert info["enabled"] is True
        assert info["active_for_algorithm"] is True
        assert info["algorithm"] == "imitation"
        names = sorted([g.get("name") for g in learner._optimizer.param_groups
                        if g.get("name")])
        assert "policy" in names
        assert "value_semantic" in names


class TestApplyToPpoOnly:
    """apply_to=['ppo'] で PPO は active、imitation は inactive"""

    def test_ppo_only_active_for_ppo(self, tmp_path):
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True, "apply_to": ["ppo"],
                       "policy": 1e-4, "value_semantic": 1e-3})
        info = learner._lr_groups_info
        assert info["requested_enabled"] is True
        assert info["active_for_algorithm"] is True
        assert info["enabled"] is True
        assert info["apply_to"] == ["ppo"]
        assert info["algorithm"] == "ppo"
        # 2 group
        assert len(learner._optimizer.param_groups) >= 2

    def test_ppo_only_inactive_for_imitation(self, tmp_path):
        learner = _make_imitation_learner(
            tmp_path,
            lr_groups={"enabled": True, "apply_to": ["ppo"],
                       "policy": 1e-4, "value_semantic": 1e-3})
        info = learner._lr_groups_info
        assert info["requested_enabled"] is True
        # imitation では active_for_algorithm = False
        assert info["active_for_algorithm"] is False
        assert info["enabled"] is False
        assert info["apply_to"] == ["ppo"]
        assert info["algorithm"] == "imitation"
        # single group
        assert len(learner._optimizer.param_groups) == 1
        assert "all" in info["groups"]


class TestApplyToImitationOnly:
    """apply_to=['imitation'] で imitation は active、PPO は inactive"""

    def test_imitation_only_active_for_imitation(self, tmp_path):
        learner = _make_imitation_learner(
            tmp_path,
            lr_groups={"enabled": True, "apply_to": ["imitation"],
                       "policy": 1e-4, "value_semantic": 1e-3})
        info = learner._lr_groups_info
        assert info["active_for_algorithm"] is True
        assert info["enabled"] is True
        assert len(learner._optimizer.param_groups) >= 2

    def test_imitation_only_inactive_for_ppo(self, tmp_path):
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True, "apply_to": ["imitation"],
                       "policy": 1e-4, "value_semantic": 1e-3})
        info = learner._lr_groups_info
        assert info["active_for_algorithm"] is False
        assert info["enabled"] is False
        # single group
        assert len(learner._optimizer.param_groups) == 1


class TestApplyToValidation:
    """apply_to の validation"""

    def test_empty_list_raises(self, tmp_path):
        with pytest.raises(ValueError, match="空 list"):
            _make_learner(
                tmp_path,
                lr_groups={"enabled": True, "apply_to": [],
                           "policy": 1e-4, "value_semantic": 1e-3})

    def test_unknown_algorithm_raises(self, tmp_path):
        with pytest.raises(ValueError, match="未知の algorithm"):
            _make_learner(
                tmp_path,
                lr_groups={"enabled": True,
                           "apply_to": ["ppo", "rainbow"],
                           "policy": 1e-4, "value_semantic": 1e-3})

    def test_non_list_raises(self, tmp_path):
        with pytest.raises(ValueError, match="list である必要"):
            _make_learner(
                tmp_path,
                lr_groups={"enabled": True, "apply_to": "ppo",
                           "policy": 1e-4, "value_semantic": 1e-3})

    def test_validation_only_when_enabled(self, tmp_path):
        """enabled=False でも apply_to の validation 自体は走る
        (config 上の typo を早期検出する目的)。
        ただしこの validation は構築段階で行うので、enabled=False で
        apply_to が valid なら crash しない。"""
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": False, "apply_to": ["ppo"]})
        info = learner._lr_groups_info
        # enabled=False のとき active_for_algorithm も False
        assert info["enabled"] is False
        assert info["active_for_algorithm"] is False
        assert info["requested_enabled"] is False


class TestApplyToDiagnosticsSchema:
    """optimizer_lr_groups diagnostics の新キーの schema"""

    def test_diagnostics_schema_active(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_smoke_shard(shard_dir)
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True, "apply_to": ["ppo"],
                       "policy": 1e-4, "value_semantic": 5e-4})
        metrics = learner.train(shard_dir)
        info = metrics["optimizer_lr_groups"]
        for k in ("enabled", "requested_enabled",
                  "active_for_algorithm", "apply_to", "algorithm",
                  "groups"):
            assert k in info
        assert info["requested_enabled"] is True
        assert info["active_for_algorithm"] is True
        assert info["algorithm"] == "ppo"

    def test_diagnostics_schema_inactive(self, tmp_path):
        """inactive (apply_to に algorithm が含まれない) でも new keys が出る"""
        shard_dir = tmp_path / "shards"
        _write_imitation_shard(shard_dir)
        learner = _make_imitation_learner(
            tmp_path,
            lr_groups={"enabled": True, "apply_to": ["ppo"],
                       "policy": 1e-4, "value_semantic": 5e-4})
        metrics = learner.train(shard_dir)
        info = metrics["optimizer_lr_groups"]
        assert info["requested_enabled"] is True
        assert info["active_for_algorithm"] is False
        assert info["enabled"] is False
        assert info["algorithm"] == "imitation"
        assert info["apply_to"] == ["ppo"]
        # single group
        assert "all" in info["groups"]

    def test_json_serializable(self, tmp_path):
        import json
        shard_dir = tmp_path / "shards"
        _write_smoke_shard(shard_dir)
        learner = _make_learner(
            tmp_path,
            lr_groups={"enabled": True, "apply_to": ["ppo"],
                       "policy": 1e-4, "value_semantic": 5e-4})
        metrics = learner.train(shard_dir)
        # crash しないこと
        json.dumps(metrics["optimizer_lr_groups"])


class TestApplyToSmoke:
    """target_kl + lr_groups apply_to ppo / imitation + apply_to ppo"""

    def test_ppo_smoke_target_kl_apply_to_ppo(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_smoke_shard(shard_dir)
        model = _make_model(semantic_aux=True)
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "ppo", "epochs": 1, "batch_size": 4,
                "lr": 1e-4,
                "lr_groups": {
                    "enabled": True,
                    "apply_to": ["ppo"],
                    "policy": 5e-4,
                    "value_semantic": 1e-2,
                },
                "ppo_target_kl": {
                    "enabled": True,
                    "target": 0.03,
                    "stop_multiplier": 1.5,
                    "skip_minibatch_on_exceed": True,
                },
                "semantic_aux": {
                    "enabled": True,
                    "terminal_loss_coef": 0.1, "yaku_loss_coef": 0.05,
                },
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        info = metrics["optimizer_lr_groups"]
        assert info["active_for_algorithm"] is True
        assert info["algorithm"] == "ppo"
        # target_kl も diagnostics に出ている
        assert "target_kl_enabled" in metrics["ppo_diag"]

    def test_imitation_smoke_apply_to_ppo_uses_single_group(self, tmp_path):
        """imitation 学習時、apply_to=['ppo'] では single group で走る"""
        shard_dir = tmp_path / "shards"
        _write_imitation_shard(shard_dir)
        learner = _make_imitation_learner(
            tmp_path,
            lr_groups={"enabled": True, "apply_to": ["ppo"],
                       "policy": 5e-4, "value_semantic": 1e-2},
            semantic_aux=False, base_lr=1e-4)
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "imitation"
        info = metrics["optimizer_lr_groups"]
        assert info["active_for_algorithm"] is False
        # optimizer が single group であること
        assert len(learner._optimizer.param_groups) == 1
        # single group の lr は base_lr
        assert learner._optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
