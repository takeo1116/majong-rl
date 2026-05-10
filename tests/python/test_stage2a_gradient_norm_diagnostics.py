"""CQ-0284: Stage2a PPO gradient norm diagnostics

確認:
- default off で `torch.autograd.grad` が呼ばれず、追加 gradient 計算が走らない
- enabled=true smoke で `ppo_diag.gradient_norms.aggregate` が出る
- component / group の schema (mean/p50/p90/max/count) が揃う
- ratio key が出る
- semantic_aux 無効でも crash せず、terminal/yaku 系は欠落する
- diagnostics 計測で optimizer step 用の .grad が汚染されない
"""
from __future__ import annotations

import json
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


# ========== fixtures / helpers ==========


def _mk_discard(step_id, version=3, **kw):
    base = dict(
        decision_type="discard",
        observation=np.zeros(10, dtype=np.float32),
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
        observation=np.zeros(10, dtype=np.float32),
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


def _make_model(*, semantic_aux: bool = False):
    sa_cfg = {"enabled": True, "policy_projection_dim": 8} if semantic_aux else None
    return Stage2aModel(
        input_dim=10, discard_hidden_dims=[8], optional_hidden_dims=[8],
        value_hidden_dims=[8], candidate_dim=8, optional_scorer_hidden=8,
        semantic_aux_config=sa_cfg,
    )


def _make_learner(tmp_path, *, gn_enabled: bool, semantic_aux: bool = False,
                   max_batches_per_epoch: int = 4, every_n_epochs: int = 1,
                   epochs: int = 1, batch_size: int = 4):
    model = _make_model(semantic_aux=semantic_aux)
    return Stage2aLearner(
        config={"training": {
            "algorithm": "ppo", "epochs": epochs, "batch_size": batch_size,
            "value_loss_coef": 0.125,
            "semantic_aux": ({
                "enabled": True,
                "terminal_loss_coef": 0.1,
                "yaku_loss_coef": 0.05,
            } if semantic_aux else {"enabled": False}),
            "diagnostics": {
                "gradient_norms": {
                    "enabled": gn_enabled,
                    "max_batches_per_epoch": max_batches_per_epoch,
                    "every_n_epochs": every_n_epochs,
                },
            },
        }},
        model=model, run_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )


def _write_mixed_shard(shard_dir, n_d=8, n_c=6, with_winners=True):
    writer = DecisionShardWriter(shard_dir, max_samples=100)
    rng = np.random.RandomState(42)
    for i in range(n_d):
        # 半分くらいに winner ラベル / yaku を付ける (yaku_loss を non-zero に)
        is_winner = with_winners and (i % 3 == 0)
        kw = {}
        if is_winner:
            kw["round_terminal_label"] = "win_menzen"
            kw["eventual_win_yaku_ids"] = [1, 4]  # Riichi, Yakuhai
        writer.add(_mk_discard(
            step_id=i,
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_d - 1),
            **kw,
        ))
    for i in range(n_c):
        is_winner = with_winners and (i % 3 == 0)
        kw = {}
        if is_winner:
            kw["round_terminal_label"] = "win_called"
            kw["eventual_win_yaku_ids"] = [4]
        writer.add(_mk_call(
            step_id=n_d + i,
            reward=float(rng.randn() * 0.1),
            value=float(rng.randn() * 0.1),
            terminated=(i == n_c - 1),
            **kw,
        ))
    writer.close()


# ========== 1. default off ==========


class TestDefaultOffNoExtraGrad:
    """default config では `torch.autograd.grad` が呼ばれず、追加 gradient
    計算が走らない (= 既存挙動完全互換)"""

    def test_no_gradient_norms_in_diag(self, tmp_path):
        """default config では ppo_diag に gradient_norms が出ない"""
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir)
        learner = _make_learner(tmp_path, gn_enabled=False)
        metrics = learner.train(shard_dir)
        assert "gradient_norms" not in metrics["ppo_diag"]

    def test_default_config_means_disabled(self, tmp_path):
        """training.diagnostics.gradient_norms 自体未指定でも動作する"""
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir)
        model = _make_model(semantic_aux=False)
        # diagnostics は完全未指定
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "ppo", "epochs": 1, "batch_size": 4,
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert "gradient_norms" not in metrics["ppo_diag"]

    def test_autograd_grad_not_called_when_disabled(self, tmp_path,
                                                     monkeypatch):
        """default off で torch.autograd.grad が呼ばれないこと"""
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir)
        learner = _make_learner(tmp_path, gn_enabled=False)

        call_count = {"n": 0}
        original = torch.autograd.grad

        def _spy(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(torch.autograd, "grad", _spy)
        learner.train(shard_dir)
        assert call_count["n"] == 0, (
            f"expected no autograd.grad calls when disabled, got "
            f"{call_count['n']}")


# ========== 2. enabled smoke ==========


class TestEnabledSmoke:
    """enabled=true smoke で完走 + aggregate が存在"""

    def test_aggregate_present(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir)
        learner = _make_learner(tmp_path, gn_enabled=True)
        metrics = learner.train(shard_dir)
        gn = metrics["ppo_diag"]["gradient_norms"]
        assert "aggregate" in gn
        assert "discard" in gn
        assert "call" in gn

    def test_branch_stats_populated(self, tmp_path):
        """discard / call の少なくとも一方に stats が入る"""
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir)
        learner = _make_learner(tmp_path, gn_enabled=True)
        metrics = learner.train(shard_dir)
        gn = metrics["ppo_diag"]["gradient_norms"]
        assert gn["aggregate"]
        # 最低でも policy_loss は出る
        assert "policy_loss" in gn["aggregate"]

    def test_config_recorded(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir)
        learner = _make_learner(tmp_path, gn_enabled=True,
                                 max_batches_per_epoch=2, every_n_epochs=1)
        metrics = learner.train(shard_dir)
        cfg = metrics["ppo_diag"]["gradient_norms"]["config"]
        assert cfg["enabled"] is True
        assert cfg["max_batches_per_epoch"] == 2
        assert cfg["every_n_epochs"] == 1


# ========== 3. schema ==========


class TestSchema:
    """component / group / stats / ratio key の schema を確認"""

    @pytest.fixture
    def metrics(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir, n_d=12, n_c=8)
        learner = _make_learner(tmp_path, gn_enabled=True, semantic_aux=True,
                                 max_batches_per_epoch=4)
        return learner.train(shard_dir)

    def test_aggregate_components_present(self, metrics):
        gn = metrics["ppo_diag"]["gradient_norms"]
        agg = gn["aggregate"]
        # 最低限の component
        for c in ("policy_loss", "value_loss", "weighted_value_loss",
                  "total_loss_pre_clip"):
            assert c in agg, f"missing component {c}"
        # semantic_aux 有効時
        for c in ("terminal_loss", "weighted_terminal_loss",
                  "yaku_loss", "weighted_yaku_loss", "semantic_aux_loss"):
            assert c in agg, f"missing semantic component {c}"

    def test_groups_present(self, metrics):
        gn = metrics["ppo_diag"]["gradient_norms"]
        agg = gn["aggregate"]
        # policy_loss は policy / all は必ず出る
        pol = agg["policy_loss"]
        assert "all" in pol
        assert "policy" in pol
        # value_loss は value_semantic / all
        val = agg["value_loss"]
        assert "all" in val
        assert "value_semantic" in val

    def test_stats_have_required_keys(self, metrics):
        gn = metrics["ppo_diag"]["gradient_norms"]
        agg = gn["aggregate"]
        for cname, gdict in agg.items():
            if cname == "ratios":
                continue
            for gname, stats in gdict.items():
                for k in ("mean", "p50", "p90", "max", "count"):
                    assert k in stats, f"missing {k} in {cname}.{gname}"
                # mean / p50 etc は count > 0 なら float
                if stats["count"] > 0:
                    assert isinstance(stats["mean"], float)
                    assert isinstance(stats["max"], float)
                    assert stats["mean"] >= 0
                    assert stats["max"] >= stats["mean"] - 1e-6

    def test_ratios_present(self, metrics):
        gn = metrics["ppo_diag"]["gradient_norms"]
        ratios = gn["aggregate"]["ratios"]
        for k in (
            "value_semantic_terminal_to_yaku",
            "value_semantic_weighted_terminal_to_weighted_yaku",
            "value_semantic_weighted_terminal_to_weighted_value",
            "value_semantic_weighted_yaku_to_weighted_value",
        ):
            assert k in ratios

    def test_json_serializable(self, metrics):
        gn = metrics["ppo_diag"]["gradient_norms"]
        # crash しないこと
        s = json.dumps(gn)
        assert len(s) > 0


# ========== 4. semantic disabled ==========


class TestSemanticDisabled:
    """semantic_aux.enabled=false で terminal/yaku 系が欠落"""

    def test_no_semantic_components(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir)
        learner = _make_learner(tmp_path, gn_enabled=True,
                                 semantic_aux=False)
        metrics = learner.train(shard_dir)
        agg = metrics["ppo_diag"]["gradient_norms"]["aggregate"]
        # 必須は残る
        assert "policy_loss" in agg
        assert "value_loss" in agg
        assert "total_loss_pre_clip" in agg
        # semantic 系は欠落 (component dict に出ない)
        for c in ("terminal_loss", "yaku_loss", "semantic_aux_loss",
                  "weighted_terminal_loss", "weighted_yaku_loss"):
            assert c not in agg, f"unexpected semantic component {c}"

    def test_ratios_handle_missing_components(self, tmp_path):
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir)
        learner = _make_learner(tmp_path, gn_enabled=True,
                                 semantic_aux=False)
        metrics = learner.train(shard_dir)
        ratios = metrics["ppo_diag"]["gradient_norms"]["aggregate"]["ratios"]
        # semantic が無い場合は terminal/yaku ベースの ratio は None
        assert ratios["value_semantic_terminal_to_yaku"] is None
        assert ratios["value_semantic_weighted_terminal_to_weighted_yaku"] is None
        assert ratios["value_semantic_weighted_terminal_to_weighted_value"] is None
        assert ratios["value_semantic_weighted_yaku_to_weighted_value"] is None


# ========== 5. gradient pollution check ==========


class TestNoGradientPollution:
    """diagnostics 計測で optimizer step 用 .grad が汚染されないこと"""

    def _train_and_get_first_param(self, tmp_path, gn_enabled, *,
                                    max_batches_per_epoch=4, seed=0):
        """同じ seed / 同じ shard で 1 step 動かし、最初の param を返す"""
        shard_dir = tmp_path / "shards"
        if not shard_dir.exists():
            _write_mixed_shard(shard_dir, n_d=8, n_c=4)
        torch.manual_seed(seed)
        np.random.seed(seed)
        learner = _make_learner(
            tmp_path / f"run_{int(gn_enabled)}",
            gn_enabled=gn_enabled,
            max_batches_per_epoch=max_batches_per_epoch,
            batch_size=4)
        # 確定的な model 初期化
        torch.manual_seed(seed + 999)
        for p in learner._model.parameters():
            p.data.normal_(0.0, 0.1)
        # train
        torch.manual_seed(seed)
        np.random.seed(seed)
        learner.train(shard_dir)
        # 最初の param をスナップショット
        return [p.detach().clone() for p in learner._model.parameters()]

    def test_param_updates_identical_with_and_without_diagnostics(
            self, tmp_path):
        """同じ seed で enabled=False と enabled=True を実行して、
        param の更新後値が一致 (diagnostics は学習挙動を変えない)"""
        # 同じ shard を共有
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir, n_d=8, n_c=4)

        params_off = self._train_and_get_first_param(tmp_path, False, seed=7)
        params_on = self._train_and_get_first_param(tmp_path, True, seed=7)

        assert len(params_off) == len(params_on)
        for p_off, p_on in zip(params_off, params_on):
            # tensor 値が完全に一致するはず
            assert p_off.shape == p_on.shape
            diff = (p_off - p_on).abs().max().item()
            assert diff < 1e-6, (
                f"param updated differently with vs without gradient norm "
                f"diagnostics: max abs diff = {diff}")

    def test_grad_zero_after_diagnostics_when_only_grad_path(self, tmp_path):
        """autograd.grad は .grad を populate しない: 単独で測ったあと
        モデルの .grad は依然 None / zero のまま"""
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir, n_d=8, n_c=0)
        learner = _make_learner(tmp_path, gn_enabled=True, batch_size=8,
                                 max_batches_per_epoch=1)
        # zero_grad してから手動で 1 minibatch 分の loss を作って
        # _gn_compute_minibatch_norms を呼ぶだけのテスト
        for p in learner._model.parameters():
            p.grad = None
        # forward
        obs = torch.zeros(2, 10, dtype=torch.float32)
        masks = torch.ones(2, 34, dtype=torch.float32)
        out = learner._model.forward_discard(obs, masks)
        log_p = torch.log_softmax(out.discard_logits, dim=-1)
        action_lp = log_p[:, 0]
        old_lp = torch.zeros(2)
        log_ratio = action_lp - old_lp
        ratio = torch.exp(log_ratio)
        adv = torch.tensor([1.0, -1.0])
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 0.85, 1.15) * adv
        policy_loss = -(torch.min(surr1, surr2)).mean()
        value = out.values["round_delta"].squeeze(-1)
        target = torch.tensor([0.5, -0.5])
        value_loss = (value - target).pow(2).mean()
        loss = policy_loss + 0.125 * value_loss

        result = learner._gn_compute_minibatch_norms(
            policy_loss=policy_loss,
            value_loss=value_loss,
            sa_loss_total=None,
            terminal_loss_t=None,
            yaku_loss_t=None,
            total_loss=loss,
        )
        assert result is not None
        # diagnostics 計測後でも .grad は None のまま (autograd.grad なので)
        any_grad = any(p.grad is not None for p in learner._model.parameters())
        assert not any_grad, (
            "torch.autograd.grad should NOT populate .grad")


# ========== 6. budget control ==========


class TestBudgetControl:
    """max_batches_per_epoch と every_n_epochs が効く"""

    def test_max_batches_limits_count(self, tmp_path):
        """max_batches_per_epoch=1 で 1 epoch あたり最大 1 minibatch しか
        計測しない"""
        shard_dir = tmp_path / "shards"
        # 大きめの shard で minibatch がたくさん作られる
        _write_mixed_shard(shard_dir, n_d=20, n_c=12)
        learner = _make_learner(
            tmp_path, gn_enabled=True,
            max_batches_per_epoch=1, every_n_epochs=1, epochs=1, batch_size=4)
        metrics = learner.train(shard_dir)
        gn = metrics["ppo_diag"]["gradient_norms"]
        # discard / call それぞれ高々 1 minibatch
        for branch in ("discard", "call"):
            stats = gn[branch]
            for cname, gdict in stats.items():
                if cname == "ratios":
                    continue
                for gname, stat in gdict.items():
                    assert stat["count"] <= 1, (
                        f"{branch}.{cname}.{gname} count={stat['count']} > 1")
        # aggregate (discard + call) は最大 2
        for cname, gdict in gn["aggregate"].items():
            if cname == "ratios":
                continue
            for gname, stat in gdict.items():
                assert stat["count"] <= 2

    def test_max_batches_zero_disables(self, tmp_path):
        """max_batches_per_epoch=0 で計測ゼロ"""
        shard_dir = tmp_path / "shards"
        _write_mixed_shard(shard_dir)
        learner = _make_learner(
            tmp_path, gn_enabled=True,
            max_batches_per_epoch=0, every_n_epochs=1, epochs=1)
        metrics = learner.train(shard_dir)
        # buffer は空 → aggregate 内 component は空 dict
        gn = metrics["ppo_diag"]["gradient_norms"]
        agg = gn["aggregate"]
        # ratios は present
        assert "ratios" in agg
        # それ以外の component は欠落
        for cname in agg:
            if cname == "ratios":
                continue
            for gname, stat in agg[cname].items():
                assert stat["count"] == 0
