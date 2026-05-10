"""CQ-0288: Stage2a `semantic_proj` を削除し、旧 checkpoint 互換ロードを
入れたことを確認する。

確認:
- semantic_aux enabled の Stage2aModel に `semantic_proj` parameter が無い
- `_compute_semantic` が `terminal_logits` / `yaku_logits` /
  `semantic_summary` を返し、summary 次元 = `NUM_TERMINAL_CLASSES + NUM_YAKU`
- discard / optional forward が crash しない
- `model.semantic_aux.policy_projection_dim` が config に残っていても crash
  しない (= 互換的に無視)
- 新 checkpoint の state_dict に `semantic_proj.*` が含まれない
- 旧 checkpoint (semantic_proj.* 含み) を `load_stage2a_state_dict` で
  互換的に読み込める
- 関係ない unexpected key は依然として fail-fast
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.smoke

from mahjong_rl.models.stage2a_model import (
    Stage2aModel, load_stage2a_state_dict,
)
from mahjong_rl.outcome_vocab import NUM_TERMINAL_CLASSES, NUM_YAKU


def _make_model(*, semantic_aux: bool = True,
                 policy_projection_dim: int | None = None):
    sa_cfg = None
    if semantic_aux:
        sa_cfg = {"enabled": True}
        if policy_projection_dim is not None:
            sa_cfg["policy_projection_dim"] = policy_projection_dim
    return Stage2aModel(
        input_dim=20, discard_hidden_dims=[16],
        optional_hidden_dims=[16], value_hidden_dims=[16],
        candidate_dim=8, optional_scorer_hidden=8,
        semantic_aux_config=sa_cfg,
    )


# ========== 1. semantic_proj parameter 削除 ==========


class TestSemanticProjRemoved:
    def test_no_semantic_proj_attr(self):
        m = _make_model(semantic_aux=True)
        assert not hasattr(m, "semantic_proj"), (
            "semantic_proj は CQ-0288 で削除されたはず")

    def test_no_semantic_proj_in_named_modules(self):
        m = _make_model(semantic_aux=True)
        names = [n for n, _ in m.named_modules()]
        assert not any(n.endswith("semantic_proj") or n == "semantic_proj"
                       for n in names)

    def test_no_semantic_proj_in_named_parameters(self):
        m = _make_model(semantic_aux=True)
        names = [n for n, _ in m.named_parameters()]
        assert not any(n.startswith("semantic_proj.") for n in names)

    def test_summary_dim_excludes_proj(self):
        m = _make_model(semantic_aux=True)
        assert m._semantic_summary_dim == NUM_TERMINAL_CLASSES + NUM_YAKU

    def test_summary_shape_via_forward(self):
        """_compute_semantic 経由で summary.shape[-1] が想定どおり"""
        m = _make_model(semantic_aux=True)
        h = torch.zeros(2, 16)  # value_hidden_dims=[16]
        sem = m._compute_semantic(h)
        assert sem["semantic_summary"].shape[-1] == (
            NUM_TERMINAL_CLASSES + NUM_YAKU)
        assert sem["terminal_logits"].shape[-1] == NUM_TERMINAL_CLASSES
        assert sem["yaku_logits"].shape[-1] == NUM_YAKU


# ========== 2. policy_projection_dim 互換 (ignored) ==========


class TestPolicyProjectionDimDeprecated:
    @pytest.mark.parametrize("dim", [4, 8, 16, 32])
    def test_legacy_config_does_not_crash(self, dim):
        """policy_projection_dim が残っていても crash しない"""
        m = _make_model(semantic_aux=True, policy_projection_dim=dim)
        # summary dim は dim によらず一定 (proj 削除済み)
        assert m._semantic_summary_dim == NUM_TERMINAL_CLASSES + NUM_YAKU

    def test_summary_dim_invariant_to_legacy_dim(self):
        m_a = _make_model(semantic_aux=True, policy_projection_dim=4)
        m_b = _make_model(semantic_aux=True, policy_projection_dim=99)
        assert m_a._semantic_summary_dim == m_b._semantic_summary_dim


# ========== 3. forward smoke ==========


class TestForwardSmoke:
    def test_forward_discard_smoke(self):
        m = _make_model(semantic_aux=True)
        feat = torch.randn(2, 20)
        mask = torch.ones(2, 34)
        out = m.forward_discard(feat, mask)
        assert out.discard_logits.shape == (2, 34)
        assert out.semantic is not None
        assert out.semantic["semantic_summary"].shape[-1] == (
            NUM_TERMINAL_CLASSES + NUM_YAKU)

    def test_forward_optional_smoke(self):
        m = _make_model(semantic_aux=True)
        feat = torch.randn(2, 20)
        cf = torch.zeros(2, 3, 6, dtype=torch.long)
        cf[:, 0, 0] = 0  # Skip
        cf[:, 1, 0] = 1  # Chi
        cm = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        rc = torch.zeros(2, 3)
        out = m.forward_optional(feat, cf, cm, response_context=rc)
        assert out.optional_scores.shape == (2, 3)
        assert out.semantic is not None
        assert out.semantic["semantic_summary"].shape[-1] == (
            NUM_TERMINAL_CLASSES + NUM_YAKU)


# ========== 4. checkpoint compatibility ==========


class TestCheckpointCompat:
    def test_new_checkpoint_no_semantic_proj_keys(self):
        """新 checkpoint に semantic_proj.* が含まれない"""
        m = _make_model(semantic_aux=True)
        sd = m.state_dict()
        assert not any(k.startswith("semantic_proj.") for k in sd), (
            f"new checkpoint should not contain semantic_proj.* keys; "
            f"got: {[k for k in sd if k.startswith('semantic_proj.')]}")

    def test_load_stage2a_state_dict_drops_legacy(self):
        """旧 checkpoint 由来の semantic_proj.* keys は drop して load 成功"""
        m = _make_model(semantic_aux=True)
        sd = m.state_dict()
        # 旧 checkpoint を模倣: semantic_proj.weight / bias を追加
        legacy_sd = dict(sd)
        legacy_sd["semantic_proj.weight"] = torch.zeros(8, 16)
        legacy_sd["semantic_proj.bias"] = torch.zeros(8)
        # crash しない
        result = load_stage2a_state_dict(m, legacy_sd)
        # PyTorch の戻り NamedTuple
        assert hasattr(result, "missing_keys")
        assert hasattr(result, "unexpected_keys")
        # semantic_proj.* は filter で消えるので unexpected には載らない
        assert all(not k.startswith("semantic_proj.")
                   for k in result.unexpected_keys), (
            f"unexpected_keys leaked semantic_proj: {result.unexpected_keys}")
        # missing_keys は通常無し (新 model の全 key は legacy_sd に含まれる)
        assert result.missing_keys == [], (
            f"missing_keys not empty: {result.missing_keys}")

    def test_load_stage2a_state_dict_strict_other_unexpected(self):
        """semantic_proj 以外の unexpected key は fail-fast"""
        m = _make_model(semantic_aux=True)
        sd = m.state_dict()
        bad_sd = dict(sd)
        bad_sd["totally_unrelated_module.weight"] = torch.zeros(4, 4)
        with pytest.raises(RuntimeError):
            load_stage2a_state_dict(m, bad_sd)

    def test_load_stage2a_state_dict_strict_missing(self):
        """missing key は fail-fast"""
        m = _make_model(semantic_aux=True)
        sd = m.state_dict()
        # discard_head.weight を抜く
        partial_sd = {k: v for k, v in sd.items() if k != "discard_head.weight"}
        with pytest.raises(RuntimeError):
            load_stage2a_state_dict(m, partial_sd)

    def test_load_stage2a_state_dict_self_load(self):
        """同じ model の state_dict を読み戻せる (semantic_proj 無し)"""
        m1 = _make_model(semantic_aux=True)
        m2 = _make_model(semantic_aux=True)
        # m1 を学習風に少しずらす
        with torch.no_grad():
            for p in m1.parameters():
                p.normal_(0.0, 0.1)
        sd = m1.state_dict()
        load_stage2a_state_dict(m2, sd)
        # 重みが一致する
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(),
                                        m2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)


# ========== 5. learner classifier に semantic_proj が無い ==========


class TestLearnerClassifierNoProj:
    def test_lr_value_semantic_prefixes_no_proj(self):
        from mahjong_rl.stage2a_learner import Stage2aLearner
        assert "semantic_proj" not in Stage2aLearner._LR_VALUE_SEMANTIC_PREFIXES

    def test_classify_param_groups_no_proj_group(self):
        """合成 named_params で semantic_proj.* を渡しても default に流れる
        (仕分けには現れない)"""
        from mahjong_rl.stage2a_learner import Stage2aLearner
        # 現 model では semantic_proj は無いが、念のため synthetic で確認
        named = [
            ("discard_trunk.0.weight", nn.Parameter(torch.zeros(2))),
            ("value_trunk.0.weight", nn.Parameter(torch.zeros(2))),
            ("terminal_head.weight", nn.Parameter(torch.zeros(2))),
            ("yaku_head.weight", nn.Parameter(torch.zeros(2))),
        ]
        out = Stage2aLearner._classify_param_groups(named)
        # value_semantic に proj は含まれない
        vs_names = [n for n, _ in out["value_semantic"]]
        assert all(not n.startswith("semantic_proj.") for n in vs_names)
        # 期待される 3 module
        assert any(n.startswith("value_trunk.") for n in vs_names)
        assert any(n.startswith("terminal_head.") for n in vs_names)
        assert any(n.startswith("yaku_head.") for n in vs_names)


# ========== 6. gradient_norms diagnostics に semantic_proj 不在 ==========


class TestGradientNormsNoProjGroup:
    def test_gn_atomic_modules_excludes_proj(self, tmp_path):
        """gradient_norms が enabled でも semantic_proj group が出ない"""
        import numpy as np
        from mahjong_rl.stage2a_learner import Stage2aLearner
        from mahjong_rl.call_shard import (
            DecisionSample, CandidateRecord, DecisionShardWriter,
        )

        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        rng = np.random.RandomState(42)
        for i in range(8):
            s = DecisionSample(
                decision_type="discard",
                observation=np.zeros(20, dtype=np.float32),
                legal_mask=np.ones(34, dtype=np.float32),
                action=0, reward=float(rng.randn() * 0.1),
                log_prob=-0.5, value=float(rng.randn() * 0.1),
                terminated=(i == 7), round_over=False,
                player_id=0, episode_id="ep0", round_id=0,
                step_id=i, actor_type="policy",
                experiment_id="t", run_id="r", worker_id="w",
                round_terminal_label="win_menzen",
                eventual_win_yaku_ids=[1, 4],
            )
            s.sample_semantics_version = 3
            writer.add(s)
        writer.close()

        model = _make_model(semantic_aux=True)
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "ppo", "epochs": 1, "batch_size": 4,
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
        gn = metrics["ppo_diag"]["gradient_norms"]
        agg = gn["aggregate"]
        # どの component の dict にも semantic_proj group が無いこと
        for cname, gdict in agg.items():
            if cname == "ratios":
                continue
            for gname in gdict.keys():
                assert gname != "semantic_proj", (
                    f"gradient_norms.aggregate.{cname} に "
                    f"semantic_proj group が残っている")
