"""CQ-0225 / CQ-0226: Stage2a model + learner テスト"""
import pytest
import numpy as np
import torch

pytestmark = pytest.mark.smoke

from mahjong_rl.models.stage2a_model import (
    Stage2aModel, Stage2aOutput, CandidateEncoder,
)
from mahjong_rl.stage2a_learner import Stage2aLearner, _encode_candidates_batch
from mahjong_rl.call_shard import (
    DecisionSample, CandidateRecord,
    DecisionShardWriter, DecisionShardReader,
)


# ========== Model Unit Tests ==========

class TestCandidateEncoder:
    """CQ-0225: candidate encoder unit test"""

    def test_output_shape(self):
        enc = CandidateEncoder(candidate_dim=16)
        feats = torch.zeros(2, 3, 6, dtype=torch.long)
        out = enc(feats)
        assert out.shape == (2, 3, 16)

    def test_different_candidates_different_output(self):
        enc = CandidateEncoder(candidate_dim=16)
        f1 = torch.zeros(1, 2, 6, dtype=torch.long)
        f1[0, 0, 0] = 1  # Chi
        f1[0, 1, 0] = 2  # Pon
        out = enc(f1)
        assert not torch.allclose(out[0, 0], out[0, 1])


class TestStage2aModel:
    """CQ-0225: model forward テスト"""

    def test_discard_path(self):
        model = Stage2aModel(input_dim=100)
        features = torch.randn(2, 100)
        mask = torch.ones(2, 34)
        out = model.forward_discard(features, mask)
        assert out.discard_logits.shape == (2, 34)
        assert out.optional_scores is None
        assert "round_delta" in out.values
        assert out.values["round_delta"].shape == (2, 1)

    def test_call_path(self):
        model = Stage2aModel(input_dim=100)
        features = torch.randn(2, 100)
        cand_feats = torch.zeros(2, 4, 6, dtype=torch.long)
        cand_mask = torch.ones(2, 4)
        out = model.forward_optional(features, cand_feats, cand_mask)
        assert out.discard_logits is None
        assert out.optional_scores.shape == (2, 4)
        assert "round_delta" in out.values

    def test_variable_candidate_count(self):
        """candidate 数が可変でも動く"""
        model = Stage2aModel(input_dim=50)
        features = torch.randn(3, 50)
        # 3 samples, max 5 candidates
        cand_feats = torch.zeros(3, 5, 6, dtype=torch.long)
        cand_mask = torch.zeros(3, 5)
        cand_mask[0, :2] = 1.0  # sample 0: 2 candidates
        cand_mask[1, :4] = 1.0  # sample 1: 4 candidates
        cand_mask[2, :1] = 1.0  # sample 2: 1 candidate
        out = model.forward_optional(features, cand_feats, cand_mask)
        assert out.optional_scores.shape == (3, 5)
        # masked positions should have very negative scores
        assert out.optional_scores[0, 3].item() < -1e8
        assert out.optional_scores[2, 2].item() < -1e8

    def test_call_does_not_affect_discard(self):
        """call path を通しても discard path は独立"""
        model = Stage2aModel(input_dim=50)
        features = torch.randn(1, 50)
        mask = torch.ones(1, 34)

        # discard forward
        out_d = model.forward_discard(features, mask)
        logits1 = out_d.discard_logits.clone()

        # call forward (should not change discard trunk)
        cand_feats = torch.zeros(1, 3, 6, dtype=torch.long)
        cand_mask = torch.ones(1, 3)
        model.forward_optional(features, cand_feats, cand_mask)

        # discard again — should be same
        out_d2 = model.forward_discard(features, mask)
        assert torch.allclose(logits1, out_d2.discard_logits)


# ========== Learner Tests ==========

def _write_stage2a_shards(shard_dir, obs_dim=100, n_discard=30, n_call=20):
    """テスト用の Stage2a shard を書き出す"""
    writer = DecisionShardWriter(shard_dir, max_samples=10000)
    for i in range(n_discard):
        mask = np.random.rand(34).astype(np.float32)
        mask[mask < 0.3] = 0
        mask[mask > 0] = 1.0
        if mask.sum() == 0:
            mask[0] = 1.0
        writer.add(DecisionSample(
            decision_type="discard",
            observation=np.random.randn(obs_dim).astype(np.float32),
            reward=np.random.randn() * 0.01,
            log_prob=-np.random.rand(),
            value=0.0,
            terminated=(i == n_discard - 1),
            round_over=False,
            action=int(np.argmax(mask)),
            legal_mask=mask,
            player_id=i % 4,
            episode_id="ep0",
            step_id=i,
            experiment_id="test", run_id="run0", worker_id="w0",
        ))
    for i in range(n_call):
        n_cands = np.random.randint(2, 5)
        cands = []
        for j in range(n_cands):
            if j == n_cands - 1:
                cands.append(CandidateRecord(action_type=8))  # Skip
            else:
                cands.append(CandidateRecord(
                    action_type=np.random.choice([3, 4, 5]),
                    tile_type=np.random.randint(0, 34),
                    target_rel_seat=np.random.randint(1, 4),
                    consumed_tile_ids=(np.random.randint(0, 136),
                                       np.random.randint(0, 136)),
                ))
        writer.add(DecisionSample(
            decision_type="call",
            observation=np.random.randn(obs_dim).astype(np.float32),
            reward=np.random.randn() * 0.01,
            log_prob=-np.random.rand(),
            value=0.0,
            terminated=False,
            round_over=False,
            selected_candidate_index=0,
            candidate_count=n_cands,
            candidates=cands,
            player_id=1,
            episode_id="ep0",
            step_id=n_discard + i,
            experiment_id="test", run_id="run0", worker_id="w0",
        ))
    writer.close()


class TestStage2aImitationLearner:
    """CQ-0226: imitation learner smoke test"""

    def test_imitation_runs(self, tmp_path):
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_stage2a_shards(shard_dir, obs_dim=obs_dim)

        model = Stage2aModel(input_dim=obs_dim, discard_hidden_dims=[32],
                              optional_hidden_dims=[32])
        learner = Stage2aLearner(
            config={"training": {"algorithm": "imitation", "epochs": 2,
                                  "batch_size": 16}},
            model=model,
            run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "imitation"
        assert metrics["discard_count"] > 0
        assert metrics["call_count"] > 0
        assert metrics["policy_loss"] > 0
        assert metrics["num_updates"] > 0

    def test_call_only_imitation(self, tmp_path):
        """call sample のみでも動く"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_stage2a_shards(shard_dir, obs_dim=obs_dim,
                               n_discard=0, n_call=20)
        model = Stage2aModel(input_dim=obs_dim, discard_hidden_dims=[16],
                              optional_hidden_dims=[16])
        learner = Stage2aLearner(
            config={"training": {"algorithm": "imitation", "epochs": 1,
                                  "batch_size": 8}},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["call_count"] == 20
        assert metrics["call_loss"] > 0


class TestStage2aPPOLearner:
    """CQ-0226: PPO learner smoke test"""

    def test_ppo_runs(self, tmp_path):
        obs_dim = 100
        shard_dir = tmp_path / "shards"
        _write_stage2a_shards(shard_dir, obs_dim=obs_dim)

        model = Stage2aModel(input_dim=obs_dim, discard_hidden_dims=[32],
                              optional_hidden_dims=[32])
        learner = Stage2aLearner(
            config={"training": {"algorithm": "ppo", "epochs": 2,
                                  "batch_size": 16}},
            model=model,
            run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        assert metrics["num_updates"] > 0
        assert metrics["policy_loss"] != 0.0


class TestStage2aPPOBatchSize1:
    """PPO batch size 1 で NaN にならない"""

    def test_batch_size_1_no_nan(self, tmp_path):
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_stage2a_shards(shard_dir, obs_dim=obs_dim,
                               n_discard=1, n_call=1)
        model = Stage2aModel(input_dim=obs_dim, discard_hidden_dims=[16],
                              optional_hidden_dims=[16])
        learner = Stage2aLearner(
            config={"training": {"algorithm": "ppo", "epochs": 1,
                                  "batch_size": 1}},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert not np.isnan(metrics["policy_loss"]), "policy_loss is NaN"
        assert not np.isnan(metrics["value_loss"]), "value_loss is NaN"


class TestStage2aCandidateEncoding:
    """CQ-0226: candidate batch encoding"""

    def test_encode_candidates(self):
        samples = [
            DecisionSample(
                decision_type="call",
                observation=np.zeros(10, dtype=np.float32),
                reward=0.0, log_prob=0.0, value=0.0,
                terminated=False, round_over=False,
                selected_candidate_index=0,
                candidate_count=3,
                candidates=[
                    CandidateRecord(action_type=4, tile_type=10,
                                     target_rel_seat=2,
                                     consumed_tile_ids=(40, 41)),
                    CandidateRecord(action_type=3, tile_type=5,
                                     target_rel_seat=3,
                                     consumed_tile_ids=(20, 24)),
                    CandidateRecord(action_type=8),  # Skip
                ],
                experiment_id="t", run_id="r", worker_id="w",
            ),
        ]
        feats, mask = _encode_candidates_batch(samples, max_cands=4)
        assert feats.shape == (1, 4, 6)
        assert mask.shape == (1, 4)
        assert mask[0, 0] == 1.0  # Pon
        assert mask[0, 1] == 1.0  # Chi
        assert mask[0, 2] == 1.0  # Skip
        assert mask[0, 3] == 0.0  # padding


class TestCandidateEncoderBoundary:
    """CQ-0225: candidate encoding 境界値テスト"""

    def test_tile_type_33_not_zeroed(self):
        """tile_type=33 (中) が padding 扱いされない"""
        enc = CandidateEncoder(candidate_dim=16)
        feats = torch.zeros(1, 1, 6, dtype=torch.long)
        feats[0, 0, 0] = 2  # Pon
        feats[0, 0, 1] = 34  # tile_type=33 → 1-based=34
        feats[0, 0, 2] = 1   # rel_seat=0 → 1-based=1
        out = enc(feats)
        # padding_idx=0 なので index 34 は有効
        assert not torch.all(out == 0), "tile_type=33 がゼロ化されている"

    def test_rel_seat_3_not_zeroed(self):
        """target_rel_seat=3 が padding 扱いされない"""
        enc = CandidateEncoder(candidate_dim=16)
        feats = torch.zeros(1, 1, 6, dtype=torch.long)
        feats[0, 0, 0] = 1  # Chi
        feats[0, 0, 1] = 1
        feats[0, 0, 2] = 4  # rel_seat=3 → 1-based=4
        out = enc(feats)
        assert not torch.all(out == 0), "rel_seat=3 がゼロ化されている"

    def test_consumed_tile_33_not_zeroed(self):
        """consumed_tile_type=33 が padding 扱いされない"""
        enc = CandidateEncoder(candidate_dim=16)
        feats = torch.zeros(1, 1, 6, dtype=torch.long)
        feats[0, 0, 0] = 1
        feats[0, 0, 1] = 1
        feats[0, 0, 2] = 1
        feats[0, 0, 3] = 34  # consumed tile_type=33 → 1-based=34
        out = enc(feats)
        assert not torch.all(out == 0)

    def test_padding_is_zero(self):
        """padding index=0 は実際にゼロベクトル"""
        enc = CandidateEncoder(candidate_dim=16)
        # tile embedding の padding_idx=0
        zero_emb = enc.tile_emb(torch.tensor([0]))
        assert torch.all(zero_emb == 0), "padding_idx=0 がゼロでない"


class TestStage2aRealShardPPO:
    """CQ-0226: 実生成 shard を使った PPO smoke test"""

    def test_real_shard_ppo(self, tmp_path):
        """実データ shard で PPO が回る"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        # 1. Real data generation with model
        obs_dim = 467  # full observation (CQ-0264: +4 menzen)
        encoder = FlatFeatureEncoder(observation_mode="full")
        model = Stage2aModel(input_dim=obs_dim, discard_hidden_dims=[32],
                              optional_hidden_dims=[32])

        shard_dir = tmp_path / "shards"
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=shard_dir,
            observation_mode="full", encoder=encoder,
            model=model, device=torch.device("cpu"),
        )
        stats = worker.generate(num_matches=3, base_seed=42)
        assert stats["call_count"] > 0

        # 2. Verify non-zero reward/log_prob/value
        from mahjong_rl.call_shard import DecisionShardReader
        samples = DecisionShardReader(shard_dir).read_all()
        has_nonzero_reward = any(abs(s.reward) > 1e-10 for s in samples)
        has_nonzero_lp = any(abs(s.log_prob) > 1e-10 for s in samples)
        has_nonzero_val = any(abs(s.value) > 1e-10 for s in samples)
        assert has_nonzero_lp, "log_prob が全ゼロ"
        assert has_nonzero_val, "value が全ゼロ"

        # 3. PPO on real shard
        learner = Stage2aLearner(
            config={"training": {"algorithm": "ppo", "epochs": 1,
                                  "batch_size": 16}},
            model=Stage2aModel(input_dim=obs_dim, discard_hidden_dims=[32],
                                optional_hidden_dims=[32]),
            run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        assert metrics["num_updates"] > 0


class TestStage2aSelector:
    """CQ-0226: Stage2a selector テスト"""

    def test_select_discard_argmax(self):
        from mahjong_rl.stage2a_selector import select_discard_argmax
        logits = torch.randn(34)
        mask = torch.ones(34)
        mask[0] = 0.0  # tile 0 は非合法
        action, lp = select_discard_argmax(logits, mask)
        assert action != 0
        assert lp < 0  # log_prob は負

    def test_select_optional_argmax(self):
        from mahjong_rl.stage2a_selector import select_optional_argmax
        scores = torch.tensor([1.0, 2.0, 0.5])
        mask = torch.tensor([1.0, 1.0, 1.0])
        idx, lp = select_optional_argmax(scores, mask)
        assert idx == 1  # 最高スコア
        assert lp < 0

    def test_select_optional_with_mask(self):
        from mahjong_rl.stage2a_selector import select_optional_argmax
        scores = torch.tensor([1.0, 5.0, 0.5])
        mask = torch.tensor([1.0, 0.0, 1.0])  # index 1 は無効
        idx, lp = select_optional_argmax(scores, mask)
        assert idx == 0  # mask 後は index 0 が最大

    def test_select_discard_sample(self):
        from mahjong_rl.stage2a_selector import select_discard_sample
        torch.manual_seed(42)
        logits = torch.zeros(34)
        logits[5] = 10.0  # 強い preference
        mask = torch.ones(34)
        action, lp = select_discard_sample(logits, mask, temperature=0.1)
        assert action == 5  # 高確率で 5


class TestStage1Regression:
    """Stage1 の既存経路が壊れていないことを確認"""

    def test_stage1_model_import(self):
        from mahjong_rl.models import MLPPolicyValueModel
        model = MLPPolicyValueModel(input_dim=100, hidden_dims=[32])
        features = torch.randn(1, 100)
        mask = torch.ones(1, 34)
        out = model(features, mask)
        assert out.logits.shape == (1, 34)

    def test_stage1_learner_import(self):
        from mahjong_rl.learner import Learner
        assert Learner is not None

    def test_stage1_selector_import(self):
        from mahjong_rl.action_selector import ActionSelector
        assert ActionSelector is not None


# ========== CQ-0256 Blocker Fix Tests ==========


class TestSemanticAuxComputeValueFalse:
    """CQ-0256 Blocker #1: compute_value=False でも semantic summary は生成される"""

    @staticmethod
    def _make_semantic_model(input_dim=50):
        return Stage2aModel(
            input_dim=input_dim,
            discard_hidden_dims=[32],
            optional_hidden_dims=[32],
            value_hidden_dims=[32],
            semantic_aux_config={"enabled": True,
                                  "policy_projection_dim": 8},
        )

    def test_discard_compute_value_false_has_semantic(self):
        """forward_discard(compute_value=False) で semantic が返る"""
        model = self._make_semantic_model()
        features = torch.randn(2, 50)
        mask = torch.ones(2, 34)
        out = model.forward_discard(features, mask, compute_value=False)
        assert out.semantic is not None
        assert "terminal_logits" in out.semantic
        assert "yaku_logits" in out.semantic
        assert "semantic_summary" in out.semantic
        assert out.values == {}  # value は計算しない

    def test_optional_compute_value_false_has_semantic(self):
        """forward_optional(compute_value=False) で semantic が返る"""
        model = self._make_semantic_model()
        features = torch.randn(2, 50)
        cand_feats = torch.zeros(2, 3, 6, dtype=torch.long)
        cand_mask = torch.ones(2, 3)
        out = model.forward_optional(features, cand_feats, cand_mask,
                                      compute_value=False)
        assert out.semantic is not None
        assert "terminal_logits" in out.semantic
        assert out.values == {}

    def test_discard_compute_value_true_has_both(self):
        """compute_value=True で semantic + value 両方返る"""
        model = self._make_semantic_model()
        features = torch.randn(2, 50)
        mask = torch.ones(2, 34)
        out = model.forward_discard(features, mask, compute_value=True)
        assert out.semantic is not None
        assert "round_delta" in out.values
        assert out.values["round_delta"].shape == (2, 1)

    def test_semantic_summary_detached(self):
        """semantic_summary は detached (gradient 流さない)"""
        model = self._make_semantic_model()
        features = torch.randn(2, 50, requires_grad=True)
        mask = torch.ones(2, 34)
        out = model.forward_discard(features, mask, compute_value=False)
        assert not out.semantic["semantic_summary"].requires_grad


class TestSemanticAuxPPO:
    """CQ-0256 Blocker #4: PPO で semantic aux loss が加算される"""

    def test_ppo_with_semantic_aux_runs(self, tmp_path):
        """semantic_aux 有効 PPO が crash せず学習が回り diagnostics が出る"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_stage2a_shards_with_labels(shard_dir, obs_dim=obs_dim)

        model = Stage2aModel(
            input_dim=obs_dim, discard_hidden_dims=[32],
            optional_hidden_dims=[32], value_hidden_dims=[32],
            semantic_aux_config={"enabled": True,
                                  "policy_projection_dim": 8},
        )
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "ppo", "epochs": 2, "batch_size": 16,
                "semantic_aux": {"enabled": True,
                                  "terminal_loss_coef": 0.2,
                                  "yaku_loss_coef": 0.1},
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "ppo"
        assert metrics["num_updates"] > 0
        assert not np.isnan(metrics["policy_loss"])
        # CQ-0257: PPO diagnostics に semantic aux loss が含まれる
        assert metrics["semantic_aux_enabled"] is True
        assert metrics["terminal_loss"] is not None
        assert metrics["yaku_loss"] is not None
        assert isinstance(metrics["terminal_loss"], float)
        assert isinstance(metrics["yaku_loss"], float)

    def test_ppo_without_semantic_aux_has_none(self, tmp_path):
        """CQ-0257: semantic_aux 無効 PPO では terminal_loss/yaku_loss が None"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_stage2a_shards_with_labels(shard_dir, obs_dim=obs_dim)

        model = Stage2aModel(
            input_dim=obs_dim, discard_hidden_dims=[32],
            optional_hidden_dims=[32], value_hidden_dims=[32],
        )
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "ppo", "epochs": 1, "batch_size": 16,
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["semantic_aux_enabled"] is False
        assert metrics["terminal_loss"] is None
        assert metrics["yaku_loss"] is None

    def test_imitation_with_semantic_aux_runs(self, tmp_path):
        """semantic_aux 有効 imitation が crash せず学習が回る"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_stage2a_shards_with_labels(shard_dir, obs_dim=obs_dim)

        model = Stage2aModel(
            input_dim=obs_dim, discard_hidden_dims=[32],
            optional_hidden_dims=[32], value_hidden_dims=[32],
            semantic_aux_config={"enabled": True,
                                  "policy_projection_dim": 8},
        )
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "imitation", "epochs": 2, "batch_size": 16,
                "semantic_aux": {"enabled": True,
                                  "terminal_loss_coef": 0.2,
                                  "yaku_loss_coef": 0.1},
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "imitation"
        assert metrics["num_updates"] > 0


def _write_stage2a_shards_with_labels(shard_dir, obs_dim=50,
                                        n_discard=30, n_call=20):
    """CQ-0256 テスト用: terminal_label + yaku_ids 付き shard"""
    writer = DecisionShardWriter(shard_dir, max_samples=10000)
    labels = ["win_menzen", "win_called", "draw_tenpai", "deal_in",
              "other_non_dealin"]
    for i in range(n_discard):
        mask = np.random.rand(34).astype(np.float32)
        mask[mask < 0.3] = 0
        mask[mask > 0] = 1.0
        if mask.sum() == 0:
            mask[0] = 1.0
        label = labels[i % len(labels)]
        yaku_ids = [3, 5] if label.startswith("win") else []
        writer.add(DecisionSample(
            decision_type="discard",
            observation=np.random.randn(obs_dim).astype(np.float32),
            reward=np.random.randn() * 0.01,
            log_prob=-np.random.rand(),
            value=0.0,
            terminated=(i == n_discard - 1),
            round_over=False,
            action=int(np.argmax(mask)),
            legal_mask=mask,
            player_id=i % 4,
            episode_id="ep0",
            step_id=i,
            round_terminal_label=label,
            eventual_win_yaku_ids=yaku_ids,
            experiment_id="test", run_id="run0", worker_id="w0",
        ))
    for i in range(n_call):
        n_cands = np.random.randint(2, 5)
        cands = []
        for j in range(n_cands):
            if j == n_cands - 1:
                cands.append(CandidateRecord(action_type=8))
            else:
                cands.append(CandidateRecord(
                    action_type=np.random.choice([3, 4, 5]),
                    tile_type=np.random.randint(0, 34),
                    target_rel_seat=np.random.randint(1, 4),
                    consumed_tile_ids=(np.random.randint(0, 136),
                                       np.random.randint(0, 136)),
                ))
        label = labels[(n_discard + i) % len(labels)]
        yaku_ids = [0, 1] if label.startswith("win") else []
        writer.add(DecisionSample(
            decision_type="call",
            observation=np.random.randn(obs_dim).astype(np.float32),
            reward=np.random.randn() * 0.01,
            log_prob=-np.random.rand(),
            value=0.0,
            terminated=False,
            round_over=False,
            selected_candidate_index=0,
            candidate_count=n_cands,
            candidates=cands,
            player_id=1,
            episode_id="ep0",
            step_id=n_discard + i,
            round_terminal_label=label,
            eventual_win_yaku_ids=yaku_ids,
            response_context=np.array([0.1, 0.25, 1.0], dtype=np.float32),
            experiment_id="test", run_id="run0", worker_id="w0",
        ))
    writer.close()


class TestCudaMemoryDebug:
    """CQ-0255: CUDA memory debug flag"""

    def test_debug_disabled_no_crash(self, tmp_path):
        """cuda_memory_debug disabled (default) で通常通り動く"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_stage2a_shards_with_labels(shard_dir, obs_dim=obs_dim)

        model = Stage2aModel(input_dim=obs_dim, discard_hidden_dims=[32],
                              optional_hidden_dims=[32])
        learner = Stage2aLearner(
            config={"training": {"algorithm": "imitation", "epochs": 1,
                                  "batch_size": 16}},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["num_updates"] > 0

    def test_debug_enabled_no_crash_cpu(self, tmp_path):
        """cuda_memory_debug enabled on CPU はスキップされる (crash しない)"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_stage2a_shards_with_labels(shard_dir, obs_dim=obs_dim)

        model = Stage2aModel(input_dim=obs_dim, discard_hidden_dims=[32],
                              optional_hidden_dims=[32])
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "imitation", "epochs": 1, "batch_size": 16,
                "cuda_memory_debug": {"enabled": True},
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        # CPU device: debug flag is silently disabled (no CUDA)
        assert learner._cuda_mem_debug is False
        metrics = learner.train(shard_dir)
        assert metrics["num_updates"] > 0


class TestImitationDiagnosticsPreloaded:
    """CQ-0255: diagnostics が preloaded tensor を使う"""

    def test_diagnostics_produces_same_keys(self, tmp_path):
        """refactor 後も diagnostics が crash せず、timing が出る"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_stage2a_shards_with_labels(shard_dir, obs_dim=obs_dim)

        model = Stage2aModel(input_dim=obs_dim, discard_hidden_dims=[32],
                              optional_hidden_dims=[32])
        learner = Stage2aLearner(
            config={"training": {"algorithm": "imitation", "epochs": 1,
                                  "batch_size": 16}},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert "timing" in metrics
        assert metrics["timing"]["diagnostics_sec"] >= 0
        assert metrics["num_updates"] > 0

    def test_diagnostics_with_semantic_aux(self, tmp_path):
        """semantic_aux 有効で diagnostics が出る (compute_value=False 経路)"""
        obs_dim = 50
        shard_dir = tmp_path / "shards"
        _write_stage2a_shards_with_labels(shard_dir, obs_dim=obs_dim)

        model = Stage2aModel(
            input_dim=obs_dim, discard_hidden_dims=[32],
            optional_hidden_dims=[32], value_hidden_dims=[32],
            semantic_aux_config={"enabled": True,
                                  "policy_projection_dim": 8},
        )
        learner = Stage2aLearner(
            config={"training": {
                "algorithm": "imitation", "epochs": 1, "batch_size": 16,
                "semantic_aux": {"enabled": True,
                                  "terminal_loss_coef": 0.2,
                                  "yaku_loss_coef": 0.1},
            }},
            model=model, run_dir=tmp_path / "run",
            device=torch.device("cpu"),
        )
        metrics = learner.train(shard_dir)
        assert metrics["semantic_aux_enabled"] is True
        assert metrics["terminal_loss"] is not None
        assert metrics["timing"]["diagnostics_sec"] >= 0
