"""CQ-0279: DecisionSample sample_semantics_version v3 化と learner fail-fast"""
import pytest
import numpy as np
import torch
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.call_shard import (
    DecisionSample, CandidateRecord,
    DecisionShardWriter, DecisionShardReader,
)
from mahjong_rl.stage2a_learner import Stage2aLearner
from mahjong_rl.models.stage2a_model import Stage2aModel


def _mk_discard_sample(step_id=0, version=None, **kw):
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
    base.update(kw)  # kw が優先
    s = DecisionSample(**base)
    if version is not None:
        s.sample_semantics_version = version
    return s


def _mk_call_sample(step_id=0, version=None, **kw):
    base = dict(
        decision_type="call",
        observation=np.zeros(10, dtype=np.float32),
        reward=0.0, log_prob=-0.5, value=0.0,
        terminated=False, round_over=False,
        selected_candidate_index=0, candidate_count=1,
        candidates=[CandidateRecord(action_type=8)],
        player_id=0, episode_id="ep0", round_id=0,
        step_id=step_id, actor_type="policy",
        experiment_id="t", run_id="r", worker_id="w",
    )
    base.update(kw)
    s = DecisionSample(**base)
    if version is not None:
        s.sample_semantics_version = version
    return s


def _make_model_and_learner(tmp_path, mode="ppo"):
    model = Stage2aModel(input_dim=10, discard_hidden_dims=[8],
                          optional_hidden_dims=[8])
    return Stage2aLearner(
        config={"training": {"algorithm": mode}},
        model=model, run_dir=tmp_path / "run",
        device=torch.device("cpu"),
    )


# ========== default version v3 ==========

class TestDefaultVersionV3:
    """DecisionSample default version が 3"""

    def test_discard_default_v3(self):
        s = _mk_discard_sample()
        assert s.sample_semantics_version == 3

    def test_call_default_v3(self):
        s = _mk_call_sample()
        assert s.sample_semantics_version == 3

    def test_required_constant(self):
        assert Stage2aLearner.REQUIRED_SAMPLE_SEMANTICS_VERSION == 3


# ========== Stage2a selfplay produces v3 shards ==========

class TestSelfplayV3Shard:
    """Stage2a selfplay の generate() が v3 shard を吐く"""

    def test_selfplay_writes_v3(self, tmp_path):
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.encoders import FlatFeatureEncoder

        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=tmp_path,
            observation_mode="full", encoder=encoder,
        )
        worker.generate(num_matches=3, base_seed=42)
        samples = DecisionShardReader(tmp_path).read_all()
        assert len(samples) > 0
        for s in samples:
            assert s.sample_semantics_version == 3, (
                f"Stage2a selfplay shard が v{s.sample_semantics_version} "
                "になっている")


# ========== read_as_tensors にバージョンが含まれる ==========

class TestReadAsTensorsHasVersion:
    """read_as_tensors の dict に sample_semantics_versions が入る"""

    def test_v3_shard_versions(self, tmp_path):
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        for i in range(3):
            writer.add(_mk_discard_sample(step_id=i))
        writer.close()
        data = DecisionShardReader(tmp_path).read_as_tensors()
        d = data["discard"]
        assert "sample_semantics_versions" in d
        v = d["sample_semantics_versions"]
        assert len(v) == 3
        assert all(int(x) == 3 for x in v)

    def test_call_branch_versions(self, tmp_path):
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        writer.add(_mk_call_sample(step_id=0))
        writer.close()
        data = DecisionShardReader(tmp_path).read_as_tensors()
        c = data["call"]
        assert "sample_semantics_versions" in c
        assert int(c["sample_semantics_versions"][0]) == 3

    def test_v2_shard_versions(self, tmp_path):
        """v2 が混ざった shard でも versions が読める"""
        writer = DecisionShardWriter(tmp_path, max_samples=100)
        writer.add(_mk_discard_sample(step_id=0, version=2))
        writer.add(_mk_discard_sample(step_id=1, version=3))
        writer.close()
        data = DecisionShardReader(tmp_path).read_as_tensors()
        d = data["discard"]
        v = sorted(int(x) for x in d["sample_semantics_versions"])
        assert v == [2, 3]


# ========== learner fail-fast ==========

class TestLearnerFailFastV2Shard:
    """v2 shard で Stage2aLearner.train() が ValueError"""

    def test_imitation_path_rejects_v2_discard(self, tmp_path):
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        # all v2 discard
        for i in range(5):
            writer.add(_mk_discard_sample(step_id=i, version=2))
        writer.close()
        learner = _make_model_and_learner(tmp_path, mode="imitation")
        with pytest.raises(ValueError, match="sample_semantics_version=2"):
            learner.train(shard_dir)

    def test_imitation_path_rejects_v2_call(self, tmp_path):
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        for i in range(3):
            writer.add(_mk_call_sample(step_id=i, version=2))
        writer.close()
        learner = _make_model_and_learner(tmp_path, mode="imitation")
        with pytest.raises(ValueError, match="sample_semantics_version=2"):
            learner.train(shard_dir)

    def test_ppo_path_rejects_v2(self, tmp_path):
        """PPO path (read_all 経由) でも v2 を拒否"""
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        for i in range(5):
            writer.add(_mk_discard_sample(
                step_id=i, version=2,
                terminated=(i == 4),
                reward=0.1 * i, value=0.05 * i,
            ))
        writer.close()
        learner = _make_model_and_learner(tmp_path, mode="ppo")
        with pytest.raises(ValueError, match="sample_semantics_version=2"):
            learner.train(shard_dir)

    def test_v3_shard_passes(self, tmp_path):
        """v3 shard なら学習は通る (fail-fast されない)"""
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        for i in range(5):
            writer.add(_mk_discard_sample(
                step_id=i, version=3,
                terminated=(i == 4),
                reward=0.1, value=0.05,
            ))
        writer.close()
        learner = _make_model_and_learner(tmp_path, mode="imitation")
        # crash しなければ OK
        metrics = learner.train(shard_dir)
        assert metrics["mode"] == "imitation"

    def test_mixed_v2_v3_rejected(self, tmp_path):
        """v2 と v3 が混在すると min が v2 で fail-fast"""
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        writer.add(_mk_discard_sample(step_id=0, version=3))
        writer.add(_mk_discard_sample(step_id=1, version=2))
        writer.add(_mk_discard_sample(step_id=2, version=3))
        writer.close()
        learner = _make_model_and_learner(tmp_path, mode="imitation")
        with pytest.raises(ValueError, match="sample_semantics_version=2"):
            learner.train(shard_dir)

    def test_error_message_contains_required_info(self, tmp_path):
        """error message に必要な情報が含まれる"""
        shard_dir = tmp_path / "shards"
        writer = DecisionShardWriter(shard_dir, max_samples=100)
        writer.add(_mk_discard_sample(step_id=0, version=2))
        writer.close()
        learner = _make_model_and_learner(tmp_path, mode="imitation")
        with pytest.raises(ValueError) as exc_info:
            learner.train(shard_dir)
        msg = str(exc_info.value)
        assert "sample_semantics_version=2" in msg
        assert "required: 3" in msg or "required: 3" in msg
        assert "CQ-0274" in msg
        assert "fail-fast" in msg
        assert "再生成" in msg


# ========== Stage1 LearningSample is unaffected ==========

class TestStage1NotAffected:
    """Stage1 LearningSample の version は今回触らない"""

    def test_stage1_default_unchanged(self):
        from mahjong_rl.shard import LearningSample
        # Stage1 LearningSample のフィールドが影響を受けていないこと
        # (default は 1: 既存 contract)
        s = LearningSample(
            observation=np.zeros(10, dtype=np.float32),
            legal_mask=np.ones(34, dtype=np.float32),
            action=0, reward=0.0, log_prob=0.0, value=0.0,
            terminated=False, round_over=False,
            experiment_id="t", run_id="r", worker_id="w",
            episode_id="e",
        )
        # Stage1 default は 1 (legacy)。CQ-0279 で触らない。
        assert s.sample_semantics_version == 1
