"""CQ-0224: Stage2a selfplay/imitation データ生成の integration テスト"""
import pytest
import numpy as np
from pathlib import Path

pytestmark = pytest.mark.smoke

from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
from mahjong_rl.call_shard import DecisionShardReader
from mahjong_rl.encoders import FlatFeatureEncoder


class TestStage2ImitationGeneration:
    """Stage2a imitation データ生成 integration"""

    def test_imitation_generates_call_decisions(self, tmp_path: Path):
        """imitation 生成で call decision を含む shard が生成される"""
        output_dir = tmp_path / "imitation_shards"
        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={},
            output_dir=output_dir,
            encoder=encoder,
        )
        stats = worker.generate(
            num_matches=5,
            base_seed=42,
            experiment_id="test_imi",
            run_id="run0",
            worker_id="imi_w0",
        )
        assert stats["total_steps"] > 0
        assert stats["discard_count"] > 0
        assert stats["call_count"] > 0

        # shard を読み返して検証
        reader = DecisionShardReader(output_dir)
        samples = reader.read_all()
        assert len(samples) > 0

        discard_samples = [s for s in samples if s.decision_type == "discard"]
        call_samples = [s for s in samples if s.decision_type == "call"]
        assert len(discard_samples) > 0
        assert len(call_samples) > 0

        # call sample の構造検証
        for cs in call_samples:
            assert cs.candidate_count > 0
            assert len(cs.candidates) == cs.candidate_count
            assert 0 <= cs.selected_candidate_index < cs.candidate_count
            assert cs.actor_type == "baseline"

    def test_observation_is_real_encoded(self, tmp_path: Path):
        """observation が placeholder zeros ではなく実特徴量"""
        output_dir = tmp_path / "obs_shards"
        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=output_dir, encoder=encoder)
        worker.generate(num_matches=2, base_seed=42)
        reader = DecisionShardReader(output_dir)
        samples = reader.read_all()
        assert len(samples) > 0
        # observation が正しい shape / dtype で、非ゼロ値を含む
        expected_dim = encoder.metadata().output_shape[0]
        for s in samples[:10]:
            assert s.observation.shape == (expected_dim,)
            assert s.observation.dtype == np.float32
        # 全 sample が all-zero でないことを確認
        has_nonzero = any(np.any(s.observation != 0) for s in samples[:10])
        assert has_nonzero, "observation が全て zeros — encoder が使われていない"

    def test_imitation_metadata_correct(self, tmp_path: Path):
        """生成された shard のメタデータが正しい"""
        output_dir = tmp_path / "shards"
        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=output_dir, encoder=encoder)
        worker.generate(
            num_matches=2,
            base_seed=100,
            experiment_id="exp_test",
            run_id="run_test",
        )
        reader = DecisionShardReader(output_dir)
        samples = reader.read_all()
        for s in samples:
            assert s.experiment_id == "exp_test"
            assert s.run_id == "run_test"
            assert s.actor_type == "baseline"


class TestStage2SelfplayGeneration:
    """Stage2a selfplay データ生成 integration"""

    def test_selfplay_completes_multiple_matches(self, tmp_path: Path):
        """selfplay が複数半荘で完走する"""
        output_dir = tmp_path / "selfplay_shards"
        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=output_dir, encoder=encoder)
        stats = worker.generate(
            num_matches=3,
            base_seed=0,
            experiment_id="sp_test",
            run_id="sp_run",
        )
        assert stats["num_matches"] == 3
        assert stats["total_steps"] > 100

    def test_consumed_tile_ids_in_shard(self, tmp_path: Path):
        """shard 内の call candidate に consumed_tile_ids が含まれる"""
        output_dir = tmp_path / "shards"
        encoder = FlatFeatureEncoder(observation_mode="full")
        worker = Stage2SelfPlayWorker(
            config={}, output_dir=output_dir, encoder=encoder)
        worker.generate(num_matches=5, base_seed=42)

        reader = DecisionShardReader(output_dir)
        samples = reader.read_all()
        call_samples = [s for s in samples if s.decision_type == "call"]
        assert len(call_samples) > 0

        # Chi/Pon/Daiminkan candidate に consumed_tile_ids がある
        found_consumed = False
        for cs in call_samples:
            for cand in cs.candidates:
                if cand.action_type in (3, 4, 5):  # Chi/Pon/Daiminkan
                    if len(cand.consumed_tile_ids) > 0:
                        found_consumed = True
        assert found_consumed, "consumed_tile_ids を持つ candidate が見つからない"


class TestStage2aRunnerIntegration:
    """Stage2a runner 経路の integration テスト"""

    @staticmethod
    def _make_stage2a_config():
        from mahjong_rl.experiment import ExperimentConfig
        config = ExperimentConfig()
        config.experiment = {
            "name": "test_stage2a",
            "stage": "stage2a",
            "observation_mode": "full",
            "global_seed": 42,
            "phases": ["imitation", "selfplay", "learner"],
        }
        config.feature_encoder = {
            "shanten_hint": True,
            "discard_ukeire_hint": True,
        }
        config.selfplay = {
            "imitation_matches": 3,
            "num_matches": 3,
            "seed_start": 0,
        }
        config.imitation = {"num_matches": 3}
        config.model = {
            "discard_hidden_dims": [32],
            "call_hidden_dims": [32],
            "candidate_dim": 8,
        }
        config.training = {
            "algorithm": "ppo",
            "lr": 1e-3,
            "batch_size": 16,
            "epochs": 1,
        }
        config.evaluation = {"num_matches": 0}
        return config

    def test_stage2a_full_run_via_runner(self, tmp_path: Path):
        """runner 経由で Stage2a imitation → selfplay → learner が通る"""
        from mahjong_rl.runner import Stage1Runner
        import json

        config = self._make_stage2a_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"

        # imitation phase
        imi = result.get("imitation_metrics", {})
        assert imi.get("stage") == "stage2a"
        assert imi.get("call_count", 0) > 0
        assert "train_metrics" in imi

        # selfplay phase
        sp = result.get("selfplay_stats", {})
        assert sp.get("stage") == "stage2a"
        assert sp.get("call_count", 0) > 0

        # learner phase
        tm = result.get("train_metrics", {})
        assert tm.get("mode") == "ppo"
        assert tm.get("num_updates", 0) > 0

        # checkpoint
        run_dir = Path(result["run_dir"])
        assert (run_dir / "checkpoints" / "checkpoint_imitation.pt").exists()
        assert (run_dir / "checkpoints" / "checkpoint_learner.pt").exists()

    def test_stage2a_imitation_only(self, tmp_path: Path):
        """imitation のみの Stage2a run"""
        from mahjong_rl.runner import Stage1Runner

        config = self._make_stage2a_config()
        config.training["algorithm"] = "imitation"
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result, f"error: {result.get('error')}"
        imi = result.get("imitation_metrics", {})
        assert imi.get("stage") == "stage2a"


class TestStage1Regression:
    """Stage1 経路の回帰テスト"""

    def test_stage1_selfplay_untouched(self):
        """Stage1 の SelfPlayWorker import が壊れない"""
        from mahjong_rl.selfplay_worker import SelfPlayWorker
        assert SelfPlayWorker is not None

    def test_stage1_shard_untouched(self):
        """Stage1 の ShardWriter/Reader import が壊れない"""
        from mahjong_rl.shard import ShardWriter, ShardReader, LearningSample
        assert ShardWriter is not None
        assert ShardReader is not None
        assert LearningSample is not None
