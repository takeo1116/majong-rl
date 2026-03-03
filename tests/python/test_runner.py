"""テスト: runner.py — Stage 1 統合ランナー"""
import pytest
from pathlib import Path

from mahjong_rl.experiment import ExperimentConfig
from mahjong_rl.runner import Stage1Runner


def _make_minimal_config() -> ExperimentConfig:
    """最小構成の実験設定"""
    return ExperimentConfig(
        experiment={"name": "test_run", "stage": 1, "observation_mode": "full"},
        feature_encoder={"name": "FlatFeatureEncoder", "observation_mode": "full"},
        model={"name": "MLPPolicyValueModel", "hidden_dims": [32], "value_heads": ["round_delta"]},
        reward={"type": "point_delta"},
        selfplay={"num_matches": 2, "policy_ratio": 1.0, "temperature": 1.0,
                  "max_samples_per_shard": 10000},
        training={"algorithm": "ppo", "lr": 0.001, "batch_size": 32,
                  "epochs": 1, "gamma": 0.99, "gae_lambda": 0.95,
                  "clip_epsilon": 0.2, "value_loss_coef": 0.5,
                  "entropy_coef": 0.01, "max_grad_norm": 0.5},
        evaluation={"num_matches": 1, "seed_start": 100000},
    )


class TestStage1Runner:
    """統合ランナーテスト (CQ-0041)"""

    def test_full_run_completes(self, tmp_path: Path):
        """config から最小 run が完走する"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert "run_dir" in result
        assert "selfplay_stats" in result
        assert "train_metrics" in result
        assert "eval_metrics" in result

    def test_run_dir_has_artifacts(self, tmp_path: Path):
        """run ディレクトリに成果物が保存される"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "notes.md").exists()

        # self-play shard
        shards = list((run_dir / "selfplay").glob("shard_*.parquet"))
        assert len(shards) >= 1

        # checkpoint
        ckpts = list((run_dir / "checkpoints").glob("*.pt"))
        assert len(ckpts) >= 1

        # eval metrics
        assert (run_dir / "eval" / "eval_metrics.json").exists()

    def test_from_yaml(self, tmp_path: Path):
        """YAML から読み込んだ config で実行できる"""
        config = _make_minimal_config()
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        loaded = ExperimentConfig.from_yaml(yaml_path)
        runner = Stage1Runner(config=loaded, base_dir=tmp_path / "runs")
        result = runner.run()

        assert "error" not in result
        assert result["selfplay_stats"]["total_steps"] > 0

    def test_result_contains_metrics(self, tmp_path: Path):
        """結果に主要指標が含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert result["train_metrics"]["total_steps"] > 0
        assert isinstance(result["train_metrics"]["policy_loss"], float)
        assert 1.0 <= result["eval_metrics"]["avg_rank"] <= 4.0


class TestPhaseConfig:
    """フェーズ設定テスト (CQ-0046)"""

    def test_default_phases(self, tmp_path: Path):
        """デフォルトは selfplay → learner → eval"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert result["phases"] == ["selfplay", "learner", "eval"]
        assert "error" not in result

    def test_custom_phases(self, tmp_path: Path):
        """experiment.phases でフェーズを指定できる"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["selfplay", "learner"]
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert result["phases"] == ["selfplay", "learner"]
        assert "eval_metrics" not in result
        assert "error" not in result

    def test_imitation_phase(self, tmp_path: Path):
        """imitation フェーズを含む run が完走する"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["imitation", "selfplay", "learner", "eval"]
        config.selfplay["imitation_matches"] = 1
        config.training["imitation_epochs"] = 1
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert "imitation_metrics" in result
        assert "selfplay_stats" in result
        assert "train_metrics" in result
        assert "eval_metrics" in result

    def test_phases_recorded_in_result(self, tmp_path: Path):
        """実行フェーズが result に記録される"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["selfplay", "eval"]
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert result["phases"] == ["selfplay", "eval"]
        assert "selfplay_stats" in result
        assert "eval_metrics" in result
        # learner スキップなので train_metrics なし
        assert "train_metrics" not in result
