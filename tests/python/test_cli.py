"""テスト: cli.py — Stage 1 CLI エントリポイント (CQ-0051)"""
import pytest
from pathlib import Path
from unittest.mock import patch

pytestmark = pytest.mark.smoke

from mahjong_rl.cli import main, _apply_override, _parse_value
from mahjong_rl.experiment import ExperimentConfig


class TestCLIParsing:
    """CLI 引数パースのテスト"""

    def test_missing_config_returns_error(self):
        """--config なしはエラー"""
        with pytest.raises(SystemExit):
            main(["--config"])  # 引数不足

    def test_nonexistent_config_returns_1(self, capsys):
        """存在しない config ファイルでエラーコード 1"""
        ret = main(["--config", "/tmp/nonexistent_config_12345.yaml"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "見つかりません" in captured.err

    def test_valid_config_runs(self, tmp_path: Path):
        """有効な config で run が完走する"""
        config = ExperimentConfig(
            experiment={"name": "cli_test", "stage": 1, "observation_mode": "full"},
            feature_encoder={"name": "FlatFeatureEncoder", "observation_mode": "full"},
            model={"name": "MLPPolicyValueModel", "hidden_dims": [32],
                   "value_heads": ["round_delta"]},
            reward={"type": "point_delta"},
            selfplay={"num_matches": 1, "policy_ratio": 1.0, "temperature": 1.0,
                      "max_samples_per_shard": 10000},
            training={"algorithm": "ppo", "lr": 0.001, "batch_size": 32,
                      "epochs": 1, "gamma": 0.99, "gae_lambda": 0.95,
                      "clip_epsilon": 0.2, "value_loss_coef": 0.5,
                      "entropy_coef": 0.01, "max_grad_norm": 0.5},
            evaluation={"num_matches": 1, "seed_start": 100000},
        )
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)

        ret = main(["--config", str(yaml_path), "--base-dir", str(tmp_path / "runs")])
        assert ret == 0

    def test_override_applies(self, tmp_path: Path):
        """--override で config 値を上書きできる"""
        config = ExperimentConfig(
            experiment={"name": "cli_test", "stage": 1, "observation_mode": "full",
                        "global_seed": 99},
            feature_encoder={"name": "FlatFeatureEncoder", "observation_mode": "full"},
            model={"name": "MLPPolicyValueModel", "hidden_dims": [32],
                   "value_heads": ["round_delta"]},
            reward={"type": "point_delta"},
            selfplay={"num_matches": 1, "policy_ratio": 1.0, "temperature": 1.0,
                      "max_samples_per_shard": 10000},
            training={"algorithm": "ppo", "lr": 0.001, "batch_size": 32,
                      "epochs": 1, "gamma": 0.99, "gae_lambda": 0.95,
                      "clip_epsilon": 0.2, "value_loss_coef": 0.5,
                      "entropy_coef": 0.01, "max_grad_norm": 0.5},
            evaluation={"num_matches": 1, "seed_start": 100000},
        )
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)

        ret = main([
            "--config", str(yaml_path),
            "--base-dir", str(tmp_path / "runs"),
            "--override", "experiment.global_seed=42",
        ])
        assert ret == 0


class TestParseValue:
    """_parse_value のテスト"""

    def test_int(self):
        assert _parse_value("42") == 42

    def test_float(self):
        assert _parse_value("3.14") == 3.14

    def test_bool(self):
        assert _parse_value("true") is True
        assert _parse_value("false") is False

    def test_list(self):
        assert _parse_value("[1, 2, 3]") == [1, 2, 3]

    def test_string(self):
        assert _parse_value("hello") == "hello"


class TestApplyOverride:
    """_apply_override のテスト"""

    def test_override_existing_key(self):
        config = ExperimentConfig(
            experiment={"name": "test", "global_seed": 0},
        )
        _apply_override(config, "experiment.global_seed", "42")
        assert config.experiment["global_seed"] == 42

    def test_override_invalid_section(self):
        config = ExperimentConfig()
        with pytest.raises(ValueError, match="不正な config セクション"):
            _apply_override(config, "nonexistent.key", "value")

    def test_override_invalid_key_format(self):
        config = ExperimentConfig()
        with pytest.raises(ValueError, match="section.key"):
            _apply_override(config, "only_one_part", "value")


class TestCLIFailureExitCodes:
    """CLI 失敗系の終了コード管理テスト (CQ-0059)"""

    def test_invalid_override_section_returns_1(self, tmp_path: Path, capsys):
        """不正な override セクションで例外終了せず終了コード 1"""
        config = ExperimentConfig(
            experiment={"name": "cli_test", "stage": 1, "observation_mode": "full"},
            feature_encoder={"name": "FlatFeatureEncoder", "observation_mode": "full"},
            model={"name": "MLPPolicyValueModel", "hidden_dims": [32],
                   "value_heads": ["round_delta"]},
            reward={"type": "point_delta"},
            selfplay={"num_matches": 1, "policy_ratio": 1.0, "temperature": 1.0,
                      "max_samples_per_shard": 10000},
            training={"algorithm": "ppo", "lr": 0.001, "batch_size": 32,
                      "epochs": 1, "gamma": 0.99, "gae_lambda": 0.95,
                      "clip_epsilon": 0.2, "value_loss_coef": 0.5,
                      "entropy_coef": 0.01, "max_grad_norm": 0.5},
            evaluation={"num_matches": 1, "seed_start": 100000},
        )
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)

        ret = main([
            "--config", str(yaml_path),
            "--base-dir", str(tmp_path / "runs"),
            "--override", "badsection.x=1",
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "override 適用失敗" in captured.err

    def test_invalid_override_key_format_returns_1(self, tmp_path: Path, capsys):
        """不正なキー形式の override で終了コード 1"""
        config = ExperimentConfig(
            experiment={"name": "cli_test", "stage": 1, "observation_mode": "full"},
            feature_encoder={"name": "FlatFeatureEncoder", "observation_mode": "full"},
            model={"name": "MLPPolicyValueModel", "hidden_dims": [32],
                   "value_heads": ["round_delta"]},
            reward={"type": "point_delta"},
            selfplay={"num_matches": 1, "policy_ratio": 1.0, "temperature": 1.0,
                      "max_samples_per_shard": 10000},
            training={"algorithm": "ppo", "lr": 0.001, "batch_size": 32,
                      "epochs": 1, "gamma": 0.99, "gae_lambda": 0.95,
                      "clip_epsilon": 0.2, "value_loss_coef": 0.5,
                      "entropy_coef": 0.01, "max_grad_norm": 0.5},
            evaluation={"num_matches": 1, "seed_start": 100000},
        )
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)

        ret = main([
            "--config", str(yaml_path),
            "--override", "no_dot_key=value",
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "override 適用失敗" in captured.err

    def test_validation_error_returns_1(self, tmp_path: Path, capsys):
        """config バリデーション失敗で終了コード 1"""
        config = ExperimentConfig(
            experiment={"name": "cli_test", "stage": 1, "observation_mode": "full",
                        "phases": ["bad_phase"]},
            feature_encoder={"name": "FlatFeatureEncoder", "observation_mode": "full"},
            model={"name": "MLPPolicyValueModel", "hidden_dims": [32],
                   "value_heads": ["round_delta"]},
            reward={"type": "point_delta"},
            selfplay={"num_matches": 1, "policy_ratio": 1.0, "temperature": 1.0,
                      "max_samples_per_shard": 10000},
            training={"algorithm": "ppo", "lr": 0.001, "batch_size": 32,
                      "epochs": 1, "gamma": 0.99, "gae_lambda": 0.95,
                      "clip_epsilon": 0.2, "value_loss_coef": 0.5,
                      "entropy_coef": 0.01, "max_grad_norm": 0.5},
            evaluation={"num_matches": 1, "seed_start": 100000},
        )
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)

        ret = main([
            "--config", str(yaml_path),
            "--base-dir", str(tmp_path / "runs"),
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "バリデーション" in captured.err
