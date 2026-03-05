"""テスト: cli.py — Stage 1 CLI エントリポイント (CQ-0051)"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch

pytestmark = pytest.mark.smoke

from mahjong_rl.cli import (
    main, _apply_override, _parse_value, run_batch, _resolve_seeds,
    run_batch_resume, _detect_completed_seeds, run_sweep,
    run_sweep_resume, _find_batch_dir,
)
from mahjong_rl.experiment import ExperimentConfig
from mahjong_rl.runner import Stage1Runner


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


class TestValidateOnly:
    """--validate-only テスト (CQ-0071)"""

    def _write_config(self, tmp_path: Path, **overrides) -> Path:
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
        for k, v in overrides.items():
            parts = k.split(".")
            getattr(config, parts[0])[parts[1]] = v
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)
        return yaml_path

    def test_validate_only_success(self, tmp_path: Path, capsys):
        """有効な config で --validate-only は 0 を返す"""
        yaml_path = self._write_config(tmp_path)
        ret = main(["--config", str(yaml_path), "--validate-only"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "OK" in captured.out

    def test_validate_only_no_run_dir(self, tmp_path: Path):
        """--validate-only で run ディレクトリが生成されない"""
        yaml_path = self._write_config(tmp_path)
        runs_dir = tmp_path / "runs"
        main(["--config", str(yaml_path), "--validate-only",
              "--base-dir", str(runs_dir)])
        assert not runs_dir.exists()

    def test_validate_only_invalid_config(self, tmp_path: Path, capsys):
        """不正な config で --validate-only は 1 を返す"""
        yaml_path = self._write_config(
            tmp_path, **{"experiment.phases": ["bad_phase"]})
        # phases は list なので直接 override ではなく config を書き換え
        config = ExperimentConfig.from_yaml(yaml_path)
        config.experiment["phases"] = ["bad_phase"]
        config.to_yaml(yaml_path)
        ret = main(["--config", str(yaml_path), "--validate-only"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "バリデーションエラー" in captured.err

    def test_validate_only_invalid_device(self, tmp_path: Path, capsys):
        """不正なデバイス指定を --validate-only で検出できる"""
        yaml_path = self._write_config(tmp_path)
        config = ExperimentConfig.from_yaml(yaml_path)
        config.training["device"] = "tpu"
        config.to_yaml(yaml_path)
        ret = main(["--config", str(yaml_path), "--validate-only"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "training.device" in captured.err


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


def _make_cli_config() -> ExperimentConfig:
    """CLI テスト用の最小構成"""
    return ExperimentConfig(
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


class TestResolveSeed:
    """seed リスト構築テスト (CQ-0077)"""

    def test_no_seeds(self):
        """seed 未指定で None を返す"""
        from argparse import Namespace
        args = Namespace(seeds=None, seed_start=None, num_seeds=None)
        seeds, err = _resolve_seeds(args)
        assert seeds is None
        assert err is None

    def test_seeds_list(self):
        """--seeds で seed リストを返す"""
        from argparse import Namespace
        args = Namespace(seeds="42,43,44", seed_start=None, num_seeds=None)
        seeds, err = _resolve_seeds(args)
        assert seeds == [42, 43, 44]
        assert err is None

    def test_seed_range(self):
        """--seed-start + --num-seeds で seed 範囲を返す"""
        from argparse import Namespace
        args = Namespace(seeds=None, seed_start=10, num_seeds=3)
        seeds, err = _resolve_seeds(args)
        assert seeds == [10, 11, 12]
        assert err is None

    def test_conflict(self):
        """--seeds と --seed-start の同時指定はエラー"""
        from argparse import Namespace
        args = Namespace(seeds="42,43", seed_start=10, num_seeds=3)
        seeds, err = _resolve_seeds(args)
        assert seeds is None
        assert err is not None
        assert "同時" in err

    def test_seed_start_without_num_seeds(self):
        """--seed-start のみはエラー"""
        from argparse import Namespace
        args = Namespace(seeds=None, seed_start=10, num_seeds=None)
        seeds, err = _resolve_seeds(args)
        assert err is not None
        assert "両方" in err


class TestBatchExecution:
    """マルチ seed バッチ実行テスト (CQ-0077, CQ-0080)"""

    def test_batch_seeds_creates_run_dirs(self, tmp_path: Path):
        """--seeds で seed ごとの run_dir が作られる"""
        config = _make_cli_config()
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        ret = main([
            "--config", str(yaml_path),
            "--base-dir", str(tmp_path / "runs"),
            "--seeds", "42,43",
        ])
        assert ret == 0

        # batch_dir が作られている
        runs_dir = tmp_path / "runs"
        batch_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "batch" in d.name]
        assert len(batch_dirs) == 1
        batch_dir = batch_dirs[0]

        # batch_dir 内に run_dir が 2 つある
        run_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
        assert len(run_dirs) == 2

    def test_batch_seed_range(self, tmp_path: Path):
        """--seed-start + --num-seeds で動作する"""
        config = _make_cli_config()
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        ret = main([
            "--config", str(yaml_path),
            "--base-dir", str(tmp_path / "runs"),
            "--seed-start", "10",
            "--num-seeds", "2",
        ])
        assert ret == 0

    def test_batch_generates_report(self, tmp_path: Path):
        """batch_summary.json と batch_table.csv が出力される"""
        config = _make_cli_config()
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        main([
            "--config", str(yaml_path),
            "--base-dir", str(tmp_path / "runs"),
            "--seeds", "42,43",
        ])

        runs_dir = tmp_path / "runs"
        batch_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "batch" in d.name]
        batch_dir = batch_dirs[0]

        assert (batch_dir / "batch_summary.json").exists()
        assert (batch_dir / "batch_table.csv").exists()

        with open(batch_dir / "batch_summary.json") as f:
            summary = json.load(f)
        assert summary["num_seeds"] == 2
        assert summary["success_count"] == 2

    def test_batch_continue_on_error(self, tmp_path: Path):
        """--continue-on-error でエラー後も続行する"""
        config = _make_cli_config()
        # 不正な selfplay 設定で失敗させる
        config.selfplay["num_matches"] = 0  # 0 は失敗を引き起こす可能性がある
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        # run_batch を直接呼んでテスト（mock で失敗させる）
        from mahjong_rl.cli import run_batch
        import copy

        call_count = 0
        original_run = Stage1Runner.run

        def mock_run(self):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("テスト用エラー")
            return original_run(self)

        good_config = _make_cli_config()
        with patch.object(Stage1Runner, "run", mock_run):
            ret = run_batch(
                good_config, [42, 43],
                tmp_path / "batch_out",
                stop_on_error=False,
            )

        assert ret == 1  # 1 つ失敗あり
        # 2 つとも実行された
        assert call_count == 2

    def test_batch_stop_on_error(self, tmp_path: Path):
        """--stop-on-error（デフォルト）でエラー時に停止する"""
        call_count = 0
        original_run = Stage1Runner.run

        def mock_run(self):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("テスト用エラー")

        config = _make_cli_config()
        with patch.object(Stage1Runner, "run", mock_run):
            ret = run_batch(
                config, [42, 43, 44],
                tmp_path / "batch_out",
                stop_on_error=True,
            )

        assert ret == 1
        # 最初の 1 つでエラー停止
        assert call_count == 1

    def test_seeds_and_seed_start_conflict(self, tmp_path: Path, capsys):
        """--seeds と --seed-start の同時指定でエラー"""
        config = _make_cli_config()
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        ret = main([
            "--config", str(yaml_path),
            "--seeds", "42,43",
            "--seed-start", "10",
            "--num-seeds", "2",
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert "同時" in captured.err


class TestBatchResume:
    """バッチ再開テスト (CQ-0096)"""

    def test_resume_skips_completed_seeds(self, tmp_path: Path):
        """完了済み seed がスキップされる"""
        config = _make_cli_config()

        # まず seed=42 のみバッチ実行
        ret = run_batch(config, [42], tmp_path / "batch_out", stop_on_error=True)
        assert ret == 0

        batch_dir = list((tmp_path / "batch_out").iterdir())[0]

        # seed=42 と seed=43 で resume
        call_count = 0
        original_run = Stage1Runner.run

        def counting_run(self):
            nonlocal call_count
            call_count += 1
            return original_run(self)

        with patch.object(Stage1Runner, "run", counting_run):
            ret = run_batch_resume(config, [42, 43], batch_dir,
                                   stop_on_error=True)

        assert ret == 0
        # seed=42 はスキップされ、seed=43 のみ実行
        assert call_count == 1

    def test_resume_regenerates_report(self, tmp_path: Path):
        """resume 後に全 seed 分のレポートが再生成される"""
        config = _make_cli_config()

        # seed=42 のみ実行
        ret = run_batch(config, [42], tmp_path / "batch_out", stop_on_error=True)
        batch_dir = list((tmp_path / "batch_out").iterdir())[0]

        # seed=42,43 で resume
        ret = run_batch_resume(config, [42, 43], batch_dir,
                               stop_on_error=True)
        assert ret == 0

        with open(batch_dir / "batch_summary.json") as f:
            summary = json.load(f)
        assert summary["num_seeds"] == 2
        assert summary["success_count"] == 2

    def test_resume_nonexistent_dir(self, tmp_path: Path):
        """存在しない batch_dir でエラー"""
        config = _make_cli_config()
        ret = run_batch_resume(config, [42], tmp_path / "nonexistent",
                               stop_on_error=True)
        assert ret == 1


class TestSweepExecution:
    """sweep 実行テスト (CQ-0095)"""

    def test_sweep_creates_condition_dirs(self, tmp_path: Path):
        """2 条件 sweep で condition_id 別のディレクトリが作られる"""
        config = _make_cli_config()
        sweep_config = {
            "conditions": [
                {"condition_id": "cond_a", "overrides": {"training.lr": 0.001}},
                {"condition_id": "cond_b", "overrides": {"training.lr": 0.0001}},
            ]
        }

        ret = run_sweep(config, [42], sweep_config,
                         tmp_path / "runs", stop_on_error=True)
        assert ret == 0

        sweep_dirs = [d for d in (tmp_path / "runs").iterdir()
                      if d.is_dir() and "sweep" in d.name]
        assert len(sweep_dirs) == 1
        sweep_dir = sweep_dirs[0]

        assert (sweep_dir / "cond_a").exists()
        assert (sweep_dir / "cond_b").exists()

    def test_sweep_ranking_output(self, tmp_path: Path):
        """sweep_ranking.json と sweep_ranking.csv が出力される"""
        config = _make_cli_config()
        sweep_config = {
            "conditions": [
                {"condition_id": "cond_a", "overrides": {"training.lr": 0.001}},
                {"condition_id": "cond_b", "overrides": {"training.lr": 0.0001}},
            ]
        }

        run_sweep(config, [42], sweep_config,
                  tmp_path / "runs", stop_on_error=True)

        sweep_dirs = [d for d in (tmp_path / "runs").iterdir()
                      if d.is_dir() and "sweep" in d.name]
        sweep_dir = sweep_dirs[0]

        assert (sweep_dir / "sweep_ranking.json").exists()
        assert (sweep_dir / "sweep_ranking.csv").exists()

        with open(sweep_dir / "sweep_ranking.json") as f:
            ranking = json.load(f)
        assert len(ranking["conditions"]) == 2

    def test_sweep_ranking_sorted(self, tmp_path: Path):
        """ランキングが avg_rank_mean 昇順"""
        config = _make_cli_config()
        sweep_config = {
            "conditions": [
                {"condition_id": "cond_a", "overrides": {"training.lr": 0.001}},
                {"condition_id": "cond_b", "overrides": {"training.lr": 0.0001}},
            ]
        }

        run_sweep(config, [42], sweep_config,
                  tmp_path / "runs", stop_on_error=True)

        sweep_dirs = [d for d in (tmp_path / "runs").iterdir()
                      if d.is_dir() and "sweep" in d.name]
        sweep_dir = sweep_dirs[0]

        with open(sweep_dir / "sweep_ranking.json") as f:
            ranking = json.load(f)
        means = [c.get("avg_rank_mean", float("inf"))
                 for c in ranking["conditions"]]
        assert means == sorted(means)

    def test_sweep_file_validation(self, tmp_path: Path, capsys):
        """不正な sweep ファイルでエラー"""
        config = _make_cli_config()
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        # conditions キーがない YAML
        bad_sweep = tmp_path / "bad_sweep.yaml"
        bad_sweep.write_text("foo: bar\n")

        ret = main([
            "--config", str(yaml_path),
            "--sweep-file", str(bad_sweep),
            "--seeds", "42",
        ])
        assert ret == 1


class TestSweepResume:
    """sweep condition 単位 resume テスト (CQ-0101)"""

    def _run_sweep_and_get_dir(self, config, tmp_path):
        """ヘルパー: 2 条件 sweep を実行し sweep_dir を返す"""
        sweep_config = {
            "conditions": [
                {"condition_id": "cond_a", "overrides": {"training.lr": 0.001}},
                {"condition_id": "cond_b", "overrides": {"training.lr": 0.0001}},
            ]
        }
        ret = run_sweep(config, [42], sweep_config,
                        tmp_path / "runs", stop_on_error=True)
        assert ret == 0
        sweep_dirs = [d for d in (tmp_path / "runs").iterdir()
                      if d.is_dir() and "sweep" in d.name]
        assert len(sweep_dirs) == 1
        return sweep_dirs[0], sweep_config

    def test_sweep_resume_skips_completed(self, tmp_path: Path):
        """完了条件内の完了 seed がスキップされる"""
        config = _make_cli_config()
        sweep_dir, sweep_config = self._run_sweep_and_get_dir(config, tmp_path)

        # seed=42 は全条件で完了済み → resume で seed=42 はスキップ
        call_count = 0
        original_run = Stage1Runner.run

        def counting_run(self):
            nonlocal call_count
            call_count += 1
            return original_run(self)

        with patch.object(Stage1Runner, "run", counting_run):
            ret = run_sweep_resume(config, [42, 43], sweep_config,
                                   sweep_dir, stop_on_error=True)

        assert ret == 0
        # cond_a: seed=42 スキップ + seed=43 実行 = 1 回
        # cond_b: seed=42 スキップ + seed=43 実行 = 1 回
        assert call_count == 2

    def test_sweep_resume_new_condition(self, tmp_path: Path):
        """sweep_dir にない条件は新規実行される"""
        config = _make_cli_config()
        sweep_dir, _ = self._run_sweep_and_get_dir(config, tmp_path)

        # cond_a は既存、cond_c は新規
        new_sweep_config = {
            "conditions": [
                {"condition_id": "cond_a", "overrides": {"training.lr": 0.001}},
                {"condition_id": "cond_c", "overrides": {"training.lr": 0.01}},
            ]
        }
        ret = run_sweep_resume(config, [42], new_sweep_config,
                               sweep_dir, stop_on_error=True)
        assert ret == 0
        # cond_c が新規作成される
        assert (sweep_dir / "cond_c").exists()

    def test_sweep_resume_ranking_regenerated(self, tmp_path: Path):
        """resume 後に sweep_ranking.json が再生成される"""
        config = _make_cli_config()
        sweep_dir, sweep_config = self._run_sweep_and_get_dir(config, tmp_path)

        # 既存ランキングを削除して再生成を確認
        ranking_path = sweep_dir / "sweep_ranking.json"
        assert ranking_path.exists()
        ranking_path.unlink()

        ret = run_sweep_resume(config, [42], sweep_config,
                               sweep_dir, stop_on_error=True)
        assert ret == 0
        assert ranking_path.exists()

        with open(ranking_path) as f:
            ranking = json.load(f)
        assert len(ranking["conditions"]) == 2
