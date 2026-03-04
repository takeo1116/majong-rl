"""テスト: runner.py — Stage 1 統合ランナー"""
import json
import pytest
import torch
from pathlib import Path
from unittest.mock import patch

from mahjong_rl.experiment import ExperimentConfig
from mahjong_rl.runner import Stage1Runner, resolve_device


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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
class TestGlobalSeed:
    """global seed テスト (CQ-0048)"""

    def test_seed_in_result(self, tmp_path: Path):
        """結果に global_seed が含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "global_seed" in result
        assert isinstance(result["global_seed"], int)

    def test_explicit_seed(self, tmp_path: Path):
        """config で指定した seed が使われる"""
        config = _make_minimal_config()
        config.experiment["global_seed"] = 12345
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert result["global_seed"] == 12345

    def test_auto_seed_differs(self, tmp_path: Path):
        """seed 未指定時は自動生成される（異なる値になりうる）"""
        config = _make_minimal_config()
        runner1 = Stage1Runner(config=config, base_dir=tmp_path / "r1")
        result1 = runner1.run()
        runner2 = Stage1Runner(config=config, base_dir=tmp_path / "r2")
        result2 = runner2.run()

        # 自動生成なので異なる値になる（極めて低確率で一致する可能性はある）
        assert isinstance(result1["global_seed"], int)
        assert isinstance(result2["global_seed"], int)


@pytest.mark.slow
class TestEvalMode:
    """eval mode テスト (CQ-0049)"""

    def test_default_single_mode(self, tmp_path: Path):
        """デフォルトは single モード"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert result["eval_metrics"]["eval_mode"] == "single"
        assert 1.0 <= result["eval_metrics"]["avg_rank"] <= 4.0

    def test_rotation_mode(self, tmp_path: Path):
        """rotation モードで評価できる"""
        config = _make_minimal_config()
        config.evaluation["mode"] = "rotation"
        config.evaluation["rotation_seats"] = [0, 1]
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert result["eval_metrics"]["eval_mode"] == "rotation"
        assert result["eval_metrics"]["rotation_seats"] == [0, 1]
        assert 1.0 <= result["eval_metrics"]["avg_rank"] <= 4.0

    def test_rotation_saves_per_seat_files(self, tmp_path: Path):
        """rotation 評価で席別ファイルが保存される"""
        config = _make_minimal_config()
        config.evaluation["mode"] = "rotation"
        config.evaluation["rotation_seats"] = [0, 1]
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        eval_dir = run_dir / "eval"
        assert (eval_dir / "eval_seat0.json").exists()
        assert (eval_dir / "eval_seat1.json").exists()
        assert (eval_dir / "eval_rotation.json").exists()


@pytest.mark.slow
class TestRunLog:
    """run.log / summary.json テスト (CQ-0050)"""

    def test_run_log_created(self, tmp_path: Path):
        """run.log が保存される"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        log_path = run_dir / "run.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert len(content) > 0

    def test_summary_json_created(self, tmp_path: Path):
        """summary.json が保存される"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        summary_path = run_dir / "summary.json"
        assert summary_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["success"] is True
        assert summary["error"] is None
        assert summary["phases"] == ["selfplay", "learner", "eval"]
        assert summary["global_seed"] == result["global_seed"]

    def test_summary_contains_phase_status(self, tmp_path: Path):
        """summary にフェーズ別の成否が記録される"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        assert summary["phase_status"]["selfplay"] == "success"
        assert summary["phase_status"]["learner"] == "success"
        assert summary["phase_status"]["eval"] == "success"

    def test_summary_contains_artifact_info(self, tmp_path: Path):
        """summary に shard 数・checkpoint・eval 情報がある"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        assert summary["shard_count"] >= 1
        assert summary["has_checkpoint"] is True
        assert summary["has_eval"] is True

    def test_summary_on_partial_run(self, tmp_path: Path):
        """一部フェーズのみでも summary が保存される"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["selfplay"]
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        assert summary["success"] is True
        assert summary["has_checkpoint"] is False
        assert summary["has_eval"] is False
        assert summary["shard_count"] >= 1


@pytest.mark.smoke
class TestConfigValidation:
    """config バリデーションテスト (CQ-0052)"""

    def test_valid_config_passes(self):
        """有効な config はエラーなし"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert errors == []

    def test_invalid_phase(self):
        """不正なフェーズ名でエラー"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["selfplay", "invalid_phase"]
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("不正なフェーズ" in e for e in errors)

    def test_imitation_without_selfplay(self):
        """imitation があるのに selfplay がないとエラー"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["imitation", "learner"]
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("selfplay フェーズがありません" in e for e in errors)

    def test_invalid_eval_mode(self):
        """不正な evaluation.mode でエラー"""
        config = _make_minimal_config()
        config.evaluation["mode"] = "invalid"
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("evaluation.mode" in e for e in errors)

    def test_invalid_observation_mode(self):
        """不正な observation_mode でエラー"""
        config = _make_minimal_config()
        config.experiment["observation_mode"] = "unknown"
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("observation_mode" in e for e in errors)

    def test_invalid_encoder(self):
        """不正な encoder 名でエラー"""
        config = _make_minimal_config()
        config.feature_encoder["name"] = "UnknownEncoder"
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("encoder" in e for e in errors)

    def test_invalid_model(self):
        """不正な model 名でエラー"""
        config = _make_minimal_config()
        config.model["name"] = "UnknownModel"
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("model" in e for e in errors)

    def test_invalid_seed_type(self):
        """global_seed が文字列だとエラー"""
        config = _make_minimal_config()
        config.experiment["global_seed"] = "not_a_number"
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("global_seed" in e for e in errors)

    def test_validation_raises_on_run(self, tmp_path: Path):
        """バリデーション失敗時に run() が ValueError を投げる"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["bad_phase"]
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        with pytest.raises(ValueError, match="バリデーションエラー"):
            runner.run()


@pytest.mark.slow
class TestPhaseStats:
    """フェーズ別統計テスト (CQ-0053)"""

    def test_summary_has_phase_stats(self, tmp_path: Path):
        """summary.json に phase_stats が含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        assert "phase_stats" in summary
        ps = summary["phase_stats"]
        assert "selfplay" in ps
        assert "learner" in ps
        assert "eval" in ps

    def test_selfplay_stats_in_summary(self, tmp_path: Path):
        """selfplay フェーズ統計が記録される"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        sp = summary["phase_stats"]["selfplay"]
        assert sp["total_steps"] > 0
        assert sp["shard_count"] >= 1

    def test_eval_stats_in_summary(self, tmp_path: Path):
        """eval フェーズ統計が記録される"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        ev = summary["phase_stats"]["eval"]
        assert 1.0 <= ev["avg_rank"] <= 4.0
        assert ev["eval_mode"] == "single"

    def test_actor_type_counts_in_summary(self, tmp_path: Path):
        """actor_type 内訳が summary に含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        assert "actor_type_counts" in summary
        # policy_ratio=1.0 なので policy のみ
        counts = summary["actor_type_counts"]
        assert isinstance(counts, dict)


@pytest.mark.slow
class TestNotesAppend:
    """notes.md 追記テスト (CQ-0057)"""

    def test_notes_has_execution_results(self, tmp_path: Path):
        """notes.md に実行結果セクションが追記される"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        notes = (run_dir / "notes.md").read_text()
        assert "## 実行結果" in notes
        assert "成功" in notes

    def test_notes_has_seed_info(self, tmp_path: Path):
        """notes.md に seed 情報が含まれる"""
        config = _make_minimal_config()
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        notes = (run_dir / "notes.md").read_text()
        assert "global_seed: 42" in notes

    def test_notes_has_eval_metrics(self, tmp_path: Path):
        """notes.md に主要指標が含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        notes = (run_dir / "notes.md").read_text()
        assert "## 主要指標" in notes
        assert "avg_rank" in notes
        assert "win_rate" in notes

    def test_notes_has_checkpoint_info(self, tmp_path: Path):
        """notes.md に checkpoint 情報が含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        notes = (run_dir / "notes.md").read_text()
        assert "checkpoint:" in notes

    def test_notes_preserves_memo(self, tmp_path: Path):
        """notes.md にメモと実行結果の両方が残る"""
        config = _make_minimal_config()
        config.experiment["memo"] = "テスト実験メモ"
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        notes = (run_dir / "notes.md").read_text()
        # メモが先頭にある
        assert "テスト実験メモ" in notes
        # 実行結果が追記されている
        assert "## 実行結果" in notes


@pytest.mark.smoke
class TestPresetConfigs:
    """プリセット config テスト (CQ-0055)"""

    def test_ppo_preset_loads(self):
        """PPO プリセットが読み込める"""
        config = ExperimentConfig.from_yaml(
            Path("configs/stage1_full_flat_mlp_ppo.yaml"))
        assert config.experiment["name"] == "stage1_full_flat_mlp_ppo"
        assert config.experiment["observation_mode"] == "full"
        assert config.training["algorithm"] == "ppo"

    def test_imitation_then_ppo_preset_loads(self):
        """Imitation→PPO プリセットが読み込める"""
        config = ExperimentConfig.from_yaml(
            Path("configs/stage1_full_flat_mlp_imitation_then_ppo.yaml"))
        assert config.experiment["name"] == "stage1_full_flat_mlp_imitation_then_ppo"
        assert "imitation" in config.experiment["phases"]
        assert "selfplay" in config.experiment["phases"]

    def test_rotation_eval_preset_loads(self):
        """Rotation 評価プリセットが読み込める"""
        config = ExperimentConfig.from_yaml(
            Path("configs/stage1_full_flat_mlp_ppo_rotation_eval.yaml"))
        assert config.experiment["name"] == "stage1_full_flat_mlp_ppo_rotation_eval"
        assert config.evaluation["mode"] == "rotation"
        assert config.evaluation["rotation_seats"] == [0, 1, 2, 3]

    def test_all_presets_pass_validation(self):
        """全プリセットがバリデーションを通る"""
        import glob
        preset_files = glob.glob("configs/stage1_*.yaml")
        assert len(preset_files) >= 3

        for path in preset_files:
            config = ExperimentConfig.from_yaml(Path(path))
            runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
            errors = runner.validate_config()
            assert errors == [], f"{path}: {errors}"


@pytest.mark.smoke
class TestSeedRangeValidation:
    """seed 値域バリデーションテスト (CQ-0060)"""

    def test_negative_seed_rejected(self):
        """負の global_seed でエラー"""
        config = _make_minimal_config()
        config.experiment["global_seed"] = -1
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("global_seed" in e and "範囲" in e for e in errors)

    def test_too_large_seed_rejected(self):
        """2**32 以上の global_seed でエラー"""
        config = _make_minimal_config()
        config.experiment["global_seed"] = 2**32
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("global_seed" in e and "範囲" in e for e in errors)

    def test_max_valid_seed_passes(self):
        """2**32 - 1 の global_seed は有効"""
        config = _make_minimal_config()
        config.experiment["global_seed"] = 2**32 - 1
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert errors == []

    def test_zero_seed_passes(self):
        """0 の global_seed は有効"""
        config = _make_minimal_config()
        config.experiment["global_seed"] = 0
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert errors == []

    def test_negative_selfplay_seed_start_rejected(self):
        """負の selfplay.seed_start でエラー"""
        config = _make_minimal_config()
        config.selfplay["seed_start"] = -10
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("selfplay.seed_start" in e for e in errors)

    def test_negative_eval_seed_start_rejected(self):
        """負の evaluation.seed_start でエラー"""
        config = _make_minimal_config()
        config.evaluation["seed_start"] = -1
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("evaluation.seed_start" in e for e in errors)


@pytest.mark.smoke
class TestDeviceValidation:
    """デバイスバリデーションテスト (CQ-0062)"""

    def test_auto_device_passes(self):
        """auto デバイスがバリデーション通過"""
        config = _make_minimal_config()
        config.training["device"] = "auto"
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert errors == []

    def test_cpu_device_passes(self):
        """cpu デバイスがバリデーション通過"""
        config = _make_minimal_config()
        config.training["device"] = "cpu"
        config.selfplay["inference_device"] = "cpu"
        config.evaluation["inference_device"] = "cpu"
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert errors == []

    def test_invalid_device_rejected(self):
        """不正デバイス値でエラー"""
        config = _make_minimal_config()
        config.training["device"] = "tpu"
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("training.device" in e for e in errors)

    def test_default_device_passes(self):
        """デバイス未指定でバリデーション通過"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert errors == []

    def test_cuda_is_valid_value(self):
        """cuda はバリデーション上は有効値"""
        config = _make_minimal_config()
        config.training["device"] = "cuda"
        config.selfplay["inference_device"] = "cuda"
        config.evaluation["inference_device"] = "cuda"
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        # cuda はバリデーション自体は通る（実行時に利用可否を判定）
        assert errors == []

    def test_invalid_inference_device_rejected(self):
        """selfplay/eval の不正デバイスでエラー"""
        config = _make_minimal_config()
        config.selfplay["inference_device"] = "invalid"
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("selfplay.inference_device" in e for e in errors)


@pytest.mark.smoke
class TestCudaDeviceErrorPath:
    """CUDA 利用不可時のエラー経路テスト (CQ-0067)"""

    def test_resolve_device_cuda_unavailable_raises(self):
        """CUDA 不可時に resolve_device('cuda') が RuntimeError を送出する"""
        with patch("mahjong_rl.runner.torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="cuda"):
                resolve_device("cuda")

    def test_resolve_device_cuda_unavailable_message_readable(self):
        """エラーメッセージが利用者に判読可能"""
        with patch("mahjong_rl.runner.torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError) as exc_info:
                resolve_device("cuda")
            msg = str(exc_info.value)
            assert "cuda" in msg
            assert "利用できません" in msg

    def test_resolve_device_auto_falls_back_to_cpu(self):
        """CUDA 不可時に auto は CPU にフォールバックする"""
        with patch("mahjong_rl.runner.torch.cuda.is_available", return_value=False):
            device = resolve_device("auto")
            assert device == torch.device("cpu")

    def test_resolve_device_cpu_always_succeeds(self):
        """CPU 指定は CUDA 有無にかかわらず成功する"""
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 利用不可")
    def test_resolve_device_cuda_available_succeeds(self):
        """CUDA 利用可能環境では cuda 指定が成功する"""
        device = resolve_device("cuda")
        assert device == torch.device("cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 利用不可")
    def test_resolve_device_auto_selects_cuda(self):
        """CUDA 利用可能環境では auto が cuda を選択する"""
        device = resolve_device("auto")
        assert device.type == "cuda"

    def test_runner_run_fails_with_cuda_training_device(self, tmp_path: Path):
        """runner で training.device=cuda 指定が CUDA 不可時にエラーになる"""
        config = _make_minimal_config()
        config.training["device"] = "cuda"
        runner = Stage1Runner(config=config, base_dir=tmp_path)

        with patch("mahjong_rl.runner.torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="cuda"):
                runner.run()

    def test_runner_run_fails_with_cuda_selfplay_device(self, tmp_path: Path):
        """runner で selfplay.inference_device=cuda が CUDA 不可時にエラーになる"""
        config = _make_minimal_config()
        config.selfplay["inference_device"] = "cuda"
        runner = Stage1Runner(config=config, base_dir=tmp_path)

        with patch("mahjong_rl.runner.torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="cuda"):
                runner.run()

    def test_runner_run_fails_with_cuda_eval_device(self, tmp_path: Path):
        """runner で evaluation.inference_device=cuda が CUDA 不可時にエラーになる"""
        config = _make_minimal_config()
        config.evaluation["inference_device"] = "cuda"
        runner = Stage1Runner(config=config, base_dir=tmp_path)

        with patch("mahjong_rl.runner.torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="cuda"):
                runner.run()


@pytest.mark.slow
class TestDeviceEnvInfo:
    """device/環境情報記録テスト (CQ-0066)"""

    def test_summary_has_device_info(self, tmp_path: Path):
        """summary.json に device_info が含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        assert "device_info" in summary
        di = summary["device_info"]
        assert "training" in di
        assert "selfplay" in di
        assert "evaluation" in di
        assert "requested" in di["training"]
        assert "resolved" in di["training"]

    def test_summary_has_env_info(self, tmp_path: Path):
        """summary.json に env_info が含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        assert "env_info" in summary
        ei = summary["env_info"]
        assert isinstance(ei["torch_version"], str)
        assert isinstance(ei["cuda_available"], bool)

    def test_resolved_devices_in_result(self, tmp_path: Path):
        """result に resolved_devices が含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "resolved_devices" in result
        rd = result["resolved_devices"]
        assert "training" in rd
        assert "selfplay" in rd
        assert "evaluation" in rd

    def test_notes_has_device_info(self, tmp_path: Path):
        """notes.md に device 情報が含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        notes = (run_dir / "notes.md").read_text()
        assert "devices:" in notes


@pytest.mark.slow
class TestEvalDiffReport:
    """学習前後差分レポートテスト (CQ-0056)"""

    def test_eval_diff_generated(self, tmp_path: Path):
        """eval_diff.json が生成される"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        diff_path = run_dir / "eval" / "eval_diff.json"
        assert diff_path.exists()

    def test_eval_diff_has_keys(self, tmp_path: Path):
        """eval_diff.json に主要指標の差分が含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "eval" / "eval_diff.json") as f:
            diff = json.load(f)

        for key in ["avg_rank", "avg_score", "win_rate", "deal_in_rate"]:
            assert key in diff
            assert "before" in diff[key]
            assert "after" in diff[key]
            assert "delta" in diff[key]

    def test_eval_diff_in_result(self, tmp_path: Path):
        """result dict に eval_diff が含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "eval_diff" in result
        assert "avg_rank" in result["eval_diff"]

    def test_no_eval_diff_without_eval_phase(self, tmp_path: Path):
        """eval フェーズがなければ eval_diff は生成されない"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["selfplay", "learner"]
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "eval_diff" not in result

    def test_eval_before_dir_created(self, tmp_path: Path):
        """eval_before ディレクトリが作成される"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        eval_before_dir = run_dir / "eval_before"
        assert eval_before_dir.exists()
        assert (eval_before_dir / "eval_metrics.json").exists()


@pytest.mark.slow
class TestLearnerNumUpdates:
    """learner 更新回数テスト (CQ-0061)"""

    def test_num_updates_in_train_metrics(self, tmp_path: Path):
        """train_metrics に num_updates が含まれる"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "train_metrics" in result
        assert "num_updates" in result["train_metrics"]
        assert result["train_metrics"]["num_updates"] > 0

    def test_num_updates_in_summary_phase_stats(self, tmp_path: Path):
        """summary.json の phase_stats.learner に num_updates がある"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        learner_stats = summary["phase_stats"]["learner"]
        assert "num_updates" in learner_stats
        assert learner_stats["num_updates"] > 0

    def test_num_updates_matches_expected(self, tmp_path: Path):
        """更新回数が epochs * ceil(samples / batch_size) と一致する"""
        config = _make_minimal_config()
        config.training["epochs"] = 2
        config.training["batch_size"] = 32
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        total_steps = result["train_metrics"]["total_steps"]
        epochs = 2
        batch_size = 32
        import math
        expected = epochs * math.ceil(total_steps / batch_size)
        assert result["train_metrics"]["num_updates"] == expected
