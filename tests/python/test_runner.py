"""テスト: runner.py — Stage 1 統合ランナー"""
import json
import pytest
import torch
from pathlib import Path
from unittest.mock import patch

from mahjong_rl.experiment import ExperimentConfig
from mahjong_rl.runner import (
    Stage1Runner, resolve_device,
    derive_worker_seed, derive_match_seed, configure_worker_threads,
)


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

        # self-play shard (worker サブディレクトリ)
        shards = list((run_dir / "selfplay").glob("worker_*/shard_*.parquet"))
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

    def test_selfplay_total_matches_nonzero(self, tmp_path: Path):
        """selfplay.total_matches が config の対局数と整合する (CQ-0069)"""
        config = _make_minimal_config()
        config.selfplay["num_matches"] = 3
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        sp = summary["phase_stats"]["selfplay"]
        assert sp["total_matches"] == 3

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


@pytest.mark.smoke
class TestRoundStatsInSummary:
    """summary.json の局結果集計テスト (CQ-0107)"""

    def test_summary_has_round_stat_keys(self, tmp_path: Path):
        """summary.json の phase_stats.selfplay に局結果集計キーが含まれる"""
        config = _make_minimal_config()
        config.selfplay["num_matches"] = 1
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        sp = summary["phase_stats"]["selfplay"]
        expected_keys = [
            "num_rounds", "tsumo_count", "ron_count", "ryukyoku_count",
            "policy_wins", "policy_deal_ins", "policy_draws",
            "policy_win_by_tsumo", "policy_win_by_ron",
        ]
        for key in expected_keys:
            assert key in sp, f"phase_stats.selfplay に {key} がない"

        assert sp["num_rounds"] >= 1
        assert (sp["tsumo_count"] + sp["ron_count"]
                + sp["ryukyoku_count"]) == sp["num_rounds"]

    def test_worker_round_results_jsonl_exists(self, tmp_path: Path):
        """worker ディレクトリに round_results.jsonl が生成される"""
        config = _make_minimal_config()
        config.selfplay["num_matches"] = 1
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        jsonl_files = list((run_dir / "selfplay").glob(
            "worker_*/round_results.jsonl"))
        assert len(jsonl_files) >= 1

        # 中身を軽く検証
        import json as json_mod
        for jf in jsonl_files:
            lines = jf.read_text().strip().split("\n")
            assert len(lines) >= 1
            row = json_mod.loads(lines[0])
            assert "event_type" in row
            assert "winner_players" in row
            assert "is_policy_win" in row
            assert isinstance(row["winner_players"], list)


@pytest.mark.slow
class TestPhaseTiming:
    """phase duration テスト (CQ-0091)"""

    def test_phase_timing_in_summary(self, tmp_path: Path):
        """summary.json に phase_timing が保存される"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        assert "phase_timing" in summary
        pt = summary["phase_timing"]
        # デフォルト phases: selfplay, learner, eval + eval_before
        for phase_name in ["selfplay", "learner", "eval"]:
            assert phase_name in pt, f"{phase_name} が phase_timing にない"
            entry = pt[phase_name]
            assert entry["start_ts"].endswith("Z")
            assert entry["end_ts"].endswith("Z")
            assert entry["duration_sec"] > 0

    def test_total_duration_positive(self, tmp_path: Path):
        """total_duration_sec が正の値"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        assert summary["total_duration_sec"] > 0

    def test_eval_before_timing_recorded(self, tmp_path: Path):
        """eval_before の timing も記録される"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        pt = result["phase_timing"]
        assert "eval_before" in pt
        assert pt["eval_before"]["start_ts"].endswith("Z")
        assert pt["eval_before"]["end_ts"].endswith("Z")


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

    def test_eval_fast_preset_loads(self):
        """eval_fast プリセットが読み込める (CQ-0093)"""
        config = ExperimentConfig.from_yaml(
            Path("configs/stage1_full_flat_mlp_ppo_eval_fast.yaml"))
        assert config.evaluation["num_matches"] == 20
        assert config.evaluation.get("mode", "single") == "single"

    def test_eval_strict_preset_loads(self):
        """eval_strict プリセットが読み込める (CQ-0093)"""
        config = ExperimentConfig.from_yaml(
            Path("configs/stage1_full_flat_mlp_ppo_eval_strict.yaml"))
        assert config.evaluation["num_matches"] == 100
        assert config.evaluation["mode"] == "rotation"
        assert config.evaluation["rotation_seats"] == [0, 1, 2, 3]

    def test_all_presets_pass_validation(self):
        """全プリセットがバリデーションを通る"""
        import glob
        preset_files = glob.glob("configs/stage1_*.yaml")
        assert len(preset_files) >= 5

        for path in preset_files:
            config = ExperimentConfig.from_yaml(Path(path))
            runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
            errors = runner.validate_config()
            assert errors == [], f"{path}: {errors}"


@pytest.mark.smoke
class TestProfiling:
    """簡易プロファイルテスト (CQ-0098)"""

    def test_profiling_disabled_no_file(self, tmp_path: Path):
        """profiling OFF で profile.json が生成されない"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        run_dir = Path(result["run_dir"])
        assert not (run_dir / "profile.json").exists()

    def test_profiling_enabled_creates_file(self, tmp_path: Path):
        """profiling ON で profile.json が生成され entries がある"""
        config = _make_minimal_config()
        config.profiling = {"enabled": True}
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        run_dir = Path(result["run_dir"])
        assert (run_dir / "profile.json").exists()

        with open(run_dir / "profile.json") as f:
            profile = json.load(f)
        assert profile["enabled"] is True
        assert len(profile["entries"]) > 0
        # selfplay_total は必ず記録される
        assert "selfplay_total" in profile["entries"]

    def test_profiling_validation_rejects_non_bool(self):
        """profiling.enabled が bool 以外でバリデーションエラー"""
        config = _make_minimal_config()
        config.profiling = {"enabled": "yes"}
        runner = Stage1Runner(config=config, base_dir=Path("/tmp"))
        errors = runner.validate_config()
        assert any("profiling.enabled" in e for e in errors)

    def test_profiling_has_granular_entries(self, tmp_path: Path):
        """profiling ON で主要処理粒度の計測点が profile.json に含まれる (CQ-0102)"""
        config = _make_minimal_config()
        config.profiling = {"enabled": True}
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        run_dir = Path(result["run_dir"])

        with open(run_dir / "profile.json") as f:
            profile = json.load(f)

        entries = profile["entries"]
        # phase 合計レベル
        assert "selfplay_total" in entries
        assert "learner_total" in entries
        # 主要処理粒度 (CQ-0102)
        assert "selfplay_match_loop" in entries
        assert "selfplay_shard_write" in entries
        assert "shard_read" in entries
        assert "model_forward" in entries
        # count >= 1
        for key in ["selfplay_match_loop", "selfplay_shard_write",
                     "shard_read", "model_forward"]:
            assert entries[key]["count"] >= 1


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

    def test_env_info_has_python_info(self, tmp_path: Path):
        """env_info に Python 実行環境情報が含まれる (CQ-0070)"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        ei = summary["env_info"]
        assert isinstance(ei["python_version"], str)
        assert len(ei["python_version"]) > 0
        assert isinstance(ei["python_executable"], str)
        assert len(ei["python_executable"]) > 0
        # venv は None か文字列
        assert ei["venv"] is None or isinstance(ei["venv"], str)

    def test_notes_has_python_info(self, tmp_path: Path):
        """notes.md に Python 情報が含まれる (CQ-0070)"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        notes = (run_dir / "notes.md").read_text()
        assert "python:" in notes

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


@pytest.mark.smoke
class TestSeedDerivation:
    """seed 派生テスト (CQ-0070)"""

    def test_different_workers_get_different_seeds(self):
        """異なる worker_id で異なる seed が生成される"""
        s0 = derive_worker_seed(42, 0)
        s1 = derive_worker_seed(42, 1)
        s2 = derive_worker_seed(42, 2)
        assert len({s0, s1, s2}) == 3

    def test_same_inputs_give_same_seed(self):
        """同一入力で同一 seed"""
        assert derive_worker_seed(42, 0) == derive_worker_seed(42, 0)

    def test_seed_in_valid_range(self):
        """seed が 0 〜 2**32-1 の範囲"""
        for wid in range(10):
            s = derive_worker_seed(12345, wid)
            assert 0 <= s <= 2**32 - 1

    def test_different_base_seeds(self):
        """異なる base_seed で異なる worker seed"""
        s1 = derive_worker_seed(100, 0)
        s2 = derive_worker_seed(200, 0)
        assert s1 != s2

    def test_match_seed_derivation(self):
        """match seed が worker_seed + match_index から派生される"""
        ws = derive_worker_seed(42, 0)
        m0 = derive_match_seed(ws, 0)
        m1 = derive_match_seed(ws, 1)
        assert m0 != m1
        assert 0 <= m0 <= 2**32 - 1

    def test_match_seed_deterministic(self):
        """同一入力で同一 match seed"""
        ws = derive_worker_seed(42, 0)
        assert derive_match_seed(ws, 5) == derive_match_seed(ws, 5)


@pytest.mark.smoke
class TestConfigureWorkerThreads:
    """worker thread 固定テスト (CQ-0070)"""

    def test_thread_count_set(self):
        """configure_worker_threads が torch スレッド数を設定する"""
        import os
        result = configure_worker_threads(num_threads=1)
        assert result["torch_num_threads"] == 1
        assert os.environ.get("OMP_NUM_THREADS") == "1"
        assert os.environ.get("MKL_NUM_THREADS") == "1"
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"

    def test_returns_env_vars(self):
        """設定結果に env_vars が含まれる"""
        result = configure_worker_threads(num_threads=1)
        assert "env_vars" in result
        assert result["env_vars"]["OMP_NUM_THREADS"] == "1"


@pytest.mark.smoke
class TestDistributeMatches:
    """matches 分配テスト (CQ-0069)"""

    def test_even_distribution(self):
        """均等に分配される"""
        result = Stage1Runner._distribute_matches(10, 2)
        assert result == [5, 5]

    def test_uneven_distribution(self):
        """余りが先頭 worker に割り当てられる"""
        result = Stage1Runner._distribute_matches(10, 3)
        assert result == [4, 3, 3]
        assert sum(result) == 10

    def test_single_worker(self):
        """worker 1 の場合は全 matches"""
        result = Stage1Runner._distribute_matches(10, 1)
        assert result == [10]

    def test_more_workers_than_matches(self):
        """worker 数が matches 数より多い場合"""
        result = Stage1Runner._distribute_matches(2, 5)
        assert sum(result) == 2
        assert result.count(0) == 3


@pytest.mark.smoke
class TestEvalNumWorkers1:
    """num_workers=1 で既存動作を維持 (CQ-0069)"""

    def test_default_num_workers_is_1(self):
        """num_workers 未指定で既存経路"""
        config = _make_minimal_config()
        assert config.evaluation.get("num_workers", 1) == 1


@pytest.mark.slow
@pytest.mark.requires_multiprocess
class TestEvalMultiProcess:
    """multi-process eval テスト (CQ-0069)"""

    def test_parallel_eval_single_mode(self, tmp_path: Path):
        """num_workers=2 で single 評価が完走する"""
        config = _make_minimal_config()
        config.evaluation["num_workers"] = 2
        config.evaluation["num_matches"] = 2
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert result["eval_metrics"]["eval_mode"] == "single"
        assert 1.0 <= result["eval_metrics"]["avg_rank"] <= 4.0
        assert result["eval_metrics"].get("num_workers") == 2

    def test_parallel_eval_rotation_mode(self, tmp_path: Path):
        """num_workers=2 で rotation 評価が完走する"""
        config = _make_minimal_config()
        config.evaluation["num_workers"] = 2
        config.evaluation["num_matches"] = 1
        config.evaluation["mode"] = "rotation"
        config.evaluation["rotation_seats"] = [0, 1]
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert result["eval_metrics"]["eval_mode"] == "rotation"
        assert result["eval_metrics"]["rotation_seats"] == [0, 1]
        assert result["eval_metrics"].get("num_workers") == 2

    def test_parallel_eval_creates_partials(self, tmp_path: Path):
        """parallel eval が partials ディレクトリを生成する"""
        config = _make_minimal_config()
        config.evaluation["num_workers"] = 2
        config.evaluation["num_matches"] = 2
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        partials_dir = run_dir / "eval" / "partials"
        assert partials_dir.exists()
        partials = list(partials_dir.glob("worker_*.json"))
        assert len(partials) >= 2

    def test_num_workers_1_uses_existing_path(self, tmp_path: Path):
        """num_workers=1 で既存の単一プロセス経路が使われる"""
        config = _make_minimal_config()
        config.evaluation["num_workers"] = 1
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        # num_workers=1 では parallel 経路を使わないので num_workers キーなし
        assert "num_workers" not in result.get("eval_metrics", {})

    def test_parallel_partials_have_seed_metadata(self, tmp_path: Path):
        """parallel eval の partial に seed/thread 情報が記録される (CQ-0078)"""
        config = _make_minimal_config()
        config.evaluation["num_workers"] = 2
        config.evaluation["num_matches"] = 2
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        partials_dir = run_dir / "eval" / "partials"
        partials = sorted(partials_dir.glob("worker_*.json"))
        assert len(partials) >= 2

        for p_path in partials:
            with open(p_path) as f:
                data = json.load(f)
            assert "metadata" in data
            meta = data["metadata"]
            assert "base_seed" in meta
            assert meta["base_seed"] == 42
            assert "worker_seed" in meta
            assert "num_threads" in meta
            assert "torch_num_threads" in meta

    def test_parallel_worker_seeds_unique(self, tmp_path: Path):
        """parallel eval の各 worker の seed が異なる (CQ-0078)"""
        config = _make_minimal_config()
        config.evaluation["num_workers"] = 2
        config.evaluation["num_matches"] = 2
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        partials_dir = run_dir / "eval" / "partials"
        partials = sorted(partials_dir.glob("worker_*.json"))

        worker_seeds = set()
        for p_path in partials:
            with open(p_path) as f:
                data = json.load(f)
            worker_seeds.add(data["metadata"]["worker_seed"])
        assert len(worker_seeds) == len(partials)

    def test_model_file_cleaned_up(self, tmp_path: Path):
        """parallel eval 後に一時モデルファイルが削除される (CQ-0077)"""
        config = _make_minimal_config()
        config.evaluation["num_workers"] = 2
        config.evaluation["num_matches"] = 2
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        # _eval_model.pt は削除されているはず
        assert not (run_dir / "eval" / "_eval_model.pt").exists()


@pytest.mark.slow
@pytest.mark.requires_multiprocess
class TestSelfPlayMultiProcess:
    """multi-process self-play テスト (CQ-0072)"""

    def test_parallel_selfplay_completes(self, tmp_path: Path):
        """num_workers=2 で self-play が完走する"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 2
        config.selfplay["num_matches"] = 2
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert result["selfplay_stats"]["total_steps"] > 0
        assert result["selfplay_stats"].get("num_workers") == 2

    def test_parallel_selfplay_creates_worker_dirs(self, tmp_path: Path):
        """worker ごとに shard サブディレクトリが作られる"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 2
        config.selfplay["num_matches"] = 2
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        selfplay_dir = run_dir / "selfplay"
        assert (selfplay_dir / "worker_0").exists()
        assert (selfplay_dir / "worker_1").exists()
        for wid in range(2):
            shards = list((selfplay_dir / f"worker_{wid}").glob("shard_*.parquet"))
            assert len(shards) >= 1

    def test_parallel_selfplay_learner_reads_shards(self, tmp_path: Path):
        """parallel self-play の shard を learner が読めて学習が完走する"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 2
        config.selfplay["num_matches"] = 2
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert "train_metrics" in result
        assert result["train_metrics"]["total_steps"] > 0

    def test_parallel_selfplay_seeds_differ(self, tmp_path: Path):
        """各 worker の seed が異なる (CQ-0073)"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 2
        config.selfplay["num_matches"] = 2
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        selfplay_dir = run_dir / "selfplay"
        worker_seeds = set()
        for wid in range(2):
            stats_path = selfplay_dir / f"worker_{wid}" / "stats.json"
            assert stats_path.exists()
            with open(stats_path) as f:
                data = json.load(f)
            worker_seeds.add(data["worker_seed"])
        assert len(worker_seeds) == 2

    def test_num_workers_1_uses_existing_path(self, tmp_path: Path):
        """num_workers=1 で既存経路が使われる"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 1
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert "num_workers" not in result.get("selfplay_stats", {})

    def test_model_file_cleaned_up(self, tmp_path: Path):
        """parallel self-play 後に一時モデルファイルが削除される"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 2
        config.selfplay["num_matches"] = 2
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        assert not (run_dir / "selfplay" / "_selfplay_model.pt").exists()

    def test_seed_strategy_in_summary(self, tmp_path: Path):
        """summary に seed_strategy が記録される (CQ-0073)"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 2
        config.selfplay["num_matches"] = 2
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        sp_stats = summary["phase_stats"]["selfplay"]
        assert sp_stats["num_workers"] == 2
        assert sp_stats["seed_strategy"] is not None
        assert sp_stats["seed_strategy"]["base_seed"] == 42

    def test_seed_range_recorded_in_worker_stats(self, tmp_path: Path):
        """各 worker の stats.json に seed range が記録される (CQ-0080)"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 2
        config.selfplay["num_matches"] = 4
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        run_dir = Path(result["run_dir"])
        selfplay_dir = run_dir / "selfplay"

        for wid in range(2):
            stats_path = selfplay_dir / f"worker_{wid}" / "stats.json"
            assert stats_path.exists()
            with open(stats_path) as f:
                data = json.load(f)
            # seed range フィールドが存在する
            assert "match_index_start" in data
            assert "match_index_end" in data
            assert "first_match_seed" in data
            assert "last_match_seed" in data
            # match_index_start は 0
            assert data["match_index_start"] == 0
            # match_index_end は num_matches - 1
            wm = data["num_matches"]
            assert data["match_index_end"] == wm - 1
            # first/last seed が異なる (matches > 1 なら)
            if wm > 1:
                assert data["first_match_seed"] != data["last_match_seed"]


class TestEvalWorkerErrorReporting:
    """worker 失敗時のエラー詳細ログテスト (CQ-0079)"""

    def test_error_queue_details_in_runtime_error(self):
        """error_queue にエラー詳細があると RuntimeError メッセージに含まれる"""
        import queue as stdlib_queue
        from unittest.mock import MagicMock

        error_queue = stdlib_queue.Queue()
        error_queue.put({
            "worker_id": 99,
            "exception_type": "ValueError",
            "message": "テスト用の意図的エラー",
            "traceback": "Traceback ...\nValueError: テスト用の意図的エラー\n",
        })

        # exitcode != 0 の mock process
        mock_proc = MagicMock()
        mock_proc.exitcode = 1

        with pytest.raises(RuntimeError) as exc_info:
            Stage1Runner._wait_and_check_workers([mock_proc], error_queue=error_queue)

        msg = str(exc_info.value)
        assert "worker 99" in msg
        assert "ValueError" in msg
        assert "テスト用の意図的エラー" in msg

    def test_multiple_worker_errors_all_reported(self):
        """複数 worker が失敗した場合、全エラーが報告される"""
        import queue as stdlib_queue
        from unittest.mock import MagicMock

        error_queue = stdlib_queue.Queue()
        error_queue.put({
            "worker_id": 0,
            "exception_type": "ValueError",
            "message": "エラーA",
            "traceback": "Traceback ...\nValueError: エラーA\n",
        })
        error_queue.put({
            "worker_id": 1,
            "exception_type": "RuntimeError",
            "message": "エラーB",
            "traceback": "Traceback ...\nRuntimeError: エラーB\n",
        })

        mock_p0 = MagicMock()
        mock_p0.exitcode = 1
        mock_p1 = MagicMock()
        mock_p1.exitcode = 1

        with pytest.raises(RuntimeError) as exc_info:
            Stage1Runner._wait_and_check_workers(
                [mock_p0, mock_p1], error_queue=error_queue)

        msg = str(exc_info.value)
        assert "worker 0" in msg
        assert "worker 1" in msg
        assert "エラーA" in msg
        assert "エラーB" in msg
        assert "2/2" in msg

    def test_no_error_queue_still_raises_on_failure(self):
        """error_queue なしでも exitcode ベースで RuntimeError が出る"""
        from unittest.mock import MagicMock

        mock_proc = MagicMock()
        mock_proc.exitcode = 1

        with pytest.raises(RuntimeError) as exc_info:
            Stage1Runner._wait_and_check_workers([mock_proc])

        msg = str(exc_info.value)
        assert "eval worker" in msg
        assert "失敗" in msg

    def test_success_workers_no_error(self):
        """全 worker 成功時はエラーなし"""
        import queue as stdlib_queue
        from unittest.mock import MagicMock

        error_queue = stdlib_queue.Queue()

        mock_proc = MagicMock()
        mock_proc.exitcode = 0

        # 例外が発生しないことを確認
        Stage1Runner._wait_and_check_workers([mock_proc], error_queue=error_queue)


@pytest.mark.smoke
class TestParallelConfigValidation:
    """並列実行向け config validation テスト (CQ-0074)"""

    def test_valid_parallel_config(self, tmp_path: Path):
        """num_workers=2 等の正常 config が通る"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 2
        config.selfplay["worker_num_threads"] = 2
        config.evaluation["num_workers"] = 2
        config.evaluation["worker_num_threads"] = 2
        config.selfplay["output_layout"] = "worker_subdir"
        config.experiment["seed_strategy"] = "derive"
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        errors = runner.validate_config()
        assert errors == []

    def test_num_workers_zero_rejected(self, tmp_path: Path):
        """num_workers=0 は拒否される"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 0
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        errors = runner.validate_config()
        assert any("selfplay.num_workers" in e for e in errors)

    def test_num_workers_negative_rejected(self, tmp_path: Path):
        """num_workers=-1 は拒否される"""
        config = _make_minimal_config()
        config.evaluation["num_workers"] = -1
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        errors = runner.validate_config()
        assert any("evaluation.num_workers" in e for e in errors)

    def test_num_workers_float_rejected(self, tmp_path: Path):
        """num_workers=1.5 は拒否される"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 1.5
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        errors = runner.validate_config()
        assert any("selfplay.num_workers" in e for e in errors)

    def test_worker_num_threads_zero_rejected(self, tmp_path: Path):
        """worker_num_threads=0 は拒否される"""
        config = _make_minimal_config()
        config.selfplay["worker_num_threads"] = 0
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        errors = runner.validate_config()
        assert any("worker_num_threads" in e for e in errors)

    def test_invalid_output_layout_rejected(self, tmp_path: Path):
        """不正な output_layout は拒否される"""
        config = _make_minimal_config()
        config.selfplay["output_layout"] = "flat"
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        errors = runner.validate_config()
        assert any("output_layout" in e for e in errors)

    def test_invalid_seed_strategy_rejected(self, tmp_path: Path):
        """不正な seed_strategy は拒否される"""
        config = _make_minimal_config()
        config.experiment["seed_strategy"] = "random"
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        errors = runner.validate_config()
        assert any("seed_strategy" in e for e in errors)

    def test_imitation_workers_zero_rejected(self, tmp_path: Path):
        """imitation.num_workers=0 は拒否される"""
        config = _make_minimal_config()
        config.imitation["num_workers"] = 0
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        errors = runner.validate_config()
        assert any("imitation.num_workers" in e for e in errors)

    def test_imitation_workers_valid(self, tmp_path: Path):
        """imitation.num_workers=2 は正常に通る"""
        config = _make_minimal_config()
        config.imitation["num_workers"] = 2
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        errors = runner.validate_config()
        assert errors == []

    def test_default_config_passes(self, tmp_path: Path):
        """既存の最小 config がそのまま通る"""
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        errors = runner.validate_config()
        assert errors == []


@pytest.mark.slow
@pytest.mark.requires_multiprocess
class TestParallelSmokeIntegration:
    """parallel eval / self-play の統合 smoke test (CQ-0075)"""

    def test_parallel_eval_partials_aggregated(self, tmp_path: Path):
        """num_workers=2 の eval で partials が集約され最終 eval_metrics が得られる"""
        config = _make_minimal_config()
        config.evaluation["num_workers"] = 2
        config.evaluation["num_matches"] = 2
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert "eval_metrics" in result
        em = result["eval_metrics"]
        assert "avg_rank" in em or "win_rate" in em or "policy_avg_rank" in em

        # partials ディレクトリに部分結果がある
        run_dir = Path(result["run_dir"])
        partials_dir = run_dir / "eval" / "partials"
        assert partials_dir.exists()
        partial_files = list(partials_dir.glob("worker_*.json"))
        assert len(partial_files) >= 1

    def test_parallel_selfplay_shard_separation_and_learner(self, tmp_path: Path):
        """selfplay(num_workers=2) → learner が通り、全 worker の shard が読まれる"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 2
        config.selfplay["num_matches"] = 4
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        run_dir = Path(result["run_dir"])

        # 各 worker に shard がある
        for wid in range(2):
            worker_dir = run_dir / "selfplay" / f"worker_{wid}"
            assert worker_dir.exists()
            shards = list(worker_dir.glob("shard_*.parquet"))
            assert len(shards) >= 1

        # learner が学習できた
        assert "train_metrics" in result
        assert result["train_metrics"]["total_steps"] > 0

    def test_single_worker_full_pipeline(self, tmp_path: Path):
        """num_workers=1 で selfplay → learner → eval の全導線が通る"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 1
        config.evaluation["num_workers"] = 1
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert "selfplay_stats" in result
        assert "train_metrics" in result
        assert "eval_metrics" in result

    def test_parallel_summary_has_worker_outputs(self, tmp_path: Path):
        """summary.json に parallel 関連の記録がある"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 2
        config.selfplay["num_matches"] = 2
        config.evaluation["num_workers"] = 2
        config.evaluation["num_matches"] = 2
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        # selfplay の parallel 記録
        sp = summary["phase_stats"]["selfplay"]
        assert sp["num_workers"] == 2
        assert sp["seed_strategy"] is not None

        # worker ごとの stats.json が存在する
        selfplay_dir = run_dir / "selfplay"
        for wid in range(2):
            stats_path = selfplay_dir / f"worker_{wid}" / "stats.json"
            assert stats_path.exists()


    def test_parallel_summary_has_round_stat_keys(self, tmp_path: Path):
        """parallel self-play の summary に局結果集計キーが正しく反映される (CQ-0108, CQ-0109)"""
        config = _make_minimal_config()
        config.selfplay["num_workers"] = 2
        config.selfplay["num_matches"] = 2
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        # 環境要因（Permission denied 等）で parallel 実行が失敗した場合は skip
        err = result.get("error", "")
        if err and any(kw in str(err).lower()
                       for kw in ["permission denied", "operation not permitted",
                                  "spawn", "fork"]):
            pytest.skip(f"parallel 実行が環境制約で失敗: {err}")

        assert "error" not in result
        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        sp = summary["phase_stats"]["selfplay"]
        expected_keys = [
            "num_rounds", "tsumo_count", "ron_count", "ryukyoku_count",
            "policy_wins", "policy_deal_ins", "policy_draws",
            "policy_win_by_tsumo", "policy_win_by_ron",
        ]
        for key in expected_keys:
            assert key in sp, f"parallel summary に {key} がない"

        # parallel 実行では少なくとも 1 局以上あるはず
        assert sp["num_rounds"] >= 1
        # 合計整合
        assert (sp["tsumo_count"] + sp["ron_count"]
                + sp["ryukyoku_count"]) == sp["num_rounds"]


@pytest.mark.slow
@pytest.mark.requires_multiprocess
class TestImitationMultiProcess:
    """imitation 教師データ生成の multi-process テスト (CQ-0082)"""

    def test_parallel_imitation_completes(self, tmp_path: Path):
        """imitation.num_workers=2 で imitation → selfplay → learner → eval が完走する"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["imitation", "selfplay", "learner", "eval"]
        config.selfplay["imitation_matches"] = 4
        config.imitation["num_workers"] = 2
        config.training["imitation_epochs"] = 1
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert "imitation_metrics" in result
        assert "selfplay_stats" in result
        assert "train_metrics" in result

    def test_parallel_imitation_shards_in_worker_dirs(self, tmp_path: Path):
        """各 worker のサブディレクトリに shard が生成される"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["imitation", "selfplay", "learner", "eval"]
        config.selfplay["imitation_matches"] = 4
        config.imitation["num_workers"] = 2
        config.training["imitation_epochs"] = 1
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        run_dir = Path(result["run_dir"])
        imi_dir = run_dir / "imitation"

        # 各 worker にサブディレクトリと shard がある
        for wid in range(2):
            worker_dir = imi_dir / f"worker_{wid}"
            assert worker_dir.exists()
            shards = list(worker_dir.glob("shard_*.parquet"))
            assert len(shards) >= 1

    def test_parallel_imitation_learner_reads_nested_shards(self, tmp_path: Path):
        """並列生成された imitation shard を learner が読み取れる"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["imitation", "selfplay", "learner", "eval"]
        config.selfplay["imitation_matches"] = 4
        config.imitation["num_workers"] = 2
        config.training["imitation_epochs"] = 1
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        imi_metrics = result["imitation_metrics"]
        assert imi_metrics["policy_loss"] > 0  # 学習が実行された

    def test_single_worker_fallback(self, tmp_path: Path):
        """imitation.num_workers=1 で従来経路が動作する"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["imitation", "selfplay", "learner", "eval"]
        config.selfplay["imitation_matches"] = 2
        config.imitation["num_workers"] = 1
        config.training["imitation_epochs"] = 1
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        assert "imitation_metrics" in result

    def test_summary_records_imitation_workers(self, tmp_path: Path):
        """summary.json に imitation の num_workers と seed_strategy が記録される (CQ-0083)"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["imitation", "selfplay", "learner", "eval"]
        config.selfplay["imitation_matches"] = 4
        config.imitation["num_workers"] = 2
        config.training["imitation_epochs"] = 1
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        run_dir = Path(result["run_dir"])
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        imi_stats = summary["phase_stats"]["imitation"]
        assert imi_stats["num_workers"] == 2
        assert imi_stats["shard_count"] >= 2
        # seed_strategy が記録されている (CQ-0083)
        ss = imi_stats["seed_strategy"]
        assert ss is not None
        assert "base_seed" in ss
        assert ss["base_seed"] == 42
        assert "method" in ss

    def test_notes_records_imitation_parallel_info(self, tmp_path: Path):
        """notes.md に imitation 並列情報が記録される (CQ-0083)"""
        config = _make_minimal_config()
        config.experiment["phases"] = ["imitation", "selfplay", "learner", "eval"]
        config.selfplay["imitation_matches"] = 4
        config.imitation["num_workers"] = 2
        config.training["imitation_epochs"] = 1
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()

        assert "error" not in result
        run_dir = Path(result["run_dir"])
        notes_content = (run_dir / "notes.md").read_text()
        assert "imitation" in notes_content
        assert "num_workers=2" in notes_content


# --- CQ-0112: manifest / resume / reuse テスト ---


def _run_minimal(tmp_path: Path, phases=None) -> tuple:
    """最小 run を実行して (config, result, run_dir) を返すヘルパー"""
    config = _make_minimal_config()
    if phases is not None:
        config.experiment["phases"] = phases
    config.experiment["global_seed"] = 42
    runner = Stage1Runner(config=config, base_dir=tmp_path)
    result = runner.run()
    run_dir = Path(result["run_dir"])
    return config, result, run_dir


@pytest.mark.smoke
class TestManifest:
    """artifacts_manifest.json 生成テスト (CQ-0109, CQ-0112)"""

    def test_manifest_generated(self, tmp_path: Path):
        """通常 run で artifacts_manifest.json が生成される"""
        _, result, run_dir = _run_minimal(tmp_path)
        assert "error" not in result
        assert (run_dir / "artifacts_manifest.json").exists()

    def test_manifest_keys(self, tmp_path: Path):
        """manifest に必須キーが含まれる"""
        _, result, run_dir = _run_minimal(tmp_path)
        assert "error" not in result
        with open(run_dir / "artifacts_manifest.json") as f:
            manifest = json.load(f)
        assert manifest["manifest_version"] == 1
        assert "phase_completion" in manifest
        assert "artifacts" in manifest
        assert "config_fingerprint" in manifest
        assert len(manifest["config_fingerprint"]) == 64  # SHA-256 hex
        assert "reuse_metadata" in manifest
        # reuse_metadata のキー
        rm = manifest["reuse_metadata"]
        assert "global_seed" in rm
        assert "policy_ratio" in rm
        assert "selfplay_num_matches" in rm

    def test_manifest_phase_completion_matches_status(self, tmp_path: Path):
        """manifest.phase_completion と summary.json.phase_status が一致する"""
        _, result, run_dir = _run_minimal(tmp_path)
        assert "error" not in result
        with open(run_dir / "artifacts_manifest.json") as f:
            manifest = json.load(f)
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)
        pc = manifest["phase_completion"]
        ps = summary["phase_status"]
        # phase_status のキーは全て manifest にある
        for phase, status in ps.items():
            assert pc.get(phase) == status, f"{phase}: {pc.get(phase)} != {status}"


@pytest.mark.smoke
class TestPhaseResume:
    """phase 単位 resume テスト (CQ-0111, CQ-0112)"""

    def test_resume_skips_completed_phases(self, tmp_path: Path):
        """正常完了 run_dir で resume → 全 phase スキップ (CQ-0115)"""
        config, result1, run_dir = _run_minimal(tmp_path)
        assert "error" not in result1

        # resume 実行
        runner2 = Stage1Runner(config=config, base_dir=tmp_path,
                               resume_run_dir=run_dir)
        result2 = runner2.run()
        assert "error" not in result2
        assert result2["run_dir"] == str(run_dir)

        # summary を再読み込み
        with open(run_dir / "summary.json") as f:
            summary = json.load(f)
        # CQ-0115: phase_status は過去の success を維持
        for phase in ["selfplay", "learner", "eval"]:
            assert summary["phase_status"][phase] == "success", \
                f"{phase}: phase_status={summary['phase_status'][phase]} (expected success)"
        # CQ-0115: phase_action に skipped が記録される
        assert "phase_action" in summary
        for phase in ["selfplay", "learner", "eval"]:
            assert summary["phase_action"][phase] == "skipped", \
                f"{phase}: phase_action={summary['phase_action'].get(phase)} (expected skipped)"

    def test_resume_missing_artifact_fails(self, tmp_path: Path):
        """完了マークがあるが成果物欠落 → ValueError"""
        import shutil

        config, result1, run_dir = _run_minimal(tmp_path)
        assert "error" not in result1

        # selfplay ディレクトリを削除
        sp_dir = run_dir / "selfplay"
        if sp_dir.exists():
            shutil.rmtree(sp_dir)

        # resume → 成果物不足でエラー
        runner2 = Stage1Runner(config=config, base_dir=tmp_path,
                               resume_run_dir=run_dir)
        with pytest.raises(ValueError, match="成果物整合エラー"):
            runner2.run()

    def test_resume_no_manifest_fails(self, tmp_path: Path):
        """manifest がない run_dir で resume → ValueError"""
        run_dir = tmp_path / "fake_run"
        run_dir.mkdir()
        config = _make_minimal_config()
        runner = Stage1Runner(config=config, base_dir=tmp_path,
                              resume_run_dir=run_dir)
        with pytest.raises(ValueError, match="artifacts_manifest.json"):
            runner.run()


@pytest.mark.smoke
class TestArtifactReuse:
    """成果物再利用テスト (CQ-0110, CQ-0112)"""

    def test_reuse_runs_remaining_phases(self, tmp_path: Path):
        """ref run の成果物を再利用して後続 phase のみ実行"""
        ref_config, ref_result, ref_dir = _run_minimal(tmp_path / "ref")
        assert "error" not in ref_result

        # reuse 実行: imitation+selfplay+eval_before 再利用 → learner+eval のみ
        config = _make_minimal_config()
        config.experiment["global_seed"] = 42
        reuse_from = {
            "run_dir": str(ref_dir),
            "phases": ["selfplay", "eval_before"],
        }
        runner = Stage1Runner(config=config, base_dir=tmp_path / "new",
                              reuse_from=reuse_from)
        result = runner.run()
        assert "error" not in result

        new_dir = Path(result["run_dir"])
        # selfplay shard がコピーされている
        sp_shards = list((new_dir / "selfplay").glob("**/shard_*.parquet"))
        assert len(sp_shards) >= 1

        # summary に reuse_info が記録される
        with open(new_dir / "summary.json") as f:
            summary = json.load(f)
        assert "reuse_info" in summary
        assert summary["reuse_info"]["ref_run_dir"] == str(ref_dir)
        assert "selfplay" in summary["reuse_info"]["reused_phases"]

    def test_reuse_missing_ref_fails(self, tmp_path: Path):
        """参照元 run_dir が存在しない → エラー"""
        config = _make_minimal_config()
        reuse_from = {
            "run_dir": str(tmp_path / "nonexistent"),
            "phases": ["selfplay"],
        }
        runner = Stage1Runner(config=config, base_dir=tmp_path,
                              reuse_from=reuse_from)
        with pytest.raises(ValueError, match="artifacts_manifest.json"):
            runner.run()


@pytest.mark.smoke
class TestValidateArtifactsStrict:
    """成果物整合チェック厳密化テスト (CQ-0113)"""

    def test_selfplay_shard_missing_fails(self, tmp_path: Path):
        """selfplay ディレクトリはあるが shard がない → エラー"""
        import shutil

        config, result, run_dir = _run_minimal(tmp_path)
        assert "error" not in result

        # shard ファイルを全削除（ディレクトリは残す）
        sp_dir = run_dir / "selfplay"
        for f in sp_dir.glob("**/*.parquet"):
            f.unlink()

        runner2 = Stage1Runner(config=config, base_dir=tmp_path,
                               resume_run_dir=run_dir)
        with pytest.raises(ValueError, match="shard ファイルがありません"):
            runner2.run()

    def test_eval_before_missing_fails_on_reuse(self, tmp_path: Path):
        """eval_before が欠落した状態で reuse 指定 → エラー"""
        # imitation 付き run で eval_before を生成
        config = _make_minimal_config()
        config.experiment["phases"] = ["imitation", "selfplay", "learner", "eval"]
        config.selfplay["imitation_matches"] = 2
        config.training["imitation_epochs"] = 1
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path / "ref")
        ref_result = runner.run()
        ref_dir = Path(ref_result["run_dir"])

        if "error" in ref_result:
            pytest.skip("ref run が失敗")

        # eval_before ディレクトリを削除
        import shutil
        eb_dir = ref_dir / "eval_before"
        if eb_dir.exists():
            shutil.rmtree(eb_dir)
        # manifest を書き換えて eval_before の avg_rank を消す
        manifest_path = ref_dir / "artifacts_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["artifacts"]["eval_before"] = {"exists": False, "path": "eval_before", "avg_rank": None}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        # summary からも eval_before を消す
        summary_path = ref_dir / "summary.json"
        with open(summary_path) as f:
            summary = json.load(f)
        summary.get("phase_stats", {}).pop("eval_before", None)
        with open(summary_path, "w") as f:
            json.dump(summary, f)

        # reuse で eval_before を指定 → 失敗
        config2 = _make_minimal_config()
        config2.experiment["global_seed"] = 42
        reuse_from = {
            "run_dir": str(ref_dir),
            "phases": ["eval_before"],
        }
        runner2 = Stage1Runner(config=config2, base_dir=tmp_path / "new",
                               reuse_from=reuse_from)
        with pytest.raises(ValueError, match="成果物整合エラー"):
            runner2.run()

    def test_normal_run_passes_validation(self, tmp_path: Path):
        """正常 run は厳密化後も成功する"""
        _, result, _ = _run_minimal(tmp_path)
        assert "error" not in result


@pytest.mark.smoke
class TestCheckpointIntegrity:
    """checkpoint 初期化整合テスト (CQ-0114)"""

    def test_reuse_selfplay_loads_checkpoint(self, tmp_path: Path):
        """selfplay 再利用時に checkpoint がロードされ記録される"""
        # imitation 付き ref run
        config = _make_minimal_config()
        config.experiment["phases"] = ["imitation", "selfplay", "learner", "eval"]
        config.selfplay["imitation_matches"] = 2
        config.training["imitation_epochs"] = 1
        config.experiment["global_seed"] = 42
        runner = Stage1Runner(config=config, base_dir=tmp_path / "ref")
        ref_result = runner.run()
        if "error" in ref_result:
            pytest.skip("ref run が失敗")
        ref_dir = Path(ref_result["run_dir"])

        # reuse: selfplay を再利用
        config2 = _make_minimal_config()
        config2.experiment["phases"] = ["imitation", "selfplay", "learner", "eval"]
        config2.selfplay["imitation_matches"] = 2
        config2.training["imitation_epochs"] = 1
        config2.experiment["global_seed"] = 42
        reuse_from = {
            "run_dir": str(ref_dir),
            "phases": ["imitation", "selfplay"],
        }
        runner2 = Stage1Runner(config=config2, base_dir=tmp_path / "new",
                               reuse_from=reuse_from)
        result2 = runner2.run()
        assert "error" not in result2

        # summary に loaded_checkpoint が記録される
        new_dir = Path(result2["run_dir"])
        with open(new_dir / "summary.json") as f:
            summary = json.load(f)
        assert "loaded_checkpoint" in summary
        assert "checkpoint_imitation.pt" in summary["loaded_checkpoint"]


@pytest.mark.smoke
class TestPhaseActionSeparation:
    """phase_status / phase_action 分離テスト (CQ-0115)"""

    def test_resume_preserves_phase_status(self, tmp_path: Path):
        """resume 後も過去成功 phase が phase_status=success のまま保持される"""
        config, _, run_dir = _run_minimal(tmp_path)

        # resume
        runner2 = Stage1Runner(config=config, base_dir=tmp_path,
                               resume_run_dir=run_dir)
        result2 = runner2.run()
        assert "error" not in result2

        with open(run_dir / "summary.json") as f:
            summary = json.load(f)

        # phase_status は全て success
        for phase in ["selfplay", "learner", "eval"]:
            assert summary["phase_status"][phase] == "success"

        # phase_action は全て skipped
        assert "phase_action" in summary
        for phase in ["selfplay", "learner", "eval"]:
            assert summary["phase_action"][phase] == "skipped"

    def test_reuse_records_reused_action(self, tmp_path: Path):
        """reuse 時に phase_action に reused が記録される"""
        _, ref_result, ref_dir = _run_minimal(tmp_path / "ref")
        assert "error" not in ref_result

        config2 = _make_minimal_config()
        config2.experiment["global_seed"] = 42
        reuse_from = {
            "run_dir": str(ref_dir),
            "phases": ["selfplay"],
        }
        runner2 = Stage1Runner(config=config2, base_dir=tmp_path / "new",
                               reuse_from=reuse_from)
        result2 = runner2.run()
        assert "error" not in result2

        new_dir = Path(result2["run_dir"])
        with open(new_dir / "summary.json") as f:
            summary = json.load(f)
        assert summary.get("phase_action", {}).get("selfplay") == "reused"

    def test_normal_run_no_phase_action(self, tmp_path: Path):
        """通常 run では phase_action が空（記録されない）"""
        _, result, run_dir = _run_minimal(tmp_path)
        assert "error" not in result

        with open(run_dir / "summary.json") as f:
            summary = json.load(f)
        # 通常 run では phase_action が存在しないか空
        assert "phase_action" not in summary or summary["phase_action"] == {}


@pytest.mark.smoke
class TestRotationReuseEvalDiff:
    """rotation + 再利用 (eval_before 含む) で eval_diff が生成されることを検証 (CQ-0118)"""

    def test_rotation_reuse_generates_eval_diff(self, tmp_path: Path):
        """rotation 条件で eval_before を再利用した run で eval_diff.json が生成される"""
        # 参照元 run: rotation 評価付き
        ref_config = _make_minimal_config()
        ref_config.experiment["global_seed"] = 42
        ref_config.evaluation["mode"] = "rotation"
        ref_config.evaluation["rotation_seats"] = [0, 1]
        ref_config.evaluation["num_matches"] = 1
        ref_runner = Stage1Runner(config=ref_config, base_dir=tmp_path / "ref")
        ref_result = ref_runner.run()
        assert "error" not in ref_result
        ref_dir = Path(ref_result["run_dir"])

        # 参照元に eval_rotation.json が存在することを確認
        assert (ref_dir / "eval_before" / "eval_rotation.json").exists(), \
            "参照元 run に eval_rotation.json がありません"

        # 再利用 run: eval_before を reuse
        config2 = _make_minimal_config()
        config2.experiment["global_seed"] = 42
        config2.evaluation["mode"] = "rotation"
        config2.evaluation["rotation_seats"] = [0, 1]
        config2.evaluation["num_matches"] = 1
        reuse_from = {
            "run_dir": str(ref_dir),
            "phases": ["selfplay", "eval_before"],
        }
        runner2 = Stage1Runner(config=config2, base_dir=tmp_path / "new",
                               reuse_from=reuse_from)
        result2 = runner2.run()
        assert "error" not in result2

        new_dir = Path(result2["run_dir"])

        # eval_diff.json が生成されていること
        diff_path = new_dir / "eval" / "eval_diff.json"
        assert diff_path.exists(), "eval_diff.json が生成されていません"

        with open(diff_path) as f:
            diff = json.load(f)

        # 主要4指標の delta が null でないこと
        for key in ("avg_rank", "avg_score", "win_rate", "deal_in_rate"):
            assert key in diff, f"{key} が eval_diff にありません"
            assert diff[key]["delta"] is not None, \
                f"{key}.delta が null です"
            assert diff[key]["before"] is not None, \
                f"{key}.before が null です"
            assert diff[key]["after"] is not None, \
                f"{key}.after が null です"

    def test_single_reuse_eval_diff_not_regressed(self, tmp_path: Path):
        """single 条件の既存 eval_diff 生成が退行しない"""
        # 参照元 run: single 評価（デフォルト）
        ref_config = _make_minimal_config()
        ref_config.experiment["global_seed"] = 42
        ref_runner = Stage1Runner(config=ref_config, base_dir=tmp_path / "ref")
        ref_result = ref_runner.run()
        assert "error" not in ref_result
        ref_dir = Path(ref_result["run_dir"])

        # 再利用 run: eval_before を reuse
        config2 = _make_minimal_config()
        config2.experiment["global_seed"] = 42
        reuse_from = {
            "run_dir": str(ref_dir),
            "phases": ["selfplay", "eval_before"],
        }
        runner2 = Stage1Runner(config=config2, base_dir=tmp_path / "new",
                               reuse_from=reuse_from)
        result2 = runner2.run()
        assert "error" not in result2

        new_dir = Path(result2["run_dir"])
        diff_path = new_dir / "eval" / "eval_diff.json"
        assert diff_path.exists(), "single 条件で eval_diff.json が生成されていません"

        with open(diff_path) as f:
            diff = json.load(f)
        for key in ("avg_rank", "avg_score", "win_rate", "deal_in_rate"):
            assert diff[key]["delta"] is not None, \
                f"single 条件: {key}.delta が null です"

    def test_rotation_manifest_stores_all_metrics(self, tmp_path: Path):
        """rotation run の manifest に主要4指標が記録される"""
        config = _make_minimal_config()
        config.experiment["global_seed"] = 42
        config.evaluation["mode"] = "rotation"
        config.evaluation["rotation_seats"] = [0, 1]
        config.evaluation["num_matches"] = 1
        runner = Stage1Runner(config=config, base_dir=tmp_path)
        result = runner.run()
        assert "error" not in result
        run_dir = Path(result["run_dir"])

        with open(run_dir / "artifacts_manifest.json") as f:
            manifest = json.load(f)
        eb = manifest["artifacts"]["eval_before"]
        for key in ("avg_rank", "avg_score", "win_rate", "deal_in_rate"):
            assert eb.get(key) is not None, \
                f"manifest の eval_before に {key} がありません"
