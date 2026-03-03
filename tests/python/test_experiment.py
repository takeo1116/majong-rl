"""テスト: experiment.py — 実験設定 YAML と run ディレクトリ"""
import pytest
from pathlib import Path

from mahjong_rl.experiment import ExperimentConfig, RunDirectory


@pytest.fixture
def sample_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment={"name": "test_exp", "stage": 1, "observation_mode": "full", "memo": "テスト用メモ"},
        feature_encoder={"name": "FlatFeatureEncoder", "observation_mode": "full"},
        model={"name": "MLPPolicyValueModel", "hidden_dims": [64], "value_heads": ["round_delta"]},
        reward={"type": "point_delta", "point_delta_scale": 0.0001},
        selfplay={"num_matches": 10, "policy_ratio": 0.5, "baseline_ratio": 0.5},
        training={"algorithm": "ppo", "lr": 0.0003, "batch_size": 64, "epochs": 2},
        evaluation={"num_matches": 10, "seed_start": 0},
        export={},
    )


class TestExperimentConfig:
    """ExperimentConfig テスト"""

    def test_yaml_roundtrip(self, tmp_path: Path, sample_config: ExperimentConfig):
        """YAML 書き出し → 読み込みで内容が一致"""
        yaml_path = tmp_path / "config.yaml"
        sample_config.to_yaml(yaml_path)

        loaded = ExperimentConfig.from_yaml(yaml_path)
        assert loaded.experiment["name"] == "test_exp"
        assert loaded.experiment["stage"] == 1
        assert loaded.model["hidden_dims"] == [64]
        assert loaded.training["algorithm"] == "ppo"

    def test_from_default_yaml(self):
        """configs/default_stage1.yaml を読み込める"""
        yaml_path = Path("configs/default_stage1.yaml")
        if not yaml_path.exists():
            pytest.skip("default_stage1.yaml not found")
        config = ExperimentConfig.from_yaml(yaml_path)
        assert config.experiment["name"] == "stage1_baseline"
        assert config.experiment["stage"] == 1

    def test_missing_sections_default_to_empty(self, tmp_path: Path):
        """YAML に未定義セクションがあっても空 dict になる"""
        yaml_path = tmp_path / "minimal.yaml"
        yaml_path.write_text("experiment:\n  name: minimal\n")

        config = ExperimentConfig.from_yaml(yaml_path)
        assert config.experiment["name"] == "minimal"
        assert config.model == {}
        assert config.training == {}


class TestRunDirectory:
    """RunDirectory テスト"""

    def test_create_run_directory(self, tmp_path: Path, sample_config: ExperimentConfig):
        """run ディレクトリが所定構造で生成される"""
        runner = RunDirectory(base_dir=tmp_path)
        run_dir = runner.create(sample_config)

        assert run_dir.exists()
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "notes.md").exists()
        assert (run_dir / "checkpoints").is_dir()
        assert (run_dir / "selfplay").is_dir()
        assert (run_dir / "eval").is_dir()

    def test_directory_name_format(self, tmp_path: Path, sample_config: ExperimentConfig):
        """ディレクトリ名が <date>_<name>_<id> 形式"""
        runner = RunDirectory(base_dir=tmp_path)
        run_dir = runner.create(sample_config)

        name = run_dir.name
        parts = name.split("_", 2)
        assert len(parts) >= 3
        # 日付部分が 8 桁の数字
        assert len(parts[0]) == 8
        assert parts[0].isdigit()

    def test_notes_contains_memo(self, tmp_path: Path, sample_config: ExperimentConfig):
        """notes.md に memo が転写される"""
        runner = RunDirectory(base_dir=tmp_path)
        run_dir = runner.create(sample_config)

        notes = (run_dir / "notes.md").read_text()
        assert "テスト用メモ" in notes

    def test_notes_contains_settings(self, tmp_path: Path, sample_config: ExperimentConfig):
        """notes.md に主要設定が含まれる"""
        runner = RunDirectory(base_dir=tmp_path)
        run_dir = runner.create(sample_config)

        notes = (run_dir / "notes.md").read_text()
        assert "FlatFeatureEncoder" in notes
        assert "MLPPolicyValueModel" in notes
        assert "ppo" in notes

    def test_config_yaml_readable(self, tmp_path: Path, sample_config: ExperimentConfig):
        """生成された config.yaml を再読み込みできる"""
        runner = RunDirectory(base_dir=tmp_path)
        run_dir = runner.create(sample_config)

        reloaded = ExperimentConfig.from_yaml(run_dir / "config.yaml")
        assert reloaded.experiment["name"] == "test_exp"
        assert reloaded.selfplay["policy_ratio"] == 0.5


class TestDistillationConfig:
    """蒸留設定テスト (CQ-0040)"""

    def test_distillation_config_roundtrip(self, tmp_path: Path):
        """蒸留設定を含む YAML の書き出し → 読み込みで内容一致"""
        config = ExperimentConfig(
            experiment={"name": "distill_test", "stage": 1},
            feature_encoder={"name": "FlatFeatureEncoder"},
            model={"name": "MLPPolicyValueModel"},
            training={"algorithm": "imitation"},
            distillation={
                "enabled": True,
                "teacher": {
                    "observation_mode": "full",
                    "encoder": "FlatFeatureEncoder",
                    "model": "MLPPolicyValueModel",
                    "checkpoint": "teacher_checkpoint.pt",
                },
                "student": {
                    "observation_mode": "partial",
                    "encoder": "FlatFeatureEncoder",
                    "model": "MLPPolicyValueModel",
                },
            },
        )
        yaml_path = tmp_path / "distill_config.yaml"
        config.to_yaml(yaml_path)

        loaded = ExperimentConfig.from_yaml(yaml_path)
        assert loaded.is_distillation is True
        assert loaded.distillation["teacher"]["observation_mode"] == "full"
        assert loaded.distillation["student"]["observation_mode"] == "partial"
        assert loaded.distillation["teacher"]["checkpoint"] == "teacher_checkpoint.pt"

    def test_is_distillation_false_by_default(self, sample_config: ExperimentConfig):
        """通常設定では is_distillation が False"""
        assert sample_config.is_distillation is False

    def test_is_distillation_false_when_not_enabled(self):
        """distillation セクションがあっても enabled=False なら False"""
        config = ExperimentConfig(
            distillation={"enabled": False, "teacher": {}, "student": {}},
        )
        assert config.is_distillation is False

    def test_distillation_omitted_in_yaml_when_empty(self, tmp_path: Path):
        """distillation が空の場合は YAML に出力されない"""
        config = ExperimentConfig(
            experiment={"name": "no_distill"},
        )
        yaml_path = tmp_path / "no_distill.yaml"
        config.to_yaml(yaml_path)

        raw = yaml_path.read_text()
        assert "distillation" not in raw

    def test_teacher_student_distinguishable(self):
        """teacher と student を config 上で区別できる"""
        config = ExperimentConfig(
            distillation={
                "enabled": True,
                "teacher": {
                    "observation_mode": "full",
                    "encoder": "FlatFeatureEncoder",
                    "model": "TeacherModel",
                },
                "student": {
                    "observation_mode": "partial",
                    "encoder": "FlatFeatureEncoder",
                    "model": "StudentModel",
                },
            },
        )
        teacher = config.distillation["teacher"]
        student = config.distillation["student"]
        assert teacher["observation_mode"] != student["observation_mode"]
        assert teacher["model"] != student["model"]
