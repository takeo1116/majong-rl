"""実験設定 YAML 読み込みと run ディレクトリ生成"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import shutil
import uuid

import yaml


@dataclass
class ExperimentConfig:
    """実験設定を保持するデータクラス"""
    experiment: dict = field(default_factory=dict)
    feature_encoder: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    reward: dict = field(default_factory=dict)
    selfplay: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    imitation: dict = field(default_factory=dict)
    export: dict = field(default_factory=dict)
    distillation: dict = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: Path) -> ExperimentConfig:
        """YAML ファイルから設定を読み込む"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return ExperimentConfig(
            experiment=data.get("experiment", {}),
            feature_encoder=data.get("feature_encoder", {}),
            model=data.get("model", {}),
            reward=data.get("reward", {}),
            selfplay=data.get("selfplay", {}),
            training=data.get("training", {}),
            evaluation=data.get("evaluation", {}),
            imitation=data.get("imitation", {}),
            export=data.get("export", {}),
            distillation=data.get("distillation", {}),
        )

    def to_yaml(self, path: Path) -> None:
        """設定を YAML ファイルに書き出す"""
        data = {
            "experiment": self.experiment,
            "feature_encoder": self.feature_encoder,
            "model": self.model,
            "reward": self.reward,
            "selfplay": self.selfplay,
            "training": self.training,
            "evaluation": self.evaluation,
            "export": self.export,
        }
        if self.imitation:
            data["imitation"] = self.imitation
        if self.distillation:
            data["distillation"] = self.distillation
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @property
    def is_distillation(self) -> bool:
        """蒸留実験かどうか"""
        return bool(self.distillation.get("enabled", False))


class RunDirectory:
    """run ディレクトリの生成と管理"""

    def __init__(self, base_dir: Path = Path("runs")):
        self._base_dir = Path(base_dir)

    def create(self, config: ExperimentConfig) -> Path:
        """run ディレクトリを生成して config.yaml と notes.md を配置する

        ディレクトリ名: <date>_<name>_<short_id>
        """
        name = config.experiment.get("name", "unnamed")
        short_id = uuid.uuid4().hex[:8]
        date_str = datetime.now().strftime("%Y%m%d")
        dir_name = f"{date_str}_{name}_{short_id}"

        run_dir = self._base_dir / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # サブディレクトリ作成
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "selfplay").mkdir(exist_ok=True)
        (run_dir / "eval").mkdir(exist_ok=True)

        # config.yaml 保存
        config.to_yaml(run_dir / "config.yaml")

        # notes.md 生成
        self._write_notes(run_dir, config)

        return run_dir

    def _write_notes(self, run_dir: Path, config: ExperimentConfig) -> None:
        """notes.md を生成する"""
        lines = [f"# {config.experiment.get('name', 'unnamed')}"]
        lines.append("")

        memo = config.experiment.get("memo", "")
        if memo:
            lines.append(f"## メモ")
            lines.append(memo)
            lines.append("")

        lines.append("## 主要設定")
        lines.append(f"- stage: {config.experiment.get('stage', '?')}")
        lines.append(f"- observation_mode: {config.experiment.get('observation_mode', '?')}")
        lines.append(f"- encoder: {config.feature_encoder.get('name', '?')}")
        lines.append(f"- model: {config.model.get('name', '?')}")
        lines.append(f"- algorithm: {config.training.get('algorithm', '?')}")
        lines.append("")

        with open(run_dir / "notes.md", "w") as f:
            f.write("\n".join(lines))
