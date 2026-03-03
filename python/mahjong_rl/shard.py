"""学習サンプル構造と shard file 入出力（形式抽象化対応）"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# --- 必須メタデータ定義 ---

# 空文字禁止フィールド（shard_id 含む — ShardWriter が add 時に自動付与してから検証）
_REQUIRED_STR_FIELDS = ("experiment_id", "run_id", "worker_id", "shard_id", "episode_id")
# 非負制約フィールド
_REQUIRED_NONNEG_FIELDS = ("model_version", "generation", "round_id", "step_id")


@dataclass
class LearningSample:
    """1ステップの学習サンプル"""
    observation: np.ndarray       # エンコード済み特徴量 (flat float32)
    legal_mask: np.ndarray        # (34,) float32
    action: int                   # 選択された TileType (0-33)
    reward: float                 # 即時報酬
    log_prob: float               # 行動選択時の log_prob
    value: float                  # 推論時の value 推定
    terminated: bool              # 半荘終了フラグ
    round_over: bool              # 局終了フラグ
    # メタデータ
    experiment_id: str = ""
    run_id: str = ""
    worker_id: str = ""
    shard_id: str = ""
    model_version: int = 0
    generation: int = 0
    timestamp: float = 0.0
    episode_id: str = ""
    round_id: int = 0
    step_id: int = 0
    player_id: int = 0
    actor_type: str = "policy"  # "policy" or "baseline"


def validate_metadata(sample: LearningSample) -> None:
    """必須メタデータの妥当性検証

    Raises:
        ValueError: 検証に失敗した場合
    """
    for field in _REQUIRED_STR_FIELDS:
        val = getattr(sample, field)
        if not isinstance(val, str) or val == "":
            raise ValueError(f"必須メタデータ '{field}' が空です")
    for field in _REQUIRED_NONNEG_FIELDS:
        val = getattr(sample, field)
        if not isinstance(val, (int, float)) or val < 0:
            raise ValueError(f"必須メタデータ '{field}' が負の値です: {val}")


# --- Backend 抽象 ---

class ShardBackend(ABC):
    """shard 保存形式の抽象インターフェース"""

    @abstractmethod
    def write(self, data: dict, path: Path) -> None:
        """データ辞書をファイルに書き出す"""

    @abstractmethod
    def read(self, path: Path) -> pa.Table:
        """ファイルからテーブルを読み込む"""

    @abstractmethod
    def file_extension(self) -> str:
        """ファイル拡張子（ドット付き）"""


class ParquetBackend(ShardBackend):
    """Parquet 形式の backend"""

    def write(self, data: dict, path: Path) -> None:
        table = pa.table(data)
        pq.write_table(table, path)

    def read(self, path: Path) -> pa.Table:
        return pq.read_table(path)

    def file_extension(self) -> str:
        return ".parquet"


# --- Writer / Reader ---

class ShardWriter:
    """サンプルをバッファリングして shard ファイルに書き出す"""

    def __init__(
        self,
        output_dir: Path,
        max_samples: int = 10000,
        backend: ShardBackend | None = None,
        validate: bool = True,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._max_samples = max_samples
        self._backend = backend or ParquetBackend()
        self._validate = validate
        self._buffer: list[LearningSample] = []
        self._shard_counter = 0

    @property
    def current_shard_name(self) -> str:
        """現在の shard 名（次に flush されるファイルの識別子）"""
        return f"shard_{self._shard_counter:04d}"

    def add(self, sample: LearningSample) -> None:
        """サンプルを追加。max_samples に達したら自動 flush

        shard_id が空の場合、現在の shard 名を自動付与してから検証する。
        validate=True の場合、shard_id 含む全必須メタデータを検証する。
        """
        if sample.shard_id == "":
            sample.shard_id = self.current_shard_name
        if self._validate:
            validate_metadata(sample)
        self._buffer.append(sample)
        if len(self._buffer) >= self._max_samples:
            self.flush()

    def flush(self) -> Path | None:
        """バッファ内のサンプルを shard ファイルに書き出す"""
        if not self._buffer:
            return None

        ext = self._backend.file_extension()
        shard_name = f"shard_{self._shard_counter:04d}"
        path = self._output_dir / f"{shard_name}{ext}"

        data = {
            "observation": [s.observation.astype(np.float32).tobytes() for s in self._buffer],
            "observation_dim": [len(s.observation) for s in self._buffer],
            "legal_mask": [s.legal_mask.astype(np.float32).tobytes() for s in self._buffer],
            "action": [s.action for s in self._buffer],
            "reward": [float(s.reward) for s in self._buffer],
            "log_prob": [float(s.log_prob) for s in self._buffer],
            "value": [float(s.value) for s in self._buffer],
            "terminated": [s.terminated for s in self._buffer],
            "round_over": [s.round_over for s in self._buffer],
            "experiment_id": [s.experiment_id for s in self._buffer],
            "run_id": [s.run_id for s in self._buffer],
            "worker_id": [s.worker_id for s in self._buffer],
            "shard_id": [s.shard_id for s in self._buffer],
            "model_version": [s.model_version for s in self._buffer],
            "generation": [s.generation for s in self._buffer],
            "timestamp": [s.timestamp for s in self._buffer],
            "episode_id": [s.episode_id for s in self._buffer],
            "round_id": [s.round_id for s in self._buffer],
            "step_id": [s.step_id for s in self._buffer],
            "player_id": [s.player_id for s in self._buffer],
            "actor_type": [s.actor_type for s in self._buffer],
        }

        self._backend.write(data, path)
        self._buffer.clear()
        self._shard_counter += 1
        return path

    def close(self) -> None:
        """残りをフラッシュしてクローズ"""
        self.flush()


class ShardReader:
    """shard ファイル群からサンプルを読み込む"""

    def __init__(self, shard_dir: Path, backend: ShardBackend | None = None):
        self._shard_dir = Path(shard_dir)
        self._backend = backend or ParquetBackend()

    def _find_shards(self) -> list[Path]:
        """shard ファイルを検索"""
        ext = self._backend.file_extension()
        return sorted(self._shard_dir.glob(f"shard_*{ext}"))

    def read_all(self) -> list[LearningSample]:
        """全サンプルを LearningSample のリストとして読み込む"""
        samples = []
        for path in self._find_shards():
            table = self._backend.read(path)
            for i in range(len(table)):
                obs_bytes = table.column("observation")[i].as_py()
                obs = np.frombuffer(obs_bytes, dtype=np.float32).copy()

                mask_bytes = table.column("legal_mask")[i].as_py()
                mask = np.frombuffer(mask_bytes, dtype=np.float32).copy()

                samples.append(LearningSample(
                    observation=obs,
                    legal_mask=mask,
                    action=table.column("action")[i].as_py(),
                    reward=table.column("reward")[i].as_py(),
                    log_prob=table.column("log_prob")[i].as_py(),
                    value=table.column("value")[i].as_py(),
                    terminated=table.column("terminated")[i].as_py(),
                    round_over=table.column("round_over")[i].as_py(),
                    experiment_id=table.column("experiment_id")[i].as_py(),
                    run_id=table.column("run_id")[i].as_py(),
                    worker_id=table.column("worker_id")[i].as_py(),
                    shard_id=table.column("shard_id")[i].as_py(),
                    model_version=table.column("model_version")[i].as_py(),
                    generation=table.column("generation")[i].as_py(),
                    timestamp=table.column("timestamp")[i].as_py(),
                    episode_id=table.column("episode_id")[i].as_py(),
                    round_id=table.column("round_id")[i].as_py(),
                    step_id=table.column("step_id")[i].as_py(),
                    player_id=table.column("player_id")[i].as_py(),
                    actor_type=self._read_column_safe(table, "actor_type", i, "policy"),
                ))
        return samples

    @staticmethod
    def _read_column_safe(table: pa.Table, column: str, index: int, default):
        """カラムが存在すれば読み込み、なければデフォルト値を返す"""
        if column in table.column_names:
            return table.column(column)[index].as_py()
        return default

    def read_as_tensors(self, filter_actor_type: str | None = None) -> dict[str, np.ndarray]:
        """バッチ処理用に numpy 配列群として読み込む

        Args:
            filter_actor_type: 指定時、該当 actor_type のサンプルのみ返す
        """
        all_obs = []
        all_masks = []
        all_actions = []
        all_rewards = []
        all_log_probs = []
        all_values = []
        all_terminateds = []
        all_actor_types = []

        for path in self._find_shards():
            table = self._backend.read(path)
            n = len(table)
            for i in range(n):
                obs_bytes = table.column("observation")[i].as_py()
                obs = np.frombuffer(obs_bytes, dtype=np.float32).copy()
                all_obs.append(obs)

                mask_bytes = table.column("legal_mask")[i].as_py()
                mask = np.frombuffer(mask_bytes, dtype=np.float32).copy()
                all_masks.append(mask)

            all_actions.extend(table.column("action").to_pylist())
            all_rewards.extend(table.column("reward").to_pylist())
            all_log_probs.extend(table.column("log_prob").to_pylist())
            all_values.extend(table.column("value").to_pylist())
            all_terminateds.extend(table.column("terminated").to_pylist())
            if "actor_type" in table.column_names:
                all_actor_types.extend(table.column("actor_type").to_pylist())
            else:
                all_actor_types.extend(["policy"] * n)

        if not all_obs:
            return {
                "observations": np.zeros((0, 0), dtype=np.float32),
                "legal_masks": np.zeros((0, 34), dtype=np.float32),
                "actions": np.zeros(0, dtype=np.int32),
                "rewards": np.zeros(0, dtype=np.float32),
                "log_probs": np.zeros(0, dtype=np.float32),
                "values": np.zeros(0, dtype=np.float32),
                "terminateds": np.zeros(0, dtype=bool),
                "actor_types": np.array([], dtype=object),
            }

        result = {
            "observations": np.stack(all_obs),
            "legal_masks": np.stack(all_masks),
            "actions": np.array(all_actions, dtype=np.int32),
            "rewards": np.array(all_rewards, dtype=np.float32),
            "log_probs": np.array(all_log_probs, dtype=np.float32),
            "values": np.array(all_values, dtype=np.float32),
            "terminateds": np.array(all_terminateds, dtype=bool),
            "actor_types": np.array(all_actor_types, dtype=object),
        }

        if filter_actor_type is not None:
            mask = result["actor_types"] == filter_actor_type
            result = {k: v[mask] for k, v in result.items()}

        return result
