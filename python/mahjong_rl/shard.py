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
    teacher_best_mask: np.ndarray | None = None  # (34,) float32, 教師最良候補集合 (CQ-0125)
    shanten_delta: float | None = None  # shanten 変化量 (CQ-0145)
    current_shanten: int | None = None  # 現在シャンテン数 (CQ-0151)
    turn_number: int | None = None  # 巡目 (CQ-0156)
    point_delta_reward: float | None = None  # 点数差分報酬成分 (CQ-0160)
    shanten_delta_reward: float | None = None  # シャンテン差分報酬成分 (CQ-0160)
    is_post_riichi_discard: bool | None = None  # 立直後打牌フラグ (CQ-0163)


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

        # teacher_best_mask: 1つでも非 None があればカラム書き出し (CQ-0125)
        if any(s.teacher_best_mask is not None for s in self._buffer):
            data["teacher_best_mask"] = [
                s.teacher_best_mask.astype(np.float32).tobytes()
                if s.teacher_best_mask is not None
                else np.zeros(34, dtype=np.float32).tobytes()
                for s in self._buffer
            ]

        # shanten_delta: 1つでも非 None があればカラム書き出し (CQ-0145)
        if any(s.shanten_delta is not None for s in self._buffer):
            data["shanten_delta"] = [
                float(s.shanten_delta) if s.shanten_delta is not None else 0.0
                for s in self._buffer
            ]

        # current_shanten: 1つでも非 None があればカラム書き出し (CQ-0151)
        if any(s.current_shanten is not None for s in self._buffer):
            data["current_shanten"] = [
                int(s.current_shanten) if s.current_shanten is not None else -1
                for s in self._buffer
            ]

        # turn_number: 1つでも非 None があればカラム書き出し (CQ-0156)
        if any(s.turn_number is not None for s in self._buffer):
            data["turn_number"] = [
                int(s.turn_number) if s.turn_number is not None else -1
                for s in self._buffer
            ]

        # point_delta_reward: 1つでも非 None があればカラム書き出し (CQ-0160)
        if any(s.point_delta_reward is not None for s in self._buffer):
            data["point_delta_reward"] = [
                float(s.point_delta_reward) if s.point_delta_reward is not None else 0.0
                for s in self._buffer
            ]

        # shanten_delta_reward: 1つでも非 None があればカラム書き出し (CQ-0160)
        if any(s.shanten_delta_reward is not None for s in self._buffer):
            data["shanten_delta_reward"] = [
                float(s.shanten_delta_reward) if s.shanten_delta_reward is not None else 0.0
                for s in self._buffer
            ]

        # is_post_riichi_discard: 1つでも非 None があればカラム書き出し (CQ-0163)
        if any(s.is_post_riichi_discard is not None for s in self._buffer):
            data["is_post_riichi_discard"] = [
                int(s.is_post_riichi_discard) if s.is_post_riichi_discard is not None else -1
                for s in self._buffer
            ]

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
        """shard ファイルを検索

        平坦な shard_*{ext} と worker_*/shard_*{ext} の両方を探索する。
        """
        ext = self._backend.file_extension()
        flat = self._shard_dir.glob(f"shard_*{ext}")
        nested = self._shard_dir.glob(f"worker_*/shard_*{ext}")
        return sorted(set(flat) | set(nested))

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
        all_teacher_best_masks: list[np.ndarray | None] = []
        has_teacher_best_mask = False
        # CQ-0128: shard ごとの teacher_best_mask 有無カウント
        tbm_shard_count = 0
        tbm_shard_total = 0
        # CQ-0145, CQ-0146: shanten_delta（欠落は NaN で表現）
        all_shanten_deltas: list[float] = []
        has_shanten_delta = False
        shanten_delta_shard_count = 0
        shanten_delta_shard_total = 0
        # CQ-0151: current_shanten
        all_current_shantens: list[int] = []
        has_current_shanten = False
        # CQ-0156: turn_number
        all_turn_numbers: list[int] = []
        has_turn_number = False
        # CQ-0160: reward components
        all_point_delta_rewards: list[float] = []
        has_point_delta_reward = False
        all_shanten_delta_rewards: list[float] = []
        has_shanten_delta_reward = False
        # CQ-0163: is_post_riichi_discard
        all_is_post_riichi_discards: list[int] = []
        has_is_post_riichi_discard = False

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

            # teacher_best_mask (CQ-0125): カラム存在時のみ読み込み
            tbm_shard_total += 1
            if "teacher_best_mask" in table.column_names:
                has_teacher_best_mask = True
                tbm_shard_count += 1
                for i in range(n):
                    tbm_bytes = table.column("teacher_best_mask")[i].as_py()
                    tbm = np.frombuffer(tbm_bytes, dtype=np.float32).copy()
                    all_teacher_best_masks.append(tbm)
            else:
                all_teacher_best_masks.extend([None] * n)

            # shanten_delta (CQ-0145, CQ-0146): カラム存在時のみ読み込み、欠落は NaN
            shanten_delta_shard_total += 1
            if "shanten_delta" in table.column_names:
                has_shanten_delta = True
                shanten_delta_shard_count += 1
                all_shanten_deltas.extend(table.column("shanten_delta").to_pylist())
            else:
                all_shanten_deltas.extend([float("nan")] * n)

            # current_shanten (CQ-0151): カラム存在時のみ読み込み
            if "current_shanten" in table.column_names:
                has_current_shanten = True
                all_current_shantens.extend(table.column("current_shanten").to_pylist())
            else:
                all_current_shantens.extend([-1] * n)

            # turn_number (CQ-0156): カラム存在時のみ読み込み
            if "turn_number" in table.column_names:
                has_turn_number = True
                all_turn_numbers.extend(table.column("turn_number").to_pylist())
            else:
                all_turn_numbers.extend([-1] * n)

            # point_delta_reward (CQ-0160): カラム存在時のみ読み込み
            if "point_delta_reward" in table.column_names:
                has_point_delta_reward = True
                all_point_delta_rewards.extend(table.column("point_delta_reward").to_pylist())
            else:
                all_point_delta_rewards.extend([float("nan")] * n)

            # shanten_delta_reward (CQ-0160): カラム存在時のみ読み込み
            if "shanten_delta_reward" in table.column_names:
                has_shanten_delta_reward = True
                all_shanten_delta_rewards.extend(table.column("shanten_delta_reward").to_pylist())
            else:
                all_shanten_delta_rewards.extend([float("nan")] * n)

            # is_post_riichi_discard (CQ-0163): カラム存在時のみ読み込み
            if "is_post_riichi_discard" in table.column_names:
                has_is_post_riichi_discard = True
                all_is_post_riichi_discards.extend(table.column("is_post_riichi_discard").to_pylist())
            else:
                all_is_post_riichi_discards.extend([-1] * n)

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
                "teacher_best_masks": None,
                "teacher_best_mask_shard_info": {"available": 0, "total": 0},
                "shanten_deltas": None,
                "shanten_delta_shard_info": {"available": 0, "total": 0},
                "current_shantens": None,
                "turn_numbers": None,
                "point_delta_rewards": None,
                "shanten_delta_rewards": None,
                "is_post_riichi_discards": None,
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

        # teacher_best_masks (CQ-0125, CQ-0191): 判定を filter 後に遅延
        # all_teacher_best_masks を raw list のまま保持し、filter 後に最終判定する
        result["_raw_teacher_best_masks"] = all_teacher_best_masks
        result["_has_teacher_best_mask"] = has_teacher_best_mask
        # CQ-0128: shard ごとの有無情報
        result["teacher_best_mask_shard_info"] = {
            "available": tbm_shard_count,
            "total": tbm_shard_total,
        }

        # shanten_deltas (CQ-0145, CQ-0146): カラムがある shard があれば配列化（欠落は NaN）
        if has_shanten_delta:
            result["shanten_deltas"] = np.array(all_shanten_deltas, dtype=np.float32)
        else:
            result["shanten_deltas"] = None
        result["shanten_delta_shard_info"] = {
            "available": shanten_delta_shard_count,
            "total": shanten_delta_shard_total,
        }

        # current_shantens (CQ-0151): 全 shard にカラムがある場合のみ配列化
        if has_current_shanten and all(v >= 0 for v in all_current_shantens):
            result["current_shantens"] = np.array(all_current_shantens, dtype=np.int32)
        else:
            result["current_shantens"] = None

        # turn_numbers (CQ-0156): 全 shard にカラムがある場合のみ配列化
        if has_turn_number and all(v >= 0 for v in all_turn_numbers):
            result["turn_numbers"] = np.array(all_turn_numbers, dtype=np.int32)
        else:
            result["turn_numbers"] = None

        # point_delta_rewards (CQ-0160): カラムがある shard があれば配列化（欠落は NaN）
        if has_point_delta_reward:
            result["point_delta_rewards"] = np.array(all_point_delta_rewards, dtype=np.float32)
        else:
            result["point_delta_rewards"] = None

        # shanten_delta_rewards (CQ-0160): カラムがある shard があれば配列化（欠落は NaN）
        if has_shanten_delta_reward:
            result["shanten_delta_rewards"] = np.array(all_shanten_delta_rewards, dtype=np.float32)
        else:
            result["shanten_delta_rewards"] = None

        # is_post_riichi_discards (CQ-0163): 全 shard にカラムがある場合のみ配列化
        if has_is_post_riichi_discard and all(v >= 0 for v in all_is_post_riichi_discards):
            result["is_post_riichi_discards"] = np.array(all_is_post_riichi_discards, dtype=np.bool_)
        else:
            result["is_post_riichi_discards"] = None

        if filter_actor_type is not None:
            actor_mask = result["actor_types"] == filter_actor_type
            raw_tbm = result.pop("_raw_teacher_best_masks")
            has_tbm = result.pop("_has_teacher_best_mask")
            shard_info = result.pop("teacher_best_mask_shard_info")
            sd = result.pop("shanten_deltas")
            sd_shard_info = result.pop("shanten_delta_shard_info")
            cs = result.pop("current_shantens")
            tn = result.pop("turn_numbers")
            pdr = result.pop("point_delta_rewards")
            sdr = result.pop("shanten_delta_rewards")
            iprd = result.pop("is_post_riichi_discards")
            result = {k: v[actor_mask] for k, v in result.items()}
            # CQ-0191: filter 後の行で teacher_best_masks を判定
            filtered_tbm = [raw_tbm[i] for i, m in enumerate(actor_mask) if m]
            if has_tbm and filtered_tbm and all(m is not None for m in filtered_tbm):
                result["teacher_best_masks"] = np.stack(filtered_tbm)
            else:
                result["teacher_best_masks"] = None
            result["teacher_best_mask_shard_info"] = shard_info
            result["shanten_deltas"] = sd[actor_mask] if sd is not None else None
            result["shanten_delta_shard_info"] = sd_shard_info
            result["current_shantens"] = cs[actor_mask] if cs is not None else None
            result["turn_numbers"] = tn[actor_mask] if tn is not None else None
            result["point_delta_rewards"] = pdr[actor_mask] if pdr is not None else None
            result["shanten_delta_rewards"] = sdr[actor_mask] if sdr is not None else None
            result["is_post_riichi_discards"] = iprd[actor_mask] if iprd is not None else None
        else:
            # non-filter: raw list を最終化 (CQ-0191)
            raw_tbm = result.pop("_raw_teacher_best_masks")
            has_tbm = result.pop("_has_teacher_best_mask")
            if has_tbm and all(m is not None for m in raw_tbm):
                result["teacher_best_masks"] = np.stack(raw_tbm)
            else:
                result["teacher_best_masks"] = None

        return result
