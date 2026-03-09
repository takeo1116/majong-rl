"""Reward shaping: shanten delta と schedule ユーティリティ (CQ-0139, CQ-0140)"""
from __future__ import annotations

from mahjong_rl.baseline.shanten import compute_shanten


class ShantenDeltaTracker:
    """プレイヤーごとのシャンテン追跡による shanten_delta_reward 計算

    各プレイヤーの打牌後13枚手牌のシャンテンを追跡し、
    前回打牌後との差分を reward 化する。
    """

    _VALID_MODES = {"both", "improve_only"}

    def __init__(self, scale: float = 0.01, mode: str = "both"):
        """
        Args:
            scale: シャンテン delta に掛ける倍率
            mode: "both"（改善/悪化の両方）or "improve_only"（改善のみ正、悪化は 0）
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Unknown shanten_delta mode: {mode!r}. 有効値: {self._VALID_MODES}")
        self._scale = scale
        self._mode = mode
        self._player_shanten: list[int | None] = [None, None, None, None]

    def reset(self) -> None:
        """match 単位でリセットする"""
        self._player_shanten = [None, None, None, None]

    def compute_reward(
        self, player: int, hand_tile_ids: list[int], discard_tile_type: int,
    ) -> tuple[float, float]:
        """打牌に対する shanten delta reward と raw delta を計算する (CQ-0148)

        Args:
            player: プレイヤー index (0-3)
            hand_tile_ids: 打牌前の手牌 TileId リスト (14枚)
            discard_tile_type: 打牌する牌種 (0-33)

        Returns:
            (shaping_reward, raw_delta) のタプル
            - shaping_reward: mode/scale 適用済みの reward 値
            - raw_delta: mode/scale/schedule 非依存の生シャンテン差分
              (prev_shanten - next_shanten, 正=改善, 0=維持, 負=悪化)
              初回打牌 (prev 未定義) は NaN
        """
        counts = [0] * 34
        for tid in hand_tile_ids:
            counts[tid // 4] += 1
        counts[discard_tile_type] -= 1  # 13枚
        sh_after = compute_shanten(counts)

        prev = self._player_shanten[player]
        self._player_shanten[player] = sh_after

        if prev is None:
            return 0.0, float("nan")  # CQ-0149: 初回打牌は raw delta 未定義

        raw_delta = prev - sh_after  # 正 = 改善（mode/scale 非依存）

        # mode 適用: reward 用のみ
        reward_delta = raw_delta
        if self._mode == "improve_only" and reward_delta < 0:
            reward_delta = 0

        return reward_delta * self._scale, float(raw_delta)


class RewardSchedule:
    """Reward shaping のスケール schedule

    training progress に応じて shaping の強度を変化させる。
    """

    _VALID_TYPES = {"constant", "linear_decay"}

    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}
        self._type = config.get("type", "constant")
        if self._type not in self._VALID_TYPES:
            raise ValueError(
                f"Unknown schedule type: {self._type!r}. 有効値: {self._VALID_TYPES}")

    def get_scale_factor(self, progress: float) -> float:
        """progress ∈ [0, 1] に対するスケール係数を返す

        constant: 常に 1.0
        linear_decay: 1.0 → 0.0 に線形減衰
        """
        progress = max(0.0, min(1.0, progress))
        if self._type == "constant":
            return 1.0
        else:  # linear_decay
            return 1.0 - progress
