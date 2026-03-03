"""ChannelTensorEncoder: チャネル分割した固定テンソル (CNN向け)"""
from __future__ import annotations

import numpy as np

from mahjong_rl._mahjong_core import (
    PartialObservation, FullObservation, NUM_TILE_TYPES, NUM_PLAYERS,
)
from .base import FeatureEncoder, EncoderMetadata, Observation

# テンソル形状: (C, 4, 9)
# 行: スート (萬子=0, 筒子=1, 索子=2, 字牌=3)
# 列: 数字 (0-8), 字牌は 東=0 南=1 西=2 北=3 白=4 發=5 中=6 (残り2列は0)
_ROWS = 4
_COLS = 9


def _tile_type_to_rc(tile_type: int) -> tuple[int, int]:
    """TileType (0-33) を (row, col) に変換する"""
    if tile_type < 9:
        return 0, tile_type        # 萬子
    elif tile_type < 18:
        return 1, tile_type - 9    # 筒子
    elif tile_type < 27:
        return 2, tile_type - 18   # 索子
    else:
        return 3, tile_type - 27   # 字牌


class ChannelTensorEncoder(FeatureEncoder):
    """チャネルテンソルエンコーダ

    Observation を (C, 4, 9) テンソルに変換する。
    CNN 系モデル向け。

    チャネル構成 (Partial):
      ch 0-3: 自家手牌 (1枚目〜4枚目 binary)
      ch 4-7: 4家河カウント
      ch 8-11: 4家副露カウント
      ch 12: ドラ表示牌カウント
      ch 13: スカラー (broadcast fill: round_number/8)
      ch 14: スカラー (broadcast fill: turn_number/18)
      ch 15: スカラー (broadcast fill: honba/10)
      合計: 16ch

    Full 追加:
      ch 16-31: 全4家手牌 × 4枚 binary (4家 × 4ch = 16ch)
      合計: 32ch
    """

    _PARTIAL_CHANNELS = 16
    _FULL_CHANNELS = 32

    def __init__(self, observation_mode: str = "both"):
        self._observation_mode = observation_mode

    def encode(self, obs: Observation) -> np.ndarray:
        if isinstance(obs, FullObservation):
            return self._encode_full(obs)
        elif isinstance(obs, PartialObservation):
            return self._encode_partial(obs)
        else:
            raise TypeError(f"未対応の Observation 型: {type(obs)}")

    def metadata(self) -> EncoderMetadata:
        if self._observation_mode == "full":
            channels = self._FULL_CHANNELS
        elif self._observation_mode == "partial":
            channels = self._PARTIAL_CHANNELS
        else:
            channels = self._FULL_CHANNELS
        return EncoderMetadata(
            output_shape=(channels, _ROWS, _COLS),
            dtype=np.dtype(np.float32),
            observation_mode=self._observation_mode,
            name="ChannelTensorEncoder",
            description="チャネル分割テンソル (CNN向け)",
        )

    def _encode_partial(self, obs: PartialObservation) -> np.ndarray:
        tensor = np.zeros((self._PARTIAL_CHANNELS, _ROWS, _COLS), dtype=np.float32)

        # ch 0-3: 自家手牌 binary planes (1枚目〜4枚目)
        hand_counts = [0] * NUM_TILE_TYPES
        for tid in obs.hand:
            t = tid // 4
            hand_counts[t] += 1
        for t in range(NUM_TILE_TYPES):
            r, c = _tile_type_to_rc(t)
            for k in range(min(hand_counts[t], 4)):
                tensor[k, r, c] = 1.0

        # ch 4-7: 4家河
        for p in range(NUM_PLAYERS):
            for di in obs.discards[p]:
                r, c = _tile_type_to_rc(di.tile // 4)
                tensor[4 + p, r, c] += 1.0

        # ch 8-11: 4家副露
        for p in range(NUM_PLAYERS):
            for meld in obs.public_melds[p]:
                for i in range(meld.tile_count):
                    tiles = meld.tiles
                    if i < len(tiles):
                        r, c = _tile_type_to_rc(tiles[i] // 4)
                        tensor[8 + p, r, c] += 1.0

        # ch 12: ドラ
        for ind in obs.dora_indicators:
            r, c = _tile_type_to_rc(ind // 4)
            tensor[12, r, c] += 1.0

        # ch 13-15: スカラー broadcast
        tensor[13, :, :] = obs.round_number / 8.0
        tensor[14, :, :] = obs.turn_number / 18.0
        tensor[15, :, :] = obs.honba / 10.0

        return tensor

    def _encode_full(self, obs: FullObservation) -> np.ndarray:
        tensor = np.zeros((self._FULL_CHANNELS, _ROWS, _COLS), dtype=np.float32)

        # ch 0-3: 最初のプレイヤー手牌 (current_player) binary planes
        hand_counts = [0] * NUM_TILE_TYPES
        if len(obs.hands) > 0 and len(obs.hands[0]) > 0:
            for tid in obs.hands[0]:
                hand_counts[tid // 4] += 1
        for t in range(NUM_TILE_TYPES):
            r, c = _tile_type_to_rc(t)
            for k in range(min(hand_counts[t], 4)):
                tensor[k, r, c] = 1.0

        # ch 4-7: 4家河
        for p in range(NUM_PLAYERS):
            for di in obs.discards[p]:
                r, c = _tile_type_to_rc(di.tile // 4)
                tensor[4 + p, r, c] += 1.0

        # ch 8-11: 4家副露
        for p in range(NUM_PLAYERS):
            for meld in obs.melds[p]:
                for i in range(meld.tile_count):
                    tiles = meld.tiles
                    if i < len(tiles):
                        r, c = _tile_type_to_rc(tiles[i] // 4)
                        tensor[8 + p, r, c] += 1.0

        # ch 12: ドラ
        for ind in obs.dora_indicators:
            r, c = _tile_type_to_rc(ind // 4)
            tensor[12, r, c] += 1.0

        # ch 13-15: スカラー broadcast
        tensor[13, :, :] = obs.round_number / 8.0
        tensor[14, :, :] = obs.turn_number / 18.0
        tensor[15, :, :] = obs.honba / 10.0

        # ch 16-31: 全4家手牌 × 4枚 binary (4家 × 4ch)
        for p in range(NUM_PLAYERS):
            p_hand_counts = [0] * NUM_TILE_TYPES
            for tid in obs.hands[p]:
                p_hand_counts[tid // 4] += 1
            for t in range(NUM_TILE_TYPES):
                r, c = _tile_type_to_rc(t)
                for k in range(min(p_hand_counts[t], 4)):
                    tensor[16 + p * 4 + k, r, c] = 1.0

        return tensor
