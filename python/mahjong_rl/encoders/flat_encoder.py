"""FlatFeatureEncoder: フラットな固定長数値ベクトル"""
from __future__ import annotations

import numpy as np

from mahjong_rl._mahjong_core import (
    PartialObservation, FullObservation, NUM_TILE_TYPES, NUM_PLAYERS,
)
from .base import FeatureEncoder, EncoderMetadata, Observation


class FlatFeatureEncoder(FeatureEncoder):
    """フラット特徴量エンコーダ

    Observation をフラットな固定長 float32 ベクトルに変換する。
    MLP 系モデル向け。

    Partial 特徴量構成:
      - 自家手牌 34種カウント (34)
      - 4家河 4×34 (136)
      - 4家副露 4×34 (136)
      - ドラ表示牌 34 (34)
      - スカラー: round/dealer/honba/kyotaku/turn (5)
      - スコア: 4家 (4)
      - 立直宣言: 4家 (4)
      合計: 353

    Full 追加:
      - 残り3家手牌 3×34 (102)  ※ Partial の自家手牌を4家手牌に置換
      合計: 353 + 102 = 455
    """

    # Partial 特徴量の次元
    _PARTIAL_DIM = 34 + 4 * 34 + 4 * 34 + 34 + 5 + 4 + 4  # 353
    # Full 追加分 (自家手牌34 → 4家手牌136 = +102)
    _FULL_EXTRA_DIM = 3 * 34  # 102

    def __init__(self, observation_mode: str = "both"):
        """
        Args:
            observation_mode: "full", "partial", "both"
        """
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
            dim = self._PARTIAL_DIM + self._FULL_EXTRA_DIM
        elif self._observation_mode == "partial":
            dim = self._PARTIAL_DIM
        else:
            # "both" の場合は Full 側の次元を返す（大きい方）
            dim = self._PARTIAL_DIM + self._FULL_EXTRA_DIM
        return EncoderMetadata(
            output_shape=(dim,),
            dtype=np.dtype(np.float32),
            observation_mode=self._observation_mode,
            name="FlatFeatureEncoder",
            description="フラットな固定長数値ベクトル (MLP向け)",
        )

    def _encode_partial(self, obs: PartialObservation) -> np.ndarray:
        features: list[np.ndarray] = []

        # 自家手牌 34種カウント
        hand_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for tid in obs.hand:
            hand_counts[tid // 4] += 1.0
        features.append(hand_counts)

        # 4家河
        for p in range(NUM_PLAYERS):
            discard_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
            for di in obs.discards[p]:
                discard_counts[di.tile // 4] += 1.0
            features.append(discard_counts)

        # 4家副露
        for p in range(NUM_PLAYERS):
            meld_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
            for meld in obs.public_melds[p]:
                for i in range(meld.tile_count):
                    tiles = meld.tiles
                    if i < len(tiles):
                        meld_counts[tiles[i] // 4] += 1.0
            features.append(meld_counts)

        # ドラ表示牌
        dora_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for ind in obs.dora_indicators:
            dora_counts[ind // 4] += 1.0
        features.append(dora_counts)

        # スカラー特徴量
        scalars = np.array([
            obs.round_number / 8.0,
            obs.dealer / 3.0,
            obs.honba / 10.0,
            obs.kyotaku / 10.0,
            obs.turn_number / 18.0,
        ], dtype=np.float32)
        features.append(scalars)

        # スコア
        scores = np.array(
            [obs.scores[p] / 100000.0 for p in range(NUM_PLAYERS)],
            dtype=np.float32,
        )
        features.append(scores)

        # 立直宣言
        riichi = np.array(
            [1.0 if obs.riichi_declared[p] else 0.0 for p in range(NUM_PLAYERS)],
            dtype=np.float32,
        )
        features.append(riichi)

        return np.concatenate(features)

    def _encode_full(self, obs: FullObservation) -> np.ndarray:
        features: list[np.ndarray] = []

        # 全4家手牌
        for p in range(NUM_PLAYERS):
            hand_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
            for tid in obs.hands[p]:
                hand_counts[tid // 4] += 1.0
            features.append(hand_counts)

        # 4家河
        for p in range(NUM_PLAYERS):
            discard_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
            for di in obs.discards[p]:
                discard_counts[di.tile // 4] += 1.0
            features.append(discard_counts)

        # 4家副露
        for p in range(NUM_PLAYERS):
            meld_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
            for meld in obs.melds[p]:
                for i in range(meld.tile_count):
                    tiles = meld.tiles
                    if i < len(tiles):
                        meld_counts[tiles[i] // 4] += 1.0
            features.append(meld_counts)

        # ドラ表示牌
        dora_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for ind in obs.dora_indicators:
            dora_counts[ind // 4] += 1.0
        features.append(dora_counts)

        # スカラー特徴量
        scalars = np.array([
            obs.round_number / 8.0,
            obs.dealer / 3.0,
            obs.honba / 10.0,
            obs.kyotaku / 10.0,
            obs.turn_number / 18.0,
        ], dtype=np.float32)
        features.append(scalars)

        # スコア
        scores = np.array(
            [obs.scores[p] / 100000.0 for p in range(NUM_PLAYERS)],
            dtype=np.float32,
        )
        features.append(scores)

        # 立直宣言 (FullObservation では match_state から取得可能だが簡易版)
        # FullObservation には riichi_declared がないため players から取得
        # → FullObservation には直接含まれない → 0埋め
        riichi = np.zeros(NUM_PLAYERS, dtype=np.float32)
        features.append(riichi)

        return np.concatenate(features)
