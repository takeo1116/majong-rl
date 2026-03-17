"""FlatFeatureEncoder: フラットな固定長数値ベクトル"""
from __future__ import annotations

import numpy as np

from mahjong_rl._mahjong_core import (
    PartialObservation, FullObservation, NUM_TILE_TYPES, NUM_PLAYERS,
)
from .base import FeatureEncoder, EncoderMetadata, Observation

# C++ 高速版 (ビルド済みなら使用)
try:
    from mahjong_rl._mahjong_core import analyze_discards as _analyze_discards_cpp
    _HAS_CPP_ANALYZE = True
except ImportError:
    _HAS_CPP_ANALYZE = False

try:
    from mahjong_rl._mahjong_core import compute_shape_hint as _compute_shape_hint_cpp
    _HAS_CPP_SHAPE = True
except ImportError:
    _HAS_CPP_SHAPE = False


# 手牌形状ヒント次元数 (CQ-0170)
_CHI_KIND_SIZE = 7 * 3          # 21: 各スート 中心牌2-8 の7種 × 3スート
_SERIAL_PAIR_KIND_SIZE = 8 * 3  # 24: 各スート 12,23,...,89 の8種 × 3スート
_INSIDE_WAIT_KIND_SIZE = 7 * 3  # 21: 各スート 中心牌2-8 の7種 × 3スート


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

    shanten_hint=True 追加 (CQ-0119):
      - delta_shanten_sign: 34 (各打牌候補のシャンテン改善/維持/悪化)
      合計: Partial=387, Full=489

    discard_ukeire_hint=True 追加 (CQ-0168):
      - discard_ukeire_norm: 34 (各打牌候補の受け入れ枚数 / 局面内max)
      値域: [0,1], 非合法候補=0.0

    current_shanten_input=True 追加 (CQ-0169):
      - current_shanten / 8.0: 1 (共通 trunk 入力)
      値域: [0,1]

    shape_hint=True 追加 (CQ-0170):
      - closed_chi_hint: 21 (順子 multihot)
      - closed_outside_wait_hint: 24 (塔子 multihot)
      - closed_inside_wait_hint: 21 (嵌張 multihot)
      合計: 66
    """

    # Partial 特徴量の次元
    _PARTIAL_DIM = 34 + 4 * 34 + 4 * 34 + 34 + 5 + 4 + 4  # 353
    # Full 追加分 (自家手牌34 → 4家手牌136 = +102)
    _FULL_EXTRA_DIM = 3 * 34  # 102
    # シャンテン補助特徴の次元 (CQ-0119)
    _SHANTEN_HINT_DIM = 34
    # 打牌候補受け入れ枚数の次元 (CQ-0168)
    _DISCARD_UKEIRE_DIM = 34
    # policy 用 current_shanten の次元 (CQ-0169)
    _CURRENT_SHANTEN_DIM = 1
    # 手牌形状ヒントの次元 (CQ-0170)
    _SHAPE_HINT_DIM = _CHI_KIND_SIZE + _SERIAL_PAIR_KIND_SIZE + _INSIDE_WAIT_KIND_SIZE  # 66
    # turn_context の次元 (CQ-0175): turn_progress(1) + bucket_one_hot(3) = 4
    _TURN_CONTEXT_DIM = 4

    def __init__(self, observation_mode: str = "both",
                 shanten_hint: bool = False,
                 discard_ukeire_hint: bool = False,
                 current_shanten_input: bool = False,
                 shape_hint: bool = False,
                 turn_context: bool = False):
        """
        Args:
            observation_mode: "full", "partial", "both"
            shanten_hint: True でシャンテン補助特徴を追加 (CQ-0119)
            discard_ukeire_hint: True で打牌候補受け入れ枚数を追加 (CQ-0168)
            current_shanten_input: True で current_shanten を共通入力に追加 (CQ-0169)
            shape_hint: True で手牌形状ヒントを追加 (CQ-0170)
            turn_context: True で turn/time 文脈特徴を追加 (CQ-0175)
        """
        self._observation_mode = observation_mode
        self._shanten_hint = shanten_hint
        self._discard_ukeire_hint = discard_ukeire_hint
        self._current_shanten_input = current_shanten_input
        self._shape_hint = shape_hint
        self._turn_context = turn_context

    def encode(self, obs: Observation, *,
               legal_mask: np.ndarray | None = None) -> np.ndarray:
        """Observation を特徴量ベクトルに変換する

        Args:
            obs: PartialObservation or FullObservation
            legal_mask: 合法手マスク (34次元, optional, CQ-0172)。
                discard_ukeire_hint 有効時に渡すと非合法候補を 0.0 にする。
        """
        if isinstance(obs, FullObservation):
            return self._encode_full(obs, legal_mask=legal_mask)
        elif isinstance(obs, PartialObservation):
            return self._encode_partial(obs, legal_mask=legal_mask)
        else:
            raise TypeError(f"未対応の Observation 型: {type(obs)}")

    def metadata(self) -> EncoderMetadata:
        if self._observation_mode == "full":
            base_dim = self._PARTIAL_DIM + self._FULL_EXTRA_DIM
        elif self._observation_mode == "partial":
            base_dim = self._PARTIAL_DIM
        else:
            base_dim = self._PARTIAL_DIM + self._FULL_EXTRA_DIM
        dim = base_dim
        # CQ-0203: feature_ranges を構築
        ranges: dict[str, tuple[int, int]] = {}
        if self._shanten_hint:
            ranges["shanten_hint"] = (dim, dim + self._SHANTEN_HINT_DIM)
            dim += self._SHANTEN_HINT_DIM
        if self._discard_ukeire_hint:
            ranges["discard_ukeire_hint"] = (dim, dim + self._DISCARD_UKEIRE_DIM)
            dim += self._DISCARD_UKEIRE_DIM
        if self._current_shanten_input:
            ranges["current_shanten"] = (dim, dim + self._CURRENT_SHANTEN_DIM)
            dim += self._CURRENT_SHANTEN_DIM
        if self._shape_hint:
            ranges["shape_hint"] = (dim, dim + self._SHAPE_HINT_DIM)
            dim += self._SHAPE_HINT_DIM
        if self._turn_context:
            ranges["turn_context"] = (dim, dim + self._TURN_CONTEXT_DIM)
            dim += self._TURN_CONTEXT_DIM
        return EncoderMetadata(
            output_shape=(dim,),
            dtype=np.dtype(np.float32),
            observation_mode=self._observation_mode,
            name="FlatFeatureEncoder",
            description="フラットな固定長数値ベクトル (MLP向け)",
            feature_ranges=ranges,
        )

    def _encode_partial(self, obs: PartialObservation, *,
                        legal_mask: np.ndarray | None = None) -> np.ndarray:
        features: list[np.ndarray] = []

        # 自家手牌 34種カウント
        hand_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for tid in obs.hand:
            hand_counts[tid // 4] += 1.0
        features.append(hand_counts)
        # 追加特徴量用にコピーを保持
        _need_hand_copy = (self._shanten_hint or self._discard_ukeire_hint
                           or self._current_shanten_input or self._shape_hint)
        hand_counts_for_hint = hand_counts.copy() if _need_hand_copy else None

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

        # シャンテン補助特徴 + 打牌候補受け入れ枚数 (CQ-0119, CQ-0168, CQ-0172)
        if self._shanten_hint or self._discard_ukeire_hint:
            hint, ukeire = self._compute_hint_and_ukeire(
                hand_counts_for_hint, legal_mask)
            if self._shanten_hint:
                features.append(hint)
            if self._discard_ukeire_hint:
                features.append(ukeire)

        # policy 用 current_shanten (CQ-0169)
        if self._current_shanten_input:
            features.append(self._compute_current_shanten(hand_counts_for_hint))

        # 手牌形状ヒント (CQ-0170)
        if self._shape_hint:
            features.append(self._compute_shape_hint(hand_counts_for_hint))

        # turn/time 文脈特徴 (CQ-0175)
        if self._turn_context:
            features.append(self._compute_turn_context(obs.turn_number))

        return np.concatenate(features)

    def _encode_full(self, obs: FullObservation, *,
                     legal_mask: np.ndarray | None = None) -> np.ndarray:
        features: list[np.ndarray] = []

        # 全4家手牌
        _need_current = (self._shanten_hint or self._discard_ukeire_hint
                         or self._current_shanten_input or self._shape_hint)
        current_player = obs.current_player
        hand_counts_current = None  # CQ-0208: current_player の手牌
        for p in range(NUM_PLAYERS):
            hand_counts = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
            for tid in obs.hands[p]:
                hand_counts[tid // 4] += 1.0
            features.append(hand_counts)
            if p == current_player and _need_current:
                hand_counts_current = hand_counts.copy()

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

        # シャンテン補助特徴 + 打牌候補受け入れ枚数 (CQ-0119, CQ-0168, CQ-0172)
        if self._shanten_hint or self._discard_ukeire_hint:
            hint, ukeire = self._compute_hint_and_ukeire(
                hand_counts_current, legal_mask)
            if self._shanten_hint:
                features.append(hint)
            if self._discard_ukeire_hint:
                features.append(ukeire)

        # policy 用 current_shanten (CQ-0169)
        if self._current_shanten_input:
            features.append(self._compute_current_shanten(hand_counts_current))

        # 手牌形状ヒント (CQ-0170)
        if self._shape_hint:
            features.append(self._compute_shape_hint(hand_counts_current))

        # turn/time 文脈特徴 (CQ-0175)
        if self._turn_context:
            features.append(self._compute_turn_context(obs.turn_number))

        return np.concatenate(features)

    @staticmethod
    def _compute_hint_and_ukeire(
        hand_counts: np.ndarray,
        legal_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """shanten_hint と discard_ukeire_norm を一括計算する

        C++ analyze_discards を1回呼び (全候補 mask=all-1)、
        shanten_sign はそのまま、ukeire_norm は legal_mask で絞って再正規化。
        C++ 未ビルド時は Python fallback。

        Args:
            hand_counts: 34種の手牌カウント (float32)
            legal_mask: 合法手マスク (optional, discard_ukeire 用)

        Returns:
            (shanten_sign[34], ukeire_norm[34])
        """
        if _HAS_CPP_ANALYZE:
            counts_list = hand_counts.astype(int).tolist()
            # 1回の C++ 呼び出しで全候補の shanten_sign + acceptance を取得
            result = _analyze_discards_cpp(counts_list, [1] * 34)
            hint = np.array(result["shanten_sign"], dtype=np.float32)

            # ukeire: acceptance を legal_mask で絞り、Python 側で再正規化
            acceptance = np.array(result["acceptance"], dtype=np.float32)
            if legal_mask is not None:
                acceptance = acceptance * (legal_mask >= 0.5)
            max_acc = acceptance.max()
            ukeire = acceptance / max_acc if max_acc > 0 else acceptance

            return hint, ukeire
        else:
            # Python fallback
            hint = FlatFeatureEncoder._compute_shanten_hint(hand_counts.copy())
            ukeire = FlatFeatureEncoder._compute_discard_ukeire(
                hand_counts.copy(), legal_mask=legal_mask)
            return hint, ukeire

    @staticmethod
    def _compute_shanten_hint(hand_counts: np.ndarray) -> np.ndarray:
        """各打牌候補のシャンテン維持/悪化を計算する (CQ-0119, CQ-0123, CQ-0124)

        delta = shanten(手牌) - shanten(手牌 - t) の符号を返す。

        運用値域 (現行 discard 評価):
          0.0 = 維持（最適打牌候補）または手牌に存在しない牌種
         -1.0 = 悪化（シャンテン数が増加する打牌）

        +1 について:
          shanten(n枚) <= shanten(n-1枚) の単調性により、1枚減らして改善する
          ケースは数学的に発生しない。そのため現行の discard 評価では +1.0 は
          出力されない（テストで不在を保証: test_improvement_never_occurs）。
          ただし将来 draw 評価やツモ牌選択など異なる文脈で本関数を流用する
          可能性に備え、delta > 0 分岐はガード節として残している。

        Args:
            hand_counts: 34種の手牌カウント (float32, 一時的に変更→復元)

        Returns:
            delta_shanten_sign[34]: 実質 {-1.0, 0.0} のみ（上記参照）
        """
        from mahjong_rl.baseline.shanten import compute_shanten

        base = compute_shanten(hand_counts)
        hint = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for t in range(NUM_TILE_TYPES):
            if hand_counts[t] >= 1:
                hand_counts[t] -= 1
                after = compute_shanten(hand_counts)
                hand_counts[t] += 1
                delta = base - after  # 正=改善
                # NOTE: delta > 0 は discard 文脈では発生しない（単調性）。
                # 将来の拡張互換のためガード節として残す (CQ-0124)。
                if delta > 0:
                    hint[t] = 1.0
                elif delta < 0:
                    hint[t] = -1.0
        return hint

    @staticmethod
    def _compute_discard_ukeire(hand_counts: np.ndarray, *,
                                legal_mask: np.ndarray | None = None) -> np.ndarray:
        """各打牌候補の受け入れ枚数を計算し、局面内 max で正規化する (CQ-0168, CQ-0172)

        手牌にある牌種のみ計算し、非合法候補は 0.0。
        legal_mask が与えられた場合、mask が 0 の牌種も 0.0 にする。
        正規化: acceptance / max_acceptance (max=0 なら全て 0.0)。
        値域: [0, 1]

        Args:
            hand_counts: 34種の手牌カウント (float32, 一時的に変更→復元)
            legal_mask: 合法手マスク (34次元, optional)。
                与えられた場合 legal_mask[t] < 0.5 の牌種を 0.0 にする。

        Returns:
            discard_ukeire_norm[34]: 正規化受け入れ枚数
        """
        from mahjong_rl.baseline.shanten import compute_shanten

        raw = np.zeros(NUM_TILE_TYPES, dtype=np.float32)
        for t in range(NUM_TILE_TYPES):
            if hand_counts[t] < 1:
                continue
            # legal_mask が指定されていて非合法なら skip (CQ-0172)
            if legal_mask is not None and legal_mask[t] < 0.5:
                continue
            # t を切った後のシャンテン数を計算
            hand_counts[t] -= 1
            sh_after = compute_shanten(hand_counts)
            # 受け入れ枚数: sh_after が下がる牌種の残り枚数合計
            acceptance = 0
            for u in range(NUM_TILE_TYPES):
                if hand_counts[u] >= 4:
                    continue
                hand_counts[u] += 1
                if compute_shanten(hand_counts) < sh_after:
                    acceptance += 4 - int(hand_counts[u] - 1)  # 残り枚数概算
                hand_counts[u] -= 1
            hand_counts[t] += 1
            raw[t] = float(acceptance)

        max_acc = raw.max()
        if max_acc > 0:
            return raw / max_acc
        return raw

    @staticmethod
    def _compute_current_shanten(hand_counts: np.ndarray) -> np.ndarray:
        """current_shanten / 8.0 を共通特徴として返す (CQ-0169)

        Args:
            hand_counts: 34種の手牌カウント (float32)

        Returns:
            [current_shanten / 8.0]: shape=(1,), 値域 [0, 1]
        """
        from mahjong_rl.baseline.shanten import compute_shanten

        sh = compute_shanten(hand_counts)
        return np.array([sh / 8.0], dtype=np.float32)

    @staticmethod
    def _compute_shape_hint(hand_counts: np.ndarray) -> np.ndarray:
        """手牌形状ヒント: 順子/塔子/嵌張の binary multihot (CQ-0170)

        手牌中の閉じた形状を検出する。
        - closed_chi: 連続3牌 (例: 1m2m3m) → 21次元
        - closed_outside_wait: 隣接2牌 (例: 1m2m, 8m9m) → 24次元
        - closed_inside_wait: 1つ飛び2牌 (例: 1m3m) → 21次元

        各スートの数牌のみ対象 (字牌は除外)。
        C++ 版が利用可能な場合はそちらを使用。

        Args:
            hand_counts: 34種の手牌カウント (float32)

        Returns:
            shape_hint[66]: binary multihot
        """
        if _HAS_CPP_SHAPE:
            counts_list = hand_counts.astype(int).tolist()
            result = _compute_shape_hint_cpp(counts_list)
            return np.array(result, dtype=np.float32)

        chi = np.zeros(_CHI_KIND_SIZE, dtype=np.float32)
        outside_wait = np.zeros(_SERIAL_PAIR_KIND_SIZE, dtype=np.float32)
        inside_wait = np.zeros(_INSIDE_WAIT_KIND_SIZE, dtype=np.float32)

        for suit in range(3):  # 萬子, 筒子, 索子
            base = suit * 9  # 牌種インデックスのベース (0, 9, 18)

            # 順子 (closed_chi): 中心牌 2-8 (index 1-7)
            for center in range(1, 8):
                idx = base + center
                if (hand_counts[idx - 1] >= 1 and
                        hand_counts[idx] >= 1 and
                        hand_counts[idx + 1] >= 1):
                    chi[suit * 7 + (center - 1)] = 1.0

            # 塔子 (closed_outside_wait): 隣接2牌ペア (12, 23, ..., 89)
            for pair_start in range(8):  # 0-7 → ペア 12,23,...,89
                idx = base + pair_start
                if (hand_counts[idx] >= 1 and
                        hand_counts[idx + 1] >= 1):
                    outside_wait[suit * 8 + pair_start] = 1.0

            # 嵌張 (closed_inside_wait): 中心牌 2-8 (index 1-7), 間が空く
            for center in range(1, 8):
                idx = base + center
                if (hand_counts[idx - 1] >= 1 and
                        hand_counts[idx] < 1 and
                        hand_counts[idx + 1] >= 1):
                    inside_wait[suit * 7 + (center - 1)] = 1.0

        return np.concatenate([chi, outside_wait, inside_wait])

    @staticmethod
    def _compute_turn_context(turn_number: int) -> np.ndarray:
        """turn/time 文脈特徴を計算する (CQ-0175)

        turn_progress: turn_number / 18.0 (0..1 連続値)
        turn_bucket: early(0-5)/mid(6-11)/late(12-17) の 3次元 one-hot

        Args:
            turn_number: 巡目 (0-based, 最大17)

        Returns:
            turn_context[4]: [turn_progress, early, mid, late]
        """
        progress = min(turn_number / 18.0, 1.0)
        early = 1.0 if turn_number <= 5 else 0.0
        mid = 1.0 if 6 <= turn_number <= 11 else 0.0
        late = 1.0 if turn_number >= 12 else 0.0
        return np.array([progress, early, mid, late], dtype=np.float32)
