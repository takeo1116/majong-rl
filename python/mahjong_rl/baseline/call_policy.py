"""CQ-0224: Stage2a ルールベース副露方策

legal candidates (Chi / Pon / Daiminkan / Skip) から1つ選ぶ。
Stage2a v1 では actor / teacher を分けず、選択結果をそのまま
imitation 正解ラベルとして使う。

方針:
- 役牌ポン → 優先
- 喰いタン方向 → 手牌が断么九条件を崩さないなら鳴く
- 対々和方向 → 刻子候補が多ければポン優先
- 一気通貫 / 三色同順方向 → チー候補が形成に寄与すれば鳴く
- 曖昧な局面 → Skip
"""
from __future__ import annotations

import numpy as np

from mahjong_rl._mahjong_core import ActionType, NUM_TILE_TYPES
from mahjong_rl.env.response_candidate import ResponseCandidate
from mahjong_rl.baseline.shanten import compute_shanten


class RuleBasedCallPolicy:
    """ルールベース副露方策 (Stage2a v1)"""

    def select_action(
        self,
        candidates: list[ResponseCandidate],
        hand_tile_ids: list[int],
    ) -> int:
        """legal candidates から1つ選んで index を返す

        Args:
            candidates: extract_response_candidates() の返り値
            hand_tile_ids: 現在の手牌 TileId リスト

        Returns:
            選択した candidate の index
        """
        if len(candidates) <= 1:
            return 0  # Skip のみ or 候補なし

        hand_counts = self._tile_counts(hand_tile_ids)
        current_shanten = compute_shanten(hand_counts)

        best_idx = len(candidates) - 1  # default: Skip (末尾)
        best_score = 0  # Skip の score

        for i, cand in enumerate(candidates):
            if cand.action_type == ActionType.Skip:
                continue

            score = self._evaluate_candidate(
                cand, hand_counts, current_shanten, hand_tile_ids)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _evaluate_candidate(
        self,
        cand: ResponseCandidate,
        hand_counts: list[int],
        current_shanten: int,
        hand_tile_ids: list[int],
    ) -> int:
        """candidate のスコアを計算 (高いほど良い、0以下は Skip)"""
        score = 0
        tt = cand.tile_type

        if cand.action_type == ActionType.Pon:
            score = self._score_pon(tt, hand_counts, current_shanten)
        elif cand.action_type == ActionType.Daiminkan:
            score = self._score_daiminkan(tt, hand_counts)
        elif cand.action_type == ActionType.Chi:
            score = self._score_chi(cand, hand_counts, current_shanten)

        return score

    def _score_pon(
        self, tt: int, hand_counts: list[int], current_shanten: int,
    ) -> int:
        """ポンのスコア"""
        # 役牌ポンは常に高スコア
        if self._is_yakuhai(tt):
            return 100

        after_counts = list(hand_counts)
        after_counts[tt] -= 2  # ポン消費

        # 対々和方向: 残り手に刻子/対子候補が多い
        koutsu_count = sum(1 for c in after_counts if c >= 3)
        pair_count = sum(1 for c in after_counts if c >= 2)
        if koutsu_count + pair_count >= 2:
            return 60  # 対々和方向

        # 喰いタン方向: 全て中張牌
        if not self._is_yaochu(tt) and self._is_tanyao_compatible(after_counts, tt):
            return 50

        # 風牌・么九牌のポンは低優先
        if self._is_yaochu(tt):
            return 0

        # 中張牌のポンは少し価値あり
        return 15

    def _score_daiminkan(self, tt: int, hand_counts: list[int]) -> int:
        """大明槓のスコア"""
        # 役牌なら高スコア
        if self._is_yakuhai(tt):
            return 90
        # 断么九互換なら中程度
        if not self._is_yaochu(tt):
            return 20
        return 0

    def _score_chi(
        self,
        cand: ResponseCandidate,
        hand_counts: list[int],
        current_shanten: int,
    ) -> int:
        """チーのスコア"""
        tt = cand.tile_type
        consumed = cand.consumed_tile_ids
        if not consumed:
            return 0

        consumed_types = [t // 4 for t in consumed]
        chi_types = sorted(consumed_types + [tt])

        # チー後の残り手牌
        after_counts = list(hand_counts)
        for ct in consumed_types:
            if after_counts[ct] > 0:
                after_counts[ct] -= 1

        # 一気通貫 / 三色同順の方向チェック (高優先)
        if self._contributes_to_ikkitsuukan(chi_types, after_counts):
            return 70
        if self._contributes_to_sanshoku(chi_types, after_counts):
            return 70

        # 喰いタン方向
        all_types = consumed_types + [tt]
        if all(not self._is_yaochu(t) for t in all_types):
            if self._is_tanyao_compatible(after_counts, -1):
                return 40

        return 0  # 方向不明 → Skip

    # ---------- helpers ----------

    @staticmethod
    def _tile_counts(tile_ids: list[int]) -> list[int]:
        counts = [0] * NUM_TILE_TYPES
        for tid in tile_ids:
            counts[tid // 4] += 1
        return counts

    @staticmethod
    def _is_yakuhai(tt: int) -> bool:
        """役牌 (三元牌) か"""
        return tt >= 31  # 白=31, 發=32, 中=33

    @staticmethod
    def _is_yaochu(tt: int) -> bool:
        if tt >= 27:
            return True  # 字牌
        num = (tt % 9) + 1
        return num == 1 or num == 9

    @staticmethod
    def _is_tanyao_compatible(counts: list[int], extra_tt: int) -> bool:
        """手牌 + 副露が断么九条件を満たせるか (簡易判定)"""
        for t in range(NUM_TILE_TYPES):
            if counts[t] > 0 and RuleBasedCallPolicy._is_yaochu(t):
                return False
        if extra_tt >= 0 and RuleBasedCallPolicy._is_yaochu(extra_tt):
            return False
        return True

    @staticmethod
    def _contributes_to_ikkitsuukan(
        chi_types: list[int], after_counts: list[int],
    ) -> bool:
        """この chi が一気通貫に寄与するか (簡易判定)"""
        if not chi_types:
            return False
        suit = chi_types[0] // 9
        if suit >= 3:
            return False
        # このスートで 123/456/789 のいずれかを形成
        base = suit * 9
        shuntsu_starts = set()
        # chi で作った順子の先頭
        first = min(chi_types)
        shuntsu_starts.add(first)
        # 手牌中の既存順子
        for s in range(7):
            t = base + s
            if after_counts[t] >= 1 and after_counts[t+1] >= 1 and after_counts[t+2] >= 1:
                shuntsu_starts.add(t)
        needed = {base, base + 3, base + 6}
        return len(shuntsu_starts & needed) >= 2

    @staticmethod
    def _contributes_to_sanshoku(
        chi_types: list[int], after_counts: list[int],
    ) -> bool:
        """この chi が三色同順に寄与するか (簡易判定)"""
        if not chi_types:
            return False
        first = min(chi_types)
        num = first % 9
        # 3 スートでこの数字の順子があるか
        suits_with_shuntsu = 0
        for suit in range(3):
            t = suit * 9 + num
            if t + 2 < 27:
                if (after_counts[t] >= 1 and after_counts[t+1] >= 1
                        and after_counts[t+2] >= 1):
                    suits_with_shuntsu += 1
        # chi で作った順子のスート
        chi_suit = first // 9
        if chi_suit < 3:
            suits_with_shuntsu += 1  # chi 自体
        return suits_with_shuntsu >= 2
