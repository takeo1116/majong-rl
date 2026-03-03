"""シャンテン数計算 (最適化版)"""
from __future__ import annotations

import numpy as np

# 么九牌 (1m,9m,1p,9p,1s,9s,東,南,西,北,白,發,中)
_TERMINALS_AND_HONORS = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]


def compute_shanten(counts: np.ndarray | list[int]) -> int:
    """シャンテン数を計算する (通常形・七対子・国士の最小値)

    Args:
        counts: 34種の牌カウント配列

    Returns:
        シャンテン数 (-1=和了, 0=テンパイ, 1=一向聴, ...)
    """
    if isinstance(counts, np.ndarray):
        c = counts.astype(int).tolist()
    else:
        c = list(counts)

    return min(
        _regular_shanten(c),
        _chiitoitsu_shanten(c),
        _kokushi_shanten(c),
    )


def _kokushi_shanten(counts: list[int]) -> int:
    """国士無双のシャンテン数"""
    kinds = 0
    has_pair = False
    for t in _TERMINALS_AND_HONORS:
        if counts[t] > 0:
            kinds += 1
            if counts[t] >= 2:
                has_pair = True
    return 13 - kinds - (1 if has_pair else 0)


def _chiitoitsu_shanten(counts: list[int]) -> int:
    """七対子のシャンテン数"""
    pairs = 0
    for c in counts:
        if c >= 2:
            pairs += 1
    return 6 - pairs


def _regular_shanten(counts: list[int]) -> int:
    """通常形 (4面子1雀頭) のシャンテン数

    雀頭を取る/取らないで分岐し、各分岐で面子分解を探索する。
    """
    best = [8]
    c = list(counts)

    # 雀頭なしで探索
    _remove_groups(c, 0, 0, 0, best)

    # 各牌種を雀頭として取って探索
    for t in range(34):
        if c[t] >= 2:
            c[t] -= 2
            _remove_groups(c, 0, 0, 1, best)
            c[t] += 2

    return best[0]


def _remove_groups(
    counts: list[int],
    pos: int,
    mentsu: int,
    jantai: int,
    best: list[int],
) -> None:
    """完成面子 (刻子・順子) を抽出し、残りから不完全面子を数える

    Args:
        counts: 牌カウント (破壊的変更・復元)
        pos: 開始位置
        mentsu: 抽出済み完成面子数
        jantai: 雀頭数 (0 or 1)
        best: 最善シャンテン数
    """
    # 枝刈り: これ以上面子を取っても改善しない
    if mentsu >= 4:
        partial = _count_partial(counts)
        # 4面子 + partial は意味なし (面子は4で十分)
        shanten = 8 - 2 * 4 - 0 - jantai
        best[0] = min(best[0], shanten)
        return

    # 残り牌がある位置を探す (完成面子を取れる位置)
    idx = pos
    while idx < 34 and counts[idx] == 0:
        idx += 1

    if idx >= 34:
        # 全位置を処理済み → 不完全面子を数えて shanten を計算
        partial = _count_partial(counts)
        max_partial = 4 - mentsu
        partial = min(partial, max_partial)
        shanten = 8 - 2 * mentsu - partial - jantai
        best[0] = min(best[0], shanten)
        return

    # 枝刈り: 残りで最大改善しても best[0] 以下にならない場合
    remaining_tiles = 0
    for i in range(idx, 34):
        remaining_tiles += counts[i]
    max_more_mentsu = remaining_tiles // 3
    max_total_mentsu = min(4, mentsu + max_more_mentsu)
    lower_bound = 8 - 2 * max_total_mentsu - (4 - max_total_mentsu) - jantai
    if lower_bound >= best[0]:
        return

    # 刻子
    if counts[idx] >= 3:
        counts[idx] -= 3
        _remove_groups(counts, idx, mentsu + 1, jantai, best)
        counts[idx] += 3

    # 順子 (数牌のみ)
    suit = idx // 9
    rel = idx % 9
    if suit < 3 and rel <= 6:
        base = suit * 9
        if counts[base + rel + 1] > 0 and counts[base + rel + 2] > 0:
            counts[idx] -= 1
            counts[base + rel + 1] -= 1
            counts[base + rel + 2] -= 1
            _remove_groups(counts, idx, mentsu + 1, jantai, best)
            counts[idx] += 1
            counts[base + rel + 1] += 1
            counts[base + rel + 2] += 1

    # この位置で面子を取らずに次へ
    _remove_groups(counts, idx + 1, mentsu, jantai, best)


def _count_partial(counts: list[int]) -> int:
    """残り牌から不完全面子 (対子・ターツ) を貪欲に数える"""
    c = list(counts)
    partial = 0

    for suit in range(3):
        base = suit * 9
        # 両面・嵌張ターツと対子を貪欲に取る
        for i in range(9):
            t = base + i
            if c[t] >= 2:
                c[t] -= 2
                partial += 1

        for i in range(8):
            t = base + i
            if c[t] > 0 and c[t + 1] > 0:
                c[t] -= 1
                c[t + 1] -= 1
                partial += 1

        for i in range(7):
            t = base + i
            if c[t] > 0 and c[t + 2] > 0:
                c[t] -= 1
                c[t + 2] -= 1
                partial += 1

    # 字牌の対子
    for t in range(27, 34):
        if c[t] >= 2:
            partial += 1

    return partial
