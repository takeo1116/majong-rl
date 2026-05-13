"""CQ-0245: candidate encoding — 定義は1箇所、batch/single で wrapper

learner batch path と selfplay/eval single path で encoding schema を共有する。

CQ-0291 (batch 1): Riichi optional 用に synthetic action_type 値を追加。
engine の `Discard` (ActionType=0) は riichi flag によって 2 形態が存在する
ため、optional decision の candidate として扱うときは synthetic 値で区別する。
"""
from __future__ import annotations

import numpy as np
import torch

# CQ-0291: Riichi optional 用 synthetic action type ints
# engine の ActionType と被らないように 100+ を割り当てる
OPTIONAL_NORIICHI = 100  # NoRiichi (Discard with riichi=False, optional decision)
OPTIONAL_RIICHI = 101    # Riichi (Discard with riichi=True, optional decision)

# ActionType int → model embedding index
# CQ-0291 batch 2: TsumoWin (engine=1) / Ron (engine=2) を直接マップ
# CQ-0291 batch 3: Kakan (engine=6) / Ankan (engine=7) / Kyuushu (engine=9)
# (Discard と違い engine の int 値が一意に決まるため synthetic 不要)
ACTION_TYPE_MAP = {
    8: 0,  # Skip
    3: 1,  # Chi
    4: 2,  # Pon
    5: 3,  # Daiminkan
    # CQ-0291 batch 1: Riichi optional 候補
    OPTIONAL_NORIICHI: 4,  # NoRiichi
    OPTIONAL_RIICHI: 5,    # Riichi
    # CQ-0291 batch 2: 和了 optional 候補
    1: 6,  # TsumoWin
    2: 7,  # Ron
    # CQ-0291 batch 3: Kan/Kyuushu optional 候補
    6: 8,   # Kakan
    7: 9,   # Ankan
    9: 10,  # Kyuushu
}
# embedding 行数 (CandidateEncoder.action_type_emb の num_embeddings)
NUM_ACTION_TYPE_INDICES = 11
MAX_CONSUMED = 3
CAND_FEAT_DIM = 6  # [at_idx, tile_1based, seat_1based, c0, c1, c2]


def encode_single_candidate(action_type: int, tile_type: int,
                             target_rel_seat: int,
                             consumed_tile_ids: tuple[int, ...],
                             ) -> list[int]:
    """1 candidate → (CAND_FEAT_DIM,) int list (1-based encoding, 0=padding)"""
    at_idx = ACTION_TYPE_MAP.get(action_type, 0)
    tt = tile_type + 1 if tile_type >= 0 else 0
    rs = target_rel_seat + 1 if target_rel_seat >= 0 else 0
    consumed = list(consumed_tile_ids)[:MAX_CONSUMED]
    ct = [(t // 4) + 1 if t >= 0 else 0 for t in consumed]
    ct += [0] * (MAX_CONSUMED - len(ct))
    return [at_idx, tt, rs, ct[0], ct[1], ct[2]]


def encode_candidates_single(
    cand_records,
    max_cands: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """single sample の candidate list → (1, C, 6) int tensor + (1, C) mask

    dummy DecisionSample を作らずに直接 encoding する fast path。
    """
    n = len(cand_records)
    mc = max_cands or n
    feats = torch.zeros(1, mc, CAND_FEAT_DIM, dtype=torch.long)
    mask = torch.zeros(1, mc, dtype=torch.float32)
    for j, c in enumerate(cand_records):
        if j >= mc:
            break
        row = encode_single_candidate(
            c.action_type, c.tile_type, c.target_rel_seat,
            c.consumed_tile_ids)
        for k in range(CAND_FEAT_DIM):
            feats[0, j, k] = row[k]
        mask[0, j] = 1.0
    return feats, mask


def encode_candidates_batch(
    samples,
    max_cands: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """batch 用 candidate encoding (learner 互換)

    samples: list of objects with .candidates and .candidate_count
    Returns: (N, max_cands, 6) int, (N, max_cands) float
    """
    N = len(samples)
    feats = torch.zeros(N, max_cands, CAND_FEAT_DIM, dtype=torch.long)
    mask = torch.zeros(N, max_cands, dtype=torch.float32)
    for i, s in enumerate(samples):
        for j, c in enumerate(s.candidates):
            if j >= max_cands:
                break
            row = encode_single_candidate(
                c.action_type, c.tile_type, c.target_rel_seat,
                c.consumed_tile_ids)
            for k in range(CAND_FEAT_DIM):
                feats[i, j, k] = row[k]
            mask[i, j] = 1.0
    return feats, mask
