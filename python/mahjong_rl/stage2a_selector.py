"""CQ-0226: Stage2a action selection helper

Stage2a では discard と call で行動選択の責務が分かれる。

- discard: Stage1 と同じ 34-way categorical (model の discard_logits)
- call: candidate list 上の categorical (model の call_scores)

Stage2a v1 の selfplay は baseline actor を使うため、
model 推論は log_prob/value の記録用。
将来 policy actor に置き換える際は、このモジュールの
select_discard / select_optional を policy actor に接続する。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def select_discard_argmax(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[int, float]:
    """discard 用 argmax 選択

    Returns:
        (tile_type, log_prob)
    """
    masked = logits + (1.0 - legal_mask) * (-1e9)
    log_probs = F.log_softmax(masked, dim=-1)
    action = int(masked.argmax(dim=-1).item())
    return action, float(log_probs[action].item())


def select_optional_argmax(
    scores: torch.Tensor,
    cand_mask: torch.Tensor,
) -> tuple[int, float]:
    """call 用 argmax 選択

    Returns:
        (candidate_index, log_prob)
    """
    masked = scores + (1.0 - cand_mask) * (-1e9)
    log_probs = F.log_softmax(masked, dim=-1)
    idx = int(masked.argmax(dim=-1).item())
    return idx, float(log_probs[idx].item())


def select_discard_sample(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[int, float]:
    """discard 用サンプリング選択

    Returns:
        (tile_type, log_prob)
    """
    scaled = logits / max(temperature, 1e-8)
    masked = scaled + (1.0 - legal_mask) * (-1e9)
    probs = F.softmax(masked, dim=-1)
    action = int(torch.multinomial(probs, 1).item())
    log_probs = F.log_softmax(masked, dim=-1)
    return action, float(log_probs[action].item())


def select_optional_sample(
    scores: torch.Tensor,
    cand_mask: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[int, float]:
    """call 用サンプリング選択

    Returns:
        (candidate_index, log_prob)
    """
    scaled = scores / max(temperature, 1e-8)
    masked = scaled + (1.0 - cand_mask) * (-1e9)
    probs = F.softmax(masked, dim=-1)
    idx = int(torch.multinomial(probs, 1).item())
    log_probs = F.log_softmax(masked, dim=-1)
    return idx, float(log_probs[idx].item())
