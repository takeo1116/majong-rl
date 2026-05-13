"""CQ-0295: optional decision family ごとの offline audit.

既存 run の checkpoint + selfplay shard を入力にして、family 別の
teacher label 分布 / policy agreement / entropy / max_prob / teacher
action probability を JSON に出力する。

学習挙動は変更しない (pure offline)。

対象 family:
- ``discard``: Stage1 互換 34-way discard
- ``response``: Stage2a 既存 Chi/Pon/Daiminkan/Skip
- ``riichi``: NoRiichi/Riichi (CQ-0291 batch 1)
- ``tsumo``: TsumoWin/Skip (CQ-0291 batch 2)
- ``ron``: Ron/Skip
- ``ankan`` / ``kakan`` / ``kyuushu``: action/Skip (CQ-0291 batch 3)
- ``unknown``: decision_family が無い / 未知の値の sample

旧 shard で ``decision_family`` が "response" 既定になっている場合は
そのまま "response" として扱う (legacy 互換)。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from mahjong_rl.call_shard import DecisionSample, DecisionShardReader
from mahjong_rl.candidate_encoding import (
    NUM_ACTION_TYPE_INDICES,
    encode_candidates_batch,
)


# CQ-0295: 集計対象 family (offline audit が出すキー)
SUPPORTED_FAMILIES: tuple[str, ...] = (
    "discard", "response", "riichi", "tsumo", "ron",
    "ankan", "kakan", "kyuushu", "unknown",
)

# binary optional family の positive index (teacher_top1_index で
# positive action とみなす index)
# riichi: [NoRiichi=0, Riichi=1]
# tsumo/ron/ankan/kakan/kyuushu: [action=0, Skip=1]
BINARY_POSITIVE_INDEX: dict[str, int] = {
    "riichi": 1,
    "tsumo": 0,
    "ron": 0,
    "ankan": 0,
    "kakan": 0,
    "kyuushu": 0,
}


def normalize_family(value: str | None) -> str:
    """shard 由来の decision_family 文字列を SUPPORTED_FAMILIES に正規化。

    None / 空文字 / 未知は "unknown" にする。
    """
    if value is None:
        return "unknown"
    v = str(value).strip().lower()
    if v in SUPPORTED_FAMILIES:
        return v
    return "unknown"


def _percentile(values: list[float] | np.ndarray, q: float) -> float | None:
    if values is None or len(values) == 0:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def _basic_stats(values: list[float] | np.ndarray,
                  qs: tuple[int, ...] = (50, 90)) -> dict:
    """count / mean / pXX のみ返す軽量 helper。"""
    if values is None or len(values) == 0:
        return {"count": 0, "mean": None,
                **{f"p{q}": None for q in qs}}
    arr = np.asarray(values, dtype=np.float64)
    out: dict = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
    }
    for q in qs:
        out[f"p{q}"] = float(np.percentile(arr, q))
    return out


def _counts_to_top(counter: Counter, top_n: int) -> list[dict]:
    """Counter を上位 N の (value, count, rate) リストに変換。"""
    total = sum(counter.values())
    if total == 0:
        return []
    items = counter.most_common(top_n)
    out: list[dict] = []
    for v, c in items:
        out.append({
            "index": int(v),
            "count": int(c),
            "rate": float(c / total),
        })
    return out


def _to_py(obj):
    """numpy scalar / tensor を JSON serializable な python 型に変換。"""
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_py(x) for x in obj.tolist()]
    if isinstance(obj, torch.Tensor):
        return _to_py(obj.detach().cpu().numpy())
    return obj


def _compute_discard_policy_stats(
    model, encoder_dim: int, samples: list[DecisionSample],
    device: torch.device, batch_size: int,
) -> dict:
    """discard family の policy forward 集計。

    各 sample に対して ``model.forward_discard`` を呼び、policy top1,
    teacher_top1 agreement, entropy, max_prob, teacher action prob を
    集計する。
    """
    if not samples:
        return {"policy_evaluated": 0}

    teacher_top1_arr = np.array(
        [s.teacher_top1_index for s in samples], dtype=np.int64)
    has_teacher = teacher_top1_arr >= 0
    policy_top1 = np.full(len(samples), -1, dtype=np.int64)
    entropies: list[float] = []
    max_probs: list[float] = []
    teacher_action_probs: list[float] = []
    best_set_agree_count = 0
    best_set_has = 0

    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            end = min(start + batch_size, len(samples))
            batch = samples[start:end]
            obs = torch.tensor(
                np.stack([s.observation for s in batch]),
                dtype=torch.float32, device=device)
            mask_list = []
            for s in batch:
                if s.legal_mask is None:
                    mask_list.append(np.zeros(34, dtype=np.float32))
                else:
                    mask_list.append(s.legal_mask.astype(np.float32))
            masks = torch.tensor(
                np.stack(mask_list), dtype=torch.float32, device=device)
            out = model.forward_discard(obs, masks, compute_value=False)
            logits = out.discard_logits
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            top1 = probs.argmax(dim=-1)
            max_p = probs.max(dim=-1).values
            # entropy = -sum(p log p), mask 無効 idx は 0 扱い
            ent = -(probs * log_probs).sum(dim=-1)
            policy_top1[start:end] = top1.cpu().numpy()
            for i, s in enumerate(batch):
                global_i = start + i
                entropies.append(float(ent[i].item()))
                max_probs.append(float(max_p[i].item()))
                if has_teacher[global_i]:
                    t1 = int(teacher_top1_arr[global_i])
                    teacher_action_probs.append(float(probs[i, t1].item()))
                if s.teacher_best_indices:
                    best_set_has += 1
                    if int(top1[i].item()) in s.teacher_best_indices:
                        best_set_agree_count += 1

    teacher_eval_mask = has_teacher
    top1_match = int(((policy_top1 == teacher_top1_arr)
                      & teacher_eval_mask).sum())
    top1_total = int(teacher_eval_mask.sum())

    result: dict = {
        "policy_evaluated": int(len(samples)),
        "teacher_top1_agreement_count": top1_match,
        "teacher_top1_agreement_total": top1_total,
        "teacher_top1_agreement_rate": (
            top1_match / top1_total if top1_total > 0 else None),
        "entropy": _basic_stats(entropies, qs=(50, 90)),
        "max_prob": _basic_stats(max_probs, qs=(50, 90, 99)),
        "teacher_action_prob": _basic_stats(
            teacher_action_probs, qs=(50, 90)),
        "policy_top1_distribution": _counts_to_top(
            Counter(int(x) for x in policy_top1 if x >= 0), 10),
    }
    if best_set_has > 0:
        result["best_set_agreement_count"] = int(best_set_agree_count)
        result["best_set_agreement_total"] = int(best_set_has)
        result["best_set_agreement_rate"] = float(
            best_set_agree_count / best_set_has)
    return result


def _compute_optional_policy_stats(
    model, samples: list[DecisionSample],
    device: torch.device, batch_size: int,
) -> dict:
    """optional / response family の policy forward 集計。

    candidates を encode し ``model.forward_optional`` を呼ぶ。candidate
    数は family ごとに異なるため batch ごとに max_cands を決める。
    """
    if not samples:
        return {"policy_evaluated": 0}

    teacher_top1_arr = np.array(
        [s.teacher_top1_index for s in samples], dtype=np.int64)
    has_teacher = teacher_top1_arr >= 0
    policy_top1 = np.full(len(samples), -1, dtype=np.int64)
    entropies: list[float] = []
    max_probs: list[float] = []
    teacher_action_probs: list[float] = []

    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            end = min(start + batch_size, len(samples))
            batch = samples[start:end]
            obs = torch.tensor(
                np.stack([s.observation for s in batch]),
                dtype=torch.float32, device=device)
            max_cands = max(max(1, s.candidate_count) for s in batch)
            cand_feats, cand_mask = encode_candidates_batch(
                batch, max_cands)
            cand_feats = cand_feats.to(device)
            cand_mask = cand_mask.to(device)
            # response_context
            rc_list: list[np.ndarray] = []
            for s in batch:
                if s.response_context is not None:
                    rc_list.append(
                        np.asarray(s.response_context, dtype=np.float32))
                else:
                    rc_list.append(np.zeros(3, dtype=np.float32))
            rc = torch.tensor(
                np.stack(rc_list), dtype=torch.float32, device=device)
            out = model.forward_optional(
                obs, cand_feats, cand_mask,
                response_context=rc, compute_value=False)
            scores = out.optional_scores
            # mask 無効 candidate は -1e9 (forward_optional 内で適用済み)
            log_probs = F.log_softmax(scores, dim=-1)
            probs = log_probs.exp()
            top1 = probs.argmax(dim=-1)
            max_p = probs.max(dim=-1).values
            # entropy: mask 内 (有効 idx) のみで計算
            safe_lp = torch.where(
                cand_mask > 0.5, log_probs, torch.zeros_like(log_probs))
            ent = -(probs * safe_lp).sum(dim=-1)
            policy_top1[start:end] = top1.cpu().numpy()
            for i, s in enumerate(batch):
                global_i = start + i
                entropies.append(float(ent[i].item()))
                max_probs.append(float(max_p[i].item()))
                if has_teacher[global_i]:
                    t1 = int(teacher_top1_arr[global_i])
                    if 0 <= t1 < probs.size(1):
                        teacher_action_probs.append(
                            float(probs[i, t1].item()))

    teacher_eval_mask = has_teacher
    top1_match = int(((policy_top1 == teacher_top1_arr)
                      & teacher_eval_mask).sum())
    top1_total = int(teacher_eval_mask.sum())

    result: dict = {
        "policy_evaluated": int(len(samples)),
        "teacher_top1_agreement_count": top1_match,
        "teacher_top1_agreement_total": top1_total,
        "teacher_top1_agreement_rate": (
            top1_match / top1_total if top1_total > 0 else None),
        "entropy": _basic_stats(entropies, qs=(50, 90)),
        "max_prob": _basic_stats(max_probs, qs=(50, 90, 99)),
        "teacher_action_prob": _basic_stats(
            teacher_action_probs, qs=(50, 90)),
        "policy_top1_distribution": _counts_to_top(
            Counter(int(x) for x in policy_top1 if x >= 0), 10),
    }
    return result


def _summarize_family_samples(
    family: str, samples: list[DecisionSample], top_n: int = 5,
) -> dict:
    """forward なしで集計できる sample-level stats."""
    n = len(samples)
    out: dict = {
        "sample_count": int(n),
    }
    if n == 0:
        return out

    cand_counts = [s.candidate_count for s in samples
                    if s.candidate_count >= 0]
    if cand_counts:
        out["candidate_count"] = {
            "mean": float(np.mean(cand_counts)),
            "p50": _percentile(cand_counts, 50),
            "p90": _percentile(cand_counts, 90),
            "max": int(max(cand_counts)),
        }

    # teacher_top1 distribution
    teacher_top1 = [s.teacher_top1_index for s in samples
                    if s.teacher_top1_index >= 0]
    out["teacher_top1_present"] = int(len(teacher_top1))
    out["teacher_top1_distribution"] = _counts_to_top(
        Counter(teacher_top1), top_n)

    # binary positive rate
    if family in BINARY_POSITIVE_INDEX and teacher_top1:
        pos_idx = BINARY_POSITIVE_INDEX[family]
        positive = sum(1 for t in teacher_top1 if t == pos_idx)
        out["binary_positive"] = {
            "positive_index": int(pos_idx),
            "positive_count": int(positive),
            "positive_total": int(len(teacher_top1)),
            "positive_rate": float(positive / len(teacher_top1))
            if teacher_top1 else None,
        }

    # best-set distribution
    best_sizes = [len(s.teacher_best_indices) for s in samples
                  if s.teacher_best_indices]
    if best_sizes:
        out["teacher_best_set_size"] = {
            "count": int(len(best_sizes)),
            "mean": float(np.mean(best_sizes)),
            "p50": _percentile(best_sizes, 50),
            "p90": _percentile(best_sizes, 90),
            "max": int(max(best_sizes)),
        }
        # top1 ∈ best-set 率
        top1_in_best = sum(
            1 for s in samples
            if s.teacher_top1_index >= 0
            and s.teacher_best_indices
            and s.teacher_top1_index in s.teacher_best_indices)
        out["teacher_top1_in_best_set"] = {
            "count": int(top1_in_best),
            "total": int(len(best_sizes)),
            "rate": float(top1_in_best / len(best_sizes))
            if best_sizes else None,
        }

    return out


def audit_samples(
    samples: list[DecisionSample],
    model=None,
    device: torch.device | None = None,
    batch_size: int = 256,
    top_n: int = 5,
) -> dict:
    """sample リストを family 別に集計する core 関数。

    ``model`` が None の場合は teacher 分布 / candidate stats のみを
    返す。``model`` がある場合は policy forward を実行して agreement /
    entropy / max_prob / teacher_action_prob を追加する。
    """
    by_family: dict[str, list[DecisionSample]] = {
        f: [] for f in SUPPORTED_FAMILIES}
    for s in samples:
        fam = normalize_family(getattr(s, "decision_family", None))
        # decision_type が "discard" のものは family を "discard" に強制
        # (legacy shard で decision_family が "response" 既定の場合に対応)
        if s.decision_type == "discard":
            fam = "discard"
        by_family.setdefault(fam, []).append(s)

    if device is None:
        device = torch.device("cpu")

    families_out: dict[str, dict] = {}
    total = 0
    for fam, fs in by_family.items():
        info = _summarize_family_samples(fam, fs, top_n=top_n)
        total += info["sample_count"]
        if model is not None and fs:
            if fam == "discard":
                info["policy"] = _compute_discard_policy_stats(
                    model, encoder_dim=None, samples=fs,
                    device=device, batch_size=batch_size)
            else:
                info["policy"] = _compute_optional_policy_stats(
                    model, samples=fs, device=device,
                    batch_size=batch_size)
        families_out[fam] = info

    return _to_py({
        "total_samples": int(total),
        "families": families_out,
        "supported_families": list(SUPPORTED_FAMILIES),
    })


def format_summary(result: dict, label: str = "") -> str:
    """JSON 結果を Markdown summary に整形する。"""
    lines: list[str] = []
    suffix = f" ({label})" if label else ""
    lines.append(f"# Optional Family Audit Summary{suffix}")
    lines.append("")
    lines.append(f"- total_samples: {result.get('total_samples', 0)}")
    lines.append("")
    fams = result.get("families", {})
    for fam in SUPPORTED_FAMILIES:
        info = fams.get(fam)
        if not info or info.get("sample_count", 0) == 0:
            continue
        lines.append(f"## family: {fam}")
        lines.append(f"- sample_count: {info['sample_count']}")
        cc = info.get("candidate_count")
        if cc:
            lines.append(
                f"- candidate_count: "
                f"mean={cc.get('mean')}, p50={cc.get('p50')}, "
                f"p90={cc.get('p90')}, max={cc.get('max')}")
        if "binary_positive" in info:
            bp = info["binary_positive"]
            lines.append(
                f"- binary positive (idx={bp['positive_index']}): "
                f"{bp['positive_count']} / {bp['positive_total']} "
                f"(rate={bp['positive_rate']:.4f})")
        td = info.get("teacher_top1_distribution")
        if td:
            top = ", ".join(
                f"{e['index']}:{e['count']}({e['rate']:.3f})"
                for e in td[:5])
            lines.append(f"- teacher_top1 top: {top}")
        pol = info.get("policy")
        if pol:
            lines.append(
                f"- policy_top1 agreement: "
                f"{pol.get('teacher_top1_agreement_count')}/"
                f"{pol.get('teacher_top1_agreement_total')} "
                f"(rate={pol.get('teacher_top1_agreement_rate')})")
            ent = pol.get("entropy") or {}
            lines.append(
                f"- entropy mean={ent.get('mean')}, p50={ent.get('p50')}, "
                f"p90={ent.get('p90')}")
            mp = pol.get("max_prob") or {}
            lines.append(
                f"- max_prob mean={mp.get('mean')}, p50={mp.get('p50')}, "
                f"p90={mp.get('p90')}, p99={mp.get('p99')}")
            tap = pol.get("teacher_action_prob") or {}
            lines.append(
                f"- teacher_action_prob mean={tap.get('mean')}, "
                f"p50={tap.get('p50')}, p90={tap.get('p90')}")
            if "best_set_agreement_rate" in pol:
                lines.append(
                    f"- best_set agreement: "
                    f"{pol.get('best_set_agreement_count')}/"
                    f"{pol.get('best_set_agreement_total')} "
                    f"(rate={pol.get('best_set_agreement_rate'):.4f})")
        lines.append("")
    return "\n".join(lines)


def audit_shard_dir(
    shard_dir: Path,
    model=None,
    device: torch.device | None = None,
    batch_size: int = 256,
    max_samples: int | None = None,
) -> dict:
    """shard dir 全体を読み込んで audit する high-level API。"""
    reader = DecisionShardReader(Path(shard_dir))
    samples = reader.read_all()
    if max_samples is not None and len(samples) > max_samples:
        samples = samples[:int(max_samples)]
    return audit_samples(
        samples, model=model, device=device, batch_size=batch_size)
