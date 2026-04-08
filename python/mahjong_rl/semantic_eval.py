"""CQ-0258/0262: semantic auxiliary trunk の eval-only diagnostics

checkpoint + shard から terminal_head / yaku_head の予測品質を測定する。
学習ロジックには依存しない。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mahjong_rl.outcome_vocab import (
    TERMINAL_CLASSES, TERMINAL_CLASS_MAP, NUM_TERMINAL_CLASSES,
    YAKU_VOCAB, NUM_YAKU,
)


def evaluate_semantic_heads(
    model,
    shard_dir: Path,
    device: torch.device,
    batch_size: int = 256,
    label: str = "",
    checkpoint_path: str = "",
    encoder_metadata=None,
) -> dict:
    """semantic head の予測品質を測定する

    encoder_metadata: CQ-0269 deal_in subset 診断用 (feature_ranges が必要)
    """
    if not model._semantic_aux_enabled:
        raise ValueError(
            "semantic_aux が無効な model です。"
            " semantic_aux_config={'enabled': True} の checkpoint を使ってください。")

    from mahjong_rl.call_shard import DecisionShardReader

    reader = DecisionShardReader(shard_dir)
    data = reader.read_as_tensors()

    all_terminal_targets = []
    all_yaku_targets = []
    all_is_winner = []
    all_terminal_preds = []
    all_yaku_preds = []
    all_terminal_probs = []
    all_yaku_probs = []
    all_tenpai_flags = []
    all_remaining_draws = []

    # CQ-0269: extract feature ranges for subset diagnostics
    _has_subset_features = False
    _tenpai_range = _remaining_range = None
    if encoder_metadata is not None:
        fr = getattr(encoder_metadata, 'feature_ranges', {})
        if "self_tenpai_flag" in fr and "remaining_draws_norm" in fr:
            _tenpai_range = fr["self_tenpai_flag"]
            _remaining_range = fr["remaining_draws_norm"]
            _has_subset_features = True

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for branch_key, forward_fn in [("discard", _forward_discard),
                                         ("call", _forward_call)]:
            branch = data[branch_key]
            if branch is None or branch["n"] == 0:
                continue

            n = branch["n"]
            all_terminal_targets.append(branch["terminal_classes"])
            all_yaku_targets.append(branch["yaku_multihot"])
            all_is_winner.append(branch["is_winner"])

            # CQ-0269: extract subset features from observations
            if _has_subset_features:
                obs_arr = branch["observations"]
                s, e = _tenpai_range
                all_tenpai_flags.append(obs_arr[:, s:e].flatten())
                s, e = _remaining_range
                all_remaining_draws.append(obs_arr[:, s:e].flatten())

            t_preds, y_preds, t_probs, y_probs = [], [], [], []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                semantic = forward_fn(model, branch, start, end, device)
                t_logits = semantic["terminal_logits"]
                y_logits = semantic["yaku_logits"]
                t_preds.append(t_logits.argmax(dim=-1).cpu().numpy())
                y_preds.append((y_logits > 0).float().cpu().numpy())
                t_probs.append(torch.softmax(t_logits, dim=-1).cpu().numpy())
                y_probs.append(torch.sigmoid(y_logits).cpu().numpy())

            all_terminal_preds.append(np.concatenate(t_preds))
            all_yaku_preds.append(np.concatenate(y_preds))
            all_terminal_probs.append(np.concatenate(t_probs))
            all_yaku_probs.append(np.concatenate(y_probs))

    if not all_terminal_targets:
        return {"error": "shard にサンプルがありません"}

    terminal_targets = np.concatenate(all_terminal_targets)
    terminal_preds = np.concatenate(all_terminal_preds)
    terminal_probs = np.concatenate(all_terminal_probs)
    yaku_targets = np.concatenate(all_yaku_targets)
    yaku_preds = np.concatenate(all_yaku_preds)
    yaku_probs = np.concatenate(all_yaku_probs)
    is_winner = np.concatenate(all_is_winner)

    num_samples = len(terminal_targets)
    num_winner = int(is_winner.sum())

    terminal_result = _compute_terminal_metrics(terminal_targets, terminal_preds)
    terminal_result["label_conditioned_confidence"] = (
        _compute_terminal_label_conditioned_confidence(
            terminal_targets, terminal_probs))

    yaku_result = _compute_yaku_metrics(yaku_targets, yaku_preds, is_winner)
    yaku_result["label_conditioned_confidence"] = (
        _compute_yaku_label_conditioned_confidence(
            yaku_targets, yaku_probs, is_winner))

    # CQ-0269: deal_in risk diagnostics
    tenpai_flags = (np.concatenate(all_tenpai_flags)
                    if all_tenpai_flags else None)
    remaining_draws = (np.concatenate(all_remaining_draws)
                       if all_remaining_draws else None)
    deal_in_risk = _compute_deal_in_risk(
        terminal_targets, terminal_probs,
        tenpai_flags, remaining_draws)

    return {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "num_samples": num_samples,
        "num_winner_samples": num_winner,
        "terminal": terminal_result,
        "yaku": yaku_result,
        "deal_in_risk": deal_in_risk,
    }


def _forward_discard(model, branch, start, end, device):
    obs = torch.tensor(branch["observations"][start:end],
                       dtype=torch.float32, device=device)
    masks = torch.tensor(branch["legal_masks"][start:end],
                         dtype=torch.float32, device=device)
    out = model.forward_discard(obs, masks, compute_value=False)
    return out.semantic


def _forward_call(model, branch, start, end, device):
    obs = torch.tensor(branch["observations"][start:end],
                       dtype=torch.float32, device=device)
    cf = torch.tensor(branch["cand_feats"][start:end],
                      dtype=torch.long, device=device)
    cm = torch.tensor(branch["cand_mask"][start:end],
                      dtype=torch.float32, device=device)
    rc = torch.tensor(branch["response_context"][start:end],
                      dtype=torch.float32, device=device)
    out = model.forward_optional(obs, cf, cm, response_context=rc,
                                  compute_value=False)
    return out.semantic


def _compute_terminal_metrics(targets: np.ndarray, preds: np.ndarray) -> dict:
    """terminal_head の class-wise + overall metrics"""
    n = len(targets)
    accuracy = float((targets == preds).mean()) if n > 0 else 0.0

    class_metrics = []
    confusion = np.zeros((NUM_TERMINAL_CLASSES, NUM_TERMINAL_CLASSES), dtype=int)
    for i in range(n):
        confusion[targets[i], preds[i]] += 1

    for c in range(NUM_TERMINAL_CLASSES):
        tp = int(confusion[c, c])
        support = int(confusion[c].sum())
        pred_total = int(confusion[:, c].sum())
        precision = tp / pred_total if pred_total > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        class_metrics.append({
            "class": TERMINAL_CLASSES[c],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "support": support,
        })

    return {
        "accuracy": round(accuracy, 4),
        "support": n,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion.tolist(),
        "class_names": list(TERMINAL_CLASSES),
    }


def _compute_terminal_label_conditioned_confidence(
    targets: np.ndarray,
    probs: np.ndarray,
) -> list[dict]:
    """CQ-0262: terminal の label-conditioned confidence

    各 class A について actual=A の sample に限定して
    p(A) の分布と top-1 hit rate を集計する。
    """
    results = []
    for c in range(NUM_TERMINAL_CLASSES):
        mask = targets == c
        support = int(mask.sum())
        if support == 0:
            results.append({
                "class": TERMINAL_CLASSES[c],
                "support": 0,
                "mean_p": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0,
                "top1_hit_rate": 0.0, "top3_hit_rate": 0.0,
                "mean_rank": 0.0, "top1_confusers": [],
            })
            continue

        p_a = probs[mask, c]  # p(A) for samples where actual=A
        top1 = probs[mask].argmax(axis=1)
        ranks = (probs[mask].argsort(axis=1)[:, ::-1] == c).argmax(axis=1) + 1

        # top-1 confusers: what class won when A didn't
        wrong_mask = top1 != c
        if wrong_mask.sum() > 0:
            confuser_counts = Counter(int(x) for x in top1[wrong_mask])
            top_confusers = [
                {"class": TERMINAL_CLASSES[cls], "count": cnt}
                for cls, cnt in confuser_counts.most_common(3)
            ]
        else:
            top_confusers = []

        # top-3 hit
        top3_classes = np.argsort(probs[mask], axis=1)[:, -3:]
        top3_hit = float(np.any(top3_classes == c, axis=1).mean())

        results.append({
            "class": TERMINAL_CLASSES[c],
            "support": support,
            "mean_p": round(float(p_a.mean()), 4),
            "p50": round(float(np.percentile(p_a, 50)), 4),
            "p90": round(float(np.percentile(p_a, 90)), 4),
            "p99": round(float(np.percentile(p_a, 99)), 4),
            "top1_hit_rate": round(float((top1 == c).mean()), 4),
            "top3_hit_rate": round(top3_hit, 4),
            "mean_rank": round(float(ranks.mean()), 2),
            "top1_confusers": top_confusers,
        })
    return results


def _compute_yaku_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    is_winner: np.ndarray,
) -> dict:
    """yaku_head の multi-label metrics (winner-only)"""
    winner_mask = is_winner > 0.5
    num_winner = int(winner_mask.sum())

    if num_winner == 0:
        return {
            "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0,
            "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
            "exact_match_rate": 0.0, "num_winner": 0,
            "per_yaku_metrics": [], "yaku_names": [n for _, n in YAKU_VOCAB],
        }

    w_targets = targets[winner_mask]
    w_preds = preds[winner_mask]

    tp_total = float(((w_targets > 0.5) & (w_preds > 0.5)).sum())
    fp_total = float(((w_targets < 0.5) & (w_preds > 0.5)).sum())
    fn_total = float(((w_targets > 0.5) & (w_preds < 0.5)).sum())

    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0.0)

    per_yaku = []
    precisions, recalls = [], []
    for j in range(NUM_YAKU):
        t_col = w_targets[:, j]
        p_col = w_preds[:, j]
        tp = float(((t_col > 0.5) & (p_col > 0.5)).sum())
        fp = float(((t_col < 0.5) & (p_col > 0.5)).sum())
        fn = float(((t_col > 0.5) & (p_col < 0.5)).sum())
        support = int((t_col > 0.5).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_yaku.append({
            "yaku": YAKU_VOCAB[j][1],
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "support": support,
        })
        precisions.append(prec)
        recalls.append(rec)

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = (2 * macro_p * macro_r / (macro_p + macro_r)
                if (macro_p + macro_r) > 0 else 0.0)

    exact = float(((w_targets > 0.5) == (w_preds > 0.5)).all(axis=1).mean())

    return {
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "exact_match_rate": round(exact, 4),
        "num_winner": num_winner,
        "per_yaku_metrics": per_yaku,
        "yaku_names": [n for _, n in YAKU_VOCAB],
    }


def _compute_yaku_label_conditioned_confidence(
    targets: np.ndarray,
    probs: np.ndarray,
    is_winner: np.ndarray,
) -> list[dict]:
    """CQ-0262: yaku の positive-conditioned confidence

    各 yaku A について actual positive (A in winner sample) に限定して
    sigmoid p(A) の分布と threshold hit rate を集計する。
    """
    winner_mask = is_winner > 0.5
    num_winner = int(winner_mask.sum())

    results = []
    for j in range(NUM_YAKU):
        # actual positive: winner かつ target[j] == 1
        if num_winner == 0:
            results.append(_empty_yaku_confidence(j))
            continue

        w_targets_j = targets[winner_mask, j]
        w_probs_j = probs[winner_mask, j]
        w_probs_all = probs[winner_mask]  # (n_winner, NUM_YAKU)

        pos_mask = w_targets_j > 0.5
        support = int(pos_mask.sum())
        if support == 0:
            results.append(_empty_yaku_confidence(j))
            continue

        p_a = w_probs_j[pos_mask]

        # ranks within each sample (1-based, lower=higher confidence)
        ranks_all = np.argsort(-w_probs_all[pos_mask], axis=1)
        ranks = np.zeros(support, dtype=int)
        for i in range(support):
            ranks[i] = int(np.where(ranks_all[i] == j)[0][0]) + 1

        top3 = ranks <= 3

        results.append({
            "yaku": YAKU_VOCAB[j][1],
            "support": support,
            "mean_p": round(float(p_a.mean()), 4),
            "p50": round(float(np.percentile(p_a, 50)), 4),
            "p90": round(float(np.percentile(p_a, 90)), 4),
            "p99": round(float(np.percentile(p_a, 99)), 4),
            "threshold_hit_rate_0p5": round(float((p_a >= 0.5).mean()), 4),
            "threshold_hit_rate_0p2": round(float((p_a >= 0.2).mean()), 4),
            "top3_hit_rate": round(float(top3.mean()), 4),
            "mean_rank": round(float(ranks.mean()), 2),
        })
    return results


def _empty_yaku_confidence(j: int) -> dict:
    return {
        "yaku": YAKU_VOCAB[j][1],
        "support": 0,
        "mean_p": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0,
        "threshold_hit_rate_0p5": 0.0, "threshold_hit_rate_0p2": 0.0,
        "top3_hit_rate": 0.0, "mean_rank": 0.0,
    }


def _compute_deal_in_risk(
    terminal_targets: np.ndarray,
    terminal_probs: np.ndarray,
    tenpai_flags: np.ndarray | None,
    remaining_draws: np.ndarray | None,
) -> dict:
    """CQ-0269: deal_in risk diagnostics"""
    deal_in_idx = TERMINAL_CLASS_MAP.get("deal_in")
    if deal_in_idx is None:
        return {"error": "deal_in class not found"}

    p_dealin = terminal_probs[:, deal_in_idx]
    is_pos = terminal_targets == deal_in_idx

    overall = _deal_in_binary_summary(p_dealin, is_pos, "overall")

    # subset diagnostics
    subsets = {}
    if tenpai_flags is not None and remaining_draws is not None:
        # late_and_noten: tenpai=0, remaining_draws < 0.3
        late_noten = (tenpai_flags < 0.5) & (remaining_draws < 0.3)
        if late_noten.sum() > 0:
            subsets["late_and_noten"] = _deal_in_binary_summary(
                p_dealin[late_noten], is_pos[late_noten], "late_and_noten")
        else:
            subsets["late_and_noten"] = _empty_subset("late_and_noten")

        # early_and_tenpai: tenpai=1, remaining_draws > 0.7
        early_tenpai = (tenpai_flags > 0.5) & (remaining_draws > 0.7)
        if early_tenpai.sum() > 0:
            subsets["early_and_tenpai"] = _deal_in_binary_summary(
                p_dealin[early_tenpai], is_pos[early_tenpai],
                "early_and_tenpai")
        else:
            subsets["early_and_tenpai"] = _empty_subset("early_and_tenpai")

    return {"overall": overall, "subsets": subsets}


def _deal_in_binary_summary(
    p_dealin: np.ndarray, is_pos: np.ndarray, name: str,
) -> dict:
    """deal_in positive/negative の分離統計"""
    n = len(p_dealin)
    n_pos = int(is_pos.sum())
    n_neg = n - n_pos
    p_pos = p_dealin[is_pos] if n_pos > 0 else np.array([])
    p_neg = p_dealin[~is_pos] if n_neg > 0 else np.array([])

    result = {
        "name": name,
        "support_pos": n_pos,
        "support_neg": n_neg,
        "mean_p_pos": round(float(p_pos.mean()), 4) if n_pos > 0 else None,
        "mean_p_neg": round(float(p_neg.mean()), 4) if n_neg > 0 else None,
        "p50_pos": round(float(np.percentile(p_pos, 50)), 4) if n_pos > 0 else None,
        "p50_neg": round(float(np.percentile(p_neg, 50)), 4) if n_neg > 0 else None,
        "p90_pos": round(float(np.percentile(p_pos, 90)), 4) if n_pos > 0 else None,
        "p90_neg": round(float(np.percentile(p_neg, 90)), 4) if n_neg > 0 else None,
    }

    # AUC
    if n_pos > 0 and n_neg > 0:
        result["roc_auc"] = round(_roc_auc(is_pos, p_dealin), 4)
        result["pr_auc"] = round(_pr_auc(is_pos, p_dealin), 4)
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None

    return result


def _empty_subset(name: str) -> dict:
    return {
        "name": name, "support_pos": 0, "support_neg": 0,
        "mean_p_pos": None, "mean_p_neg": None,
        "p50_pos": None, "p50_neg": None,
        "p90_pos": None, "p90_neg": None,
        "roc_auc": None, "pr_auc": None,
    }


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """simple ROC AUC (no sklearn dependency)"""
    pos = y_score[y_true]
    neg = y_score[~y_true]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    # Mann-Whitney U
    total = 0.0
    for p in pos:
        total += (neg < p).sum() + 0.5 * (neg == p).sum()
    return float(total / (n_pos * n_neg))


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """simple PR AUC (trapezoidal)"""
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    # sort by score descending
    order = np.argsort(-y_score)
    sorted_true = y_true[order].astype(float)
    tp_cum = np.cumsum(sorted_true)
    fp_cum = np.cumsum(1 - sorted_true)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos
    # trapezoidal
    auc = 0.0
    prev_r = 0.0
    for i in range(len(recall)):
        dr = recall[i] - prev_r
        auc += precision[i] * dr
        prev_r = recall[i]
    return float(auc)


def format_summary(result: dict) -> str:
    """human-readable summary"""
    lines = []
    lines.append(f"# Semantic Head Eval: {result.get('label', '')}")
    lines.append(f"- checkpoint: {result.get('checkpoint', '')}")
    lines.append(f"- samples: {result['num_samples']} "
                 f"(winner: {result['num_winner_samples']})")
    lines.append("")

    t = result["terminal"]
    lines.append(f"## terminal_head (accuracy={t['accuracy']:.4f})")
    lines.append("| class | precision | recall | support |")
    lines.append("|---|---|---|---|")
    for cm in t["class_metrics"]:
        lines.append(f"| {cm['class']} | {cm['precision']:.4f} "
                     f"| {cm['recall']:.4f} | {cm['support']} |")
    lines.append("")

    # CQ-0262: label-conditioned confidence
    if "label_conditioned_confidence" in t:
        lines.append("### terminal confidence (actual=A → p(A))")
        lines.append("| class | support | mean_p | p50 | p90 | p99 "
                     "| top1_hit | top3_hit | mean_rank |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for lc in t["label_conditioned_confidence"]:
            if lc["support"] == 0:
                continue
            lines.append(
                f"| {lc['class']} | {lc['support']} "
                f"| {lc['mean_p']:.4f} | {lc['p50']:.4f} "
                f"| {lc['p90']:.4f} | {lc['p99']:.4f} "
                f"| {lc['top1_hit_rate']:.4f} | {lc['top3_hit_rate']:.4f} "
                f"| {lc['mean_rank']:.1f} |")
        # confusers for key classes
        for lc in t["label_conditioned_confidence"]:
            if lc["support"] > 0 and lc["top1_confusers"]:
                conf_str = ", ".join(
                    f"{c['class']}({c['count']})" for c in lc["top1_confusers"])
                lines.append(f"- {lc['class']} confusers: {conf_str}")
        lines.append("")

    y = result["yaku"]
    lines.append(f"## yaku_head (winner-only, n={y['num_winner']})")
    lines.append(f"- micro P/R/F1: {y['micro_precision']:.4f} / "
                 f"{y['micro_recall']:.4f} / {y['micro_f1']:.4f}")
    lines.append(f"- macro P/R/F1: {y['macro_precision']:.4f} / "
                 f"{y['macro_recall']:.4f} / {y['macro_f1']:.4f}")
    lines.append(f"- exact match: {y['exact_match_rate']:.4f}")
    lines.append("")
    lines.append("| yaku | precision | recall | support |")
    lines.append("|---|---|---|---|")
    for ym in y["per_yaku_metrics"]:
        lines.append(f"| {ym['yaku']} | {ym['precision']:.4f} "
                     f"| {ym['recall']:.4f} | {ym['support']} |")

    # CQ-0262: yaku confidence
    if "label_conditioned_confidence" in y:
        lines.append("")
        lines.append("### yaku confidence (actual positive → p(A))")
        lines.append("| yaku | support | mean_p | p50 | p90 | p99 "
                     "| hit@0.5 | hit@0.2 | top3 | rank |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for yc in y["label_conditioned_confidence"]:
            if yc["support"] == 0:
                continue
            lines.append(
                f"| {yc['yaku']} | {yc['support']} "
                f"| {yc['mean_p']:.4f} | {yc['p50']:.4f} "
                f"| {yc['p90']:.4f} | {yc['p99']:.4f} "
                f"| {yc['threshold_hit_rate_0p5']:.4f} "
                f"| {yc['threshold_hit_rate_0p2']:.4f} "
                f"| {yc['top3_hit_rate']:.4f} | {yc['mean_rank']:.1f} |")

    # CQ-0269: deal_in risk
    dr = result.get("deal_in_risk", {})
    if dr:
        lines.append("")
        lines.append("## deal_in risk diagnostics")
        ov = dr.get("overall", {})
        if ov:
            _fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"
            lines.append(
                f"- overall: pos={ov.get('support_pos',0)} "
                f"neg={ov.get('support_neg',0)}")
            lines.append(
                f"  mean_p: pos={_fmt(ov.get('mean_p_pos'))} "
                f"neg={_fmt(ov.get('mean_p_neg'))}")
            lines.append(
                f"  roc_auc={_fmt(ov.get('roc_auc'))} "
                f"pr_auc={_fmt(ov.get('pr_auc'))}")
        for name, sub in dr.get("subsets", {}).items():
            lines.append(
                f"- {name}: pos={sub.get('support_pos',0)} "
                f"neg={sub.get('support_neg',0)} "
                f"pr_auc={sub.get('pr_auc', 'N/A')}")

    return "\n".join(lines)
