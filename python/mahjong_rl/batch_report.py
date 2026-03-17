"""マルチ seed バッチ実行の集約レポート生成 (CQ-0078)"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def generate_batch_report(batch_dir: Path, results: list[dict]) -> None:
    """バッチ実行結果の集約レポートを生成する

    batch_dir に batch_summary.json と batch_table.csv を書き出す。

    Args:
        batch_dir: バッチ実行の親ディレクトリ
        results: seed ごとの実行結果リスト
            各要素: {"seed": int, "success": bool, "result": dict | None, "error": str | None}
    """
    seeds = [r["seed"] for r in results]
    success_runs = [r for r in results if r["success"]]
    failure_runs = [r for r in results if not r["success"]]

    # 成功 run から eval_metrics を収集（run とペアで保持: CQ-0099）
    eval_metrics_list: list[dict] = []
    eval_metrics_runs: list[dict] = []
    for r in success_runs:
        result = r.get("result", {})
        em = result.get("eval_metrics")
        if em is not None:
            eval_metrics_list.append(em)
            eval_metrics_runs.append(r)

    # 集約統計
    aggregate = _compute_aggregate(eval_metrics_list)

    # eval_mode を集約に付加 (CQ-0092)
    if eval_metrics_list:
        modes = set(em.get("eval_mode", "single") for em in eval_metrics_list)
        aggregate["eval_mode"] = modes.pop() if len(modes) == 1 else "mixed"

    # outlier 情報を付加 (CQ-0094, CQ-0099)
    _attach_outlier_info(aggregate, eval_metrics_list, eval_metrics_runs)

    # runs 一覧
    runs_info = []
    for r in results:
        entry: dict = {
            "seed": r["seed"],
            "success": r["success"],
        }
        if r["success"]:
            result = r.get("result", {})
            entry["run_dir"] = result.get("run_dir", "")
            em = result.get("eval_metrics")
            if em is not None:
                entry["eval_metrics"] = {
                    "avg_rank": em.get("avg_rank"),
                    "avg_score": em.get("avg_score"),
                    "win_rate": em.get("win_rate"),
                    "deal_in_rate": em.get("deal_in_rate"),
                }
                entry["eval_mode"] = em.get("eval_mode", "single")
            # eval_before/after 差分
            ed = result.get("eval_diff")
            if ed is not None:
                entry["eval_diff"] = ed
            entry["global_seed"] = result.get("global_seed")
            # worker 設定 (CQ-0081)
            sp_stats = result.get("selfplay_stats", {})
            eval_m = result.get("eval_metrics", {})
            entry["worker_settings"] = {
                "selfplay_num_workers": sp_stats.get("num_workers", 1),
                "evaluation_num_workers": eval_m.get("num_workers", 1),
            }
            # device_info / env_info (CQ-0081) + imitation_metrics (CQ-0127)
            run_dir_path = Path(result["run_dir"]) if result.get("run_dir") else None
            if run_dir_path is not None:
                summary_path = run_dir_path / "summary.json"
                if summary_path.exists():
                    with open(summary_path) as sf:
                        run_summary = json.load(sf)
                    di = run_summary.get("device_info")
                    if di is not None:
                        entry["device_info"] = di
                    ei = run_summary.get("env_info")
                    if ei is not None:
                        entry["env_info"] = ei
                    # imitation 教師再現メトリクス (CQ-0127)
                    imi_stats = run_summary.get("phase_stats", {}).get("imitation", {})
                    imi_top1 = imi_stats.get("teacher_top1_match_rate")
                    imi_best_set = imi_stats.get("teacher_best_set_hit_rate")
                    if imi_top1 is not None or imi_best_set is not None:
                        imi_entry: dict = {
                            "teacher_top1_match_rate": imi_top1,
                            "teacher_best_set_hit_rate": imi_best_set,
                            "imitation_loss_mode": imi_stats.get("imitation_loss_mode"),
                            # CQ-0150, CQ-0152: joint imitation 追跡
                            "value_loss": imi_stats.get("value_loss"),
                            "imitation_value_warmstart": imi_stats.get("imitation_value_warmstart"),
                        }
                        # CQ-0206: multi-chunk imitation
                        mci_info = imi_stats.get("multi_chunk_imitation")
                        if mci_info is not None:
                            imi_entry["multi_chunk_imitation"] = mci_info
                        chunks = imi_stats.get("chunks")
                        if chunks is not None:
                            imi_entry["chunks"] = chunks
                        entry["imitation_metrics"] = imi_entry
                    # CQ-0151, CQ-0152: model_features
                    mf = run_summary.get("model_features")
                    if mf is not None:
                        entry["model_features"] = mf
                    # CQ-0171: encoder_features
                    ef = run_summary.get("encoder_features")
                    if ef is not None:
                        entry["encoder_features"] = ef
                    # CQ-0137: learner PPO 診断統計
                    learner_stats = run_summary.get("phase_stats", {}).get("learner", {})
                    ppo_diag = learner_stats.get("ppo_diag")
                    if ppo_diag is not None:
                        entry["learner_diag"] = ppo_diag
                    # CQ-0166: learner 補助統計
                    pre = learner_stats.get("post_riichi_exclusion")
                    if pre is not None:
                        entry["post_riichi_exclusion"] = pre
                    # CQ-0139: reward composition
                    sp_stats = run_summary.get("phase_stats", {}).get("selfplay", {})
                    rc = sp_stats.get("reward_composition")
                    if rc is not None:
                        entry["reward_composition"] = rc
                    # CQ-0143: reward shaping 設定
                    rs = sp_stats.get("reward_shaping")
                    if rs is not None:
                        entry["reward_shaping"] = rs
                    # CQ-0174: eval_before を転送
                    eb_stats = run_summary.get("phase_stats", {}).get("eval_before")
                    if eb_stats is not None:
                        entry["eval_before"] = eb_stats
                    # CQ-0174: phase_timing を転送
                    pt = run_summary.get("phase_timing")
                    if pt is not None:
                        entry["phase_timing"] = pt
                    # CQ-0180: cycles を転送
                    cycles = run_summary.get("phase_stats", {}).get("cycles")
                    if cycles is not None:
                        entry["cycles"] = cycles
        else:
            entry["error"] = r.get("error", "unknown")
        runs_info.append(entry)

    # imitation 教師再現メトリクス集約 (CQ-0127)
    imi_metrics_list = [
        entry["imitation_metrics"]
        for entry in runs_info
        if entry.get("imitation_metrics")
    ]
    if imi_metrics_list:
        aggregate["imitation"] = _compute_aggregate_generic(
            imi_metrics_list,
            ["teacher_top1_match_rate", "teacher_best_set_hit_rate", "value_loss"],
        )
        # CQ-0133, CQ-0134: mode 別集約（None/空文字は "unknown" に正規化）
        by_mode: dict[str, list[dict]] = {}
        for m in imi_metrics_list:
            raw_mode = m.get("imitation_loss_mode")
            mode = raw_mode if isinstance(raw_mode, str) and raw_mode else "unknown"
            by_mode.setdefault(mode, []).append(m)
        if by_mode:
            aggregate["imitation_by_loss_mode"] = {
                mode: _compute_aggregate_generic(
                    entries,
                    ["teacher_top1_match_rate", "teacher_best_set_hit_rate"],
                )
                for mode, entries in sorted(by_mode.items())
            }

    # CQ-0137: learner 診断統計の集約
    learner_diag_list = [
        entry["learner_diag"]
        for entry in runs_info
        if entry.get("learner_diag")
    ]
    if learner_diag_list:
        _LEARNER_DIAG_AGG_KEYS = [
            "advantage_mean", "advantage_std",
            "clip_fraction",
            "ratio_mean", "ratio_std",
            "old_value_mean", "new_value_mean",
            "value_error_mean", "value_error_std",
        ]
        aggregate["learner_diag"] = _compute_aggregate_generic(
            learner_diag_list, _LEARNER_DIAG_AGG_KEYS,
        )

    # CQ-0139, CQ-0142: reward composition 集約（quantile 含む）
    rc_list = [
        entry["reward_composition"]
        for entry in runs_info
        if entry.get("reward_composition")
    ]
    if rc_list:
        _RC_AGG_KEYS = ["mean", "std", "p50", "p90", "p99"]
        rc_agg: dict = {}
        for comp in ("point_delta", "shanten_delta", "total"):
            comp_dicts = [rc[comp] for rc in rc_list if comp in rc]
            if comp_dicts:
                rc_agg[comp] = _compute_aggregate_generic(
                    comp_dicts, _RC_AGG_KEYS)
        if any(rc.get("shanten_delta_enabled") for rc in rc_list):
            rc_agg["shanten_delta_enabled"] = True
        if rc_agg:
            aggregate["reward_composition"] = rc_agg

    # CQ-0174: eval_before 集約
    eb_list = [
        entry["eval_before"]
        for entry in runs_info
        if entry.get("eval_before")
    ]
    if eb_list:
        aggregate["eval_before"] = _compute_aggregate_generic(
            eb_list,
            ["avg_rank", "avg_score", "win_rate", "deal_in_rate"],
        )

    # CQ-0174: phase_timing 集約
    pt_list = [
        entry["phase_timing"]
        for entry in runs_info
        if entry.get("phase_timing")
    ]
    if pt_list:
        # 全 run に出現するフェーズ名を集める
        all_phases: set[str] = set()
        for pt in pt_list:
            all_phases.update(pt.keys())
        pt_agg: dict = {}
        for phase_name in sorted(all_phases):
            durations = [
                pt[phase_name]["duration_sec"]
                for pt in pt_list
                if phase_name in pt and pt[phase_name].get("duration_sec") is not None
            ]
            if durations:
                n = len(durations)
                mean = sum(durations) / n
                if n > 1:
                    variance = sum((d - mean) ** 2 for d in durations) / (n - 1)
                    std = variance ** 0.5
                else:
                    std = 0.0
                pt_agg[phase_name] = {
                    "mean": round(mean, 3),
                    "std": round(std, 3),
                    "count": n,
                }
        # total_duration_sec も集約
        total_durations = []
        for entry in runs_info:
            if entry.get("phase_timing"):
                total = sum(
                    p.get("duration_sec", 0)
                    for p in entry["phase_timing"].values()
                    if p.get("duration_sec") is not None
                )
                total_durations.append(total)
        if total_durations:
            n = len(total_durations)
            mean = sum(total_durations) / n
            if n > 1:
                variance = sum((d - mean) ** 2 for d in total_durations) / (n - 1)
                std = variance ** 0.5
            else:
                std = 0.0
            pt_agg["total"] = {
                "mean": round(mean, 3),
                "std": round(std, 3),
                "count": n,
            }
        if pt_agg:
            aggregate["phase_timing"] = pt_agg

    # CQ-0180: cycle 別 aggregate
    # 各 run の cycles を cycle_index でまとめて集約
    all_cycles = [entry["cycles"] for entry in runs_info if entry.get("cycles")]
    if all_cycles:
        # cycle_index ごとにメトリクスを集める
        max_cycles = max(len(c) for c in all_cycles)
        cycle_agg: list[dict] = []
        _CYCLE_EVAL_KEYS = ["avg_rank", "avg_score", "win_rate", "deal_in_rate"]
        _CYCLE_DIAG_KEYS = ["clip_fraction", "ratio_std",
                            "advantage_abs_mean_before_clip"]
        for ci in range(max_cycles):
            ci_entries = [c[ci] for c in all_cycles if ci < len(c)]
            ci_agg: dict = {"cycle_index": ci, "count": len(ci_entries)}
            # eval avg_rank 集約
            eval_dicts = [e["eval"] for e in ci_entries if e.get("eval")]
            if eval_dicts:
                ci_agg["eval"] = _compute_aggregate_generic(
                    eval_dicts, _CYCLE_EVAL_KEYS)
            # eval_diff 集約
            diff_ranks = [
                e["eval_diff"]["avg_rank"]["delta"]
                for e in ci_entries
                if e.get("eval_diff") and "avg_rank" in e["eval_diff"]
            ]
            if diff_ranks:
                n = len(diff_ranks)
                mean = sum(diff_ranks) / n
                std = (sum((v - mean) ** 2 for v in diff_ranks) / (n - 1)) ** 0.5 if n > 1 else 0.0
                ci_agg["eval_diff_avg_rank"] = {
                    "mean": round(mean, 6), "std": round(std, 6), "count": n}
            # learner_diag 集約
            diag_dicts = [e["learner_diag"] for e in ci_entries if e.get("learner_diag")]
            if diag_dicts:
                ld_agg = _compute_aggregate_generic(diag_dicts, _CYCLE_DIAG_KEYS)
                # CQ-0200: teacher_agreement 集約
                ta_list = [d["teacher_agreement"] for d in diag_dicts
                           if d.get("teacher_agreement") and d["teacher_agreement"].get("enabled")]
                if ta_list:
                    _TA_KEYS = [
                        "action_match_rate_before", "action_match_rate_after",
                        "best_set_hit_rate_before", "best_set_hit_rate_after",
                        "num_baseline_samples", "num_best_set_samples",
                    ]
                    ld_agg["teacher_agreement"] = _compute_aggregate_generic(ta_list, _TA_KEYS)
                ci_agg["learner_diag"] = ld_agg
            # CQ-0189: actor_type_counts 集約
            atc_list = [e["actor_type_counts"] for e in ci_entries if e.get("actor_type_counts")]
            if atc_list:
                atc_agg: dict = {}
                for key in ("policy", "baseline"):
                    vals = [a.get(key, 0) for a in atc_list]
                    if any(v > 0 for v in vals):
                        n = len(vals)
                        mean = sum(vals) / n
                        atc_agg[key] = {"mean": round(mean, 1), "count": n}
                if atc_agg:
                    ci_agg["actor_type_counts"] = atc_agg
            # CQ-0189: learner_stages 集約
            ls_list = [e["learner_stages"] for e in ci_entries if e.get("learner_stages")]
            if ls_list:
                ls_agg: dict = {}
                # baseline_imitation
                bl_entries = [ls["baseline_imitation"] for ls in ls_list
                              if ls.get("baseline_imitation")]
                if bl_entries:
                    executed_count = sum(1 for b in bl_entries if b.get("executed"))
                    bl_agg_data: dict = {
                        "executed_count": executed_count,
                        "total_count": len(bl_entries),
                    }
                    bl_exec = [b for b in bl_entries if b.get("executed")]
                    if bl_exec:
                        bl_agg_data.update(_compute_aggregate_generic(
                            bl_exec, ["used_samples", "policy_loss"]))
                    ls_agg["baseline_imitation"] = bl_agg_data
                # policy_ppo
                pp_entries = [ls["policy_ppo"] for ls in ls_list
                              if ls.get("policy_ppo")]
                if pp_entries:
                    pp_exec = [p for p in pp_entries if p.get("executed")]
                    if pp_exec:
                        ls_agg["policy_ppo"] = _compute_aggregate_generic(
                            pp_exec, ["used_samples", "policy_loss"])
                if ls_agg:
                    ci_agg["learner_stages"] = ls_agg
            cycle_agg.append(ci_agg)
        aggregate["cycles"] = cycle_agg

    summary = {
        "num_seeds": len(seeds),
        "seeds": seeds,
        "success_count": len(success_runs),
        "failure_count": len(failure_runs),
        "success_rate": len(success_runs) / len(seeds) if seeds else 0.0,
        "runs": runs_info,
        "aggregate": aggregate,
    }

    # batch_summary.json
    with open(batch_dir / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # batch_table.csv
    _write_batch_table_csv(batch_dir / "batch_table.csv", runs_info)


def _compute_aggregate_generic(
    metrics_list: list[dict], keys: list[str],
) -> dict:
    """汎用メトリクス集約 (CQ-0127)"""
    result = {}
    for key in keys:
        values = [m[key] for m in metrics_list if m.get(key) is not None]
        if not values:
            continue
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
        result[key] = {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "count": n,
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    return result


def _compute_aggregate(eval_metrics_list: list[dict]) -> dict:
    """eval_metrics のリストから集約統計を計算する"""
    if not eval_metrics_list:
        return {}

    metrics_keys = ["avg_rank", "avg_score", "win_rate", "deal_in_rate"]
    aggregate = {}

    for key in metrics_keys:
        values = [em[key] for em in eval_metrics_list if em.get(key) is not None]
        if not values:
            continue
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        # SE / 95% CI (CQ-0094)
        if n > 1:
            se = std / math.sqrt(n)
            t = _t_value_95(n)
            ci_lower = mean - t * se
            ci_upper = mean + t * se
        else:
            se = 0.0
            ci_lower = mean
            ci_upper = mean
        aggregate[key] = {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "se": round(se, 6),
            "ci_95_lower": round(ci_lower, 6),
            "ci_95_upper": round(ci_upper, 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "count": n,
        }

    return aggregate


# 簡易 t 分布テーブル: 自由度 → 95% 両側臨界値 (CQ-0094)
_T_TABLE_95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
}


def _t_value_95(n: int) -> float:
    """自由度 n-1 の t 分布 95% 両側臨界値を返す"""
    if n <= 1:
        return 0.0
    df = n - 1
    if df in _T_TABLE_95:
        return _T_TABLE_95[df]
    if df > 30:
        return 1.96
    # テーブルにない df は、それ以下の最大の df の値を使う（保守的）
    candidates = [k for k in _T_TABLE_95 if k <= df]
    return _T_TABLE_95[max(candidates)] if candidates else 1.96


def _attach_outlier_info(
    aggregate: dict,
    eval_metrics_list: list[dict],
    success_runs: list[dict],
) -> None:
    """aggregate の各メトリクスに outlier_min/outlier_max を付加する (CQ-0094)"""
    metrics_keys = ["avg_rank", "avg_score", "win_rate", "deal_in_rate"]
    for key in metrics_keys:
        if key not in aggregate:
            continue
        min_val = aggregate[key]["min"]
        max_val = aggregate[key]["max"]
        outlier_min: dict | None = None
        outlier_max: dict | None = None
        for em, r in zip(eval_metrics_list, success_runs):
            v = em.get(key)
            if v is None:
                continue
            result = r.get("result", {})
            if outlier_min is None or round(v, 6) == min_val:
                outlier_min = {
                    "seed": r["seed"],
                    "run_dir": result.get("run_dir", ""),
                    "value": round(v, 6),
                }
            if outlier_max is None or round(v, 6) == max_val:
                outlier_max = {
                    "seed": r["seed"],
                    "run_dir": result.get("run_dir", ""),
                    "value": round(v, 6),
                }
        if outlier_min is not None:
            aggregate[key]["outlier_min"] = outlier_min
        if outlier_max is not None:
            aggregate[key]["outlier_max"] = outlier_max


def _write_batch_table_csv(path: Path, runs_info: list[dict]) -> None:
    """バッチ結果の CSV テーブルを書き出す"""
    fieldnames = [
        "seed", "success", "run_dir", "eval_mode",
        "avg_rank", "avg_score", "win_rate", "deal_in_rate",
        "teacher_top1_match_rate", "teacher_best_set_hit_rate",
        "imitation_loss_mode",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for run in runs_info:
            row: dict = {
                "seed": run["seed"],
                "success": run["success"],
                "run_dir": run.get("run_dir", ""),
                "eval_mode": run.get("eval_mode", ""),
            }
            em = run.get("eval_metrics", {})
            if em:
                row["avg_rank"] = em.get("avg_rank", "")
                row["avg_score"] = em.get("avg_score", "")
                row["win_rate"] = em.get("win_rate", "")
                row["deal_in_rate"] = em.get("deal_in_rate", "")
            # imitation 教師再現メトリクス (CQ-0127)
            im = run.get("imitation_metrics", {})
            if im:
                row["teacher_top1_match_rate"] = im.get("teacher_top1_match_rate", "")
                row["teacher_best_set_hit_rate"] = im.get("teacher_best_set_hit_rate", "")
                row["imitation_loss_mode"] = im.get("imitation_loss_mode", "")
            writer.writerow(row)
