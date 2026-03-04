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

    # 成功 run から eval_metrics を収集
    eval_metrics_list = []
    for r in success_runs:
        result = r.get("result", {})
        em = result.get("eval_metrics")
        if em is not None:
            eval_metrics_list.append(em)

    # 集約統計
    aggregate = _compute_aggregate(eval_metrics_list)

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
            # device_info / env_info (CQ-0081)
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
        else:
            entry["error"] = r.get("error", "unknown")
        runs_info.append(entry)

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
        aggregate[key] = {
            "mean": round(mean, 6),
            "std": round(std, 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "count": n,
        }

    return aggregate


def _write_batch_table_csv(path: Path, runs_info: list[dict]) -> None:
    """バッチ結果の CSV テーブルを書き出す"""
    fieldnames = [
        "seed", "success", "run_dir",
        "avg_rank", "avg_score", "win_rate", "deal_in_rate",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for run in runs_info:
            row: dict = {
                "seed": run["seed"],
                "success": run["success"],
                "run_dir": run.get("run_dir", ""),
            }
            em = run.get("eval_metrics", {})
            if em:
                row["avg_rank"] = em.get("avg_rank", "")
                row["avg_score"] = em.get("avg_score", "")
                row["win_rate"] = em.get("win_rate", "")
                row["deal_in_rate"] = em.get("deal_in_rate", "")
            writer.writerow(row)
