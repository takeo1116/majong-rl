"""Stage 1 実験の CLI エントリポイント"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

from mahjong_rl.experiment import ExperimentConfig
from mahjong_rl.runner import Stage1Runner

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """CLI メインエントリポイント

    Returns:
        終了コード (0: 成功, 1: 失敗)
    """
    parser = argparse.ArgumentParser(
        description="Stage 1 麻雀 RL 実験ランナー")
    parser.add_argument(
        "--config", "-c", required=True, type=str,
        help="実験設定 YAML ファイルのパス")
    parser.add_argument(
        "--base-dir", "-d", type=str, default="runs",
        help="run ディレクトリの親ディレクトリ (デフォルト: runs)")
    parser.add_argument(
        "--override", "-o", type=str, nargs="*", default=[],
        help="config の上書き (key=value 形式, 例: experiment.global_seed=42)")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="config のバリデーションのみ実行し、実験は開始しない")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="詳細ログを標準出力に表示する")

    # マルチ seed バッチ実行 (CQ-0077)
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="カンマ区切りの seed リスト (例: 42,43,44)")
    parser.add_argument(
        "--seed-start", type=int, default=None,
        help="seed 開始値 (--num-seeds と併用)")
    parser.add_argument(
        "--num-seeds", type=int, default=None,
        help="seed 個数 (--seed-start と併用)")
    parser.add_argument(
        "--stop-on-error", action="store_true", default=True,
        help="バッチ実行中にエラーが発生したら停止する (デフォルト)")
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="バッチ実行中にエラーが発生しても続行する")

    # resume (CQ-0096)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="既存 batch_dir を指定してバッチ再開（完了 seed をスキップ）")

    # sweep (CQ-0095)
    parser.add_argument(
        "--sweep-file", type=str, default=None,
        help="sweep 設定 YAML ファイルのパス")

    # phase 単位 resume (CQ-0111)
    parser.add_argument(
        "--resume-run", type=str, default=None,
        help="既存 run_dir を指定して phase 単位 resume（完了済み phase をスキップ）")

    # 成果物再利用 (CQ-0110)
    parser.add_argument(
        "--reuse-from", type=str, default=None,
        help="参照元 run_dir のパス（成果物を再利用）")
    parser.add_argument(
        "--reuse-phases", type=str, default=None,
        help="再利用する phase のカンマ区切り (例: imitation,selfplay,eval_before)")

    args = parser.parse_args(argv)

    # ログ設定
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # config 読み込み
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"エラー: config ファイルが見つかりません: {config_path}",
              file=sys.stderr)
        return 1

    config = ExperimentConfig.from_yaml(config_path)

    # override 適用
    for override in args.override:
        if "=" not in override:
            print(f"エラー: override は key=value 形式で指定してください: {override}",
                  file=sys.stderr)
            return 1
        key, value = override.split("=", 1)
        try:
            _apply_override(config, key, value)
        except ValueError as e:
            print(f"エラー: override 適用失敗: {e}", file=sys.stderr)
            return 1

    # seed リスト構築
    seeds, err = _resolve_seeds(args)
    if err is not None:
        print(f"エラー: {err}", file=sys.stderr)
        return 1

    base_dir = Path(args.base_dir)

    # --validate-only: バリデーションのみ実行
    runner = Stage1Runner(config=config, base_dir=base_dir)
    if args.validate_only:
        errors = runner.validate_config()
        if errors:
            print("config バリデーションエラー:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 1
        print("config バリデーション OK")
        return 0

    stop_on_error = not args.continue_on_error

    # CQ-0116: 競合引数バリデーション
    _single_run_opts = []
    if args.resume_run:
        _single_run_opts.append("--resume-run")
    if args.reuse_from:
        _single_run_opts.append("--reuse-from")

    if _single_run_opts and seeds is not None:
        opts = " と ".join(_single_run_opts)
        print(f"エラー: {opts} は --seeds/--seed-start と併用できません"
              "（単一 run 用オプションです）",
              file=sys.stderr)
        return 1

    if args.resume_run and args.reuse_from:
        print("エラー: --resume-run と --reuse-from は同時に指定できません",
              file=sys.stderr)
        return 1

    if args.resume_run and args.resume:
        print("エラー: --resume-run と --resume は同時に指定できません"
              "（--resume-run は単一 run、--resume は batch 用です）",
              file=sys.stderr)
        return 1

    if args.reuse_from and args.resume:
        print("エラー: --reuse-from と --resume は同時に指定できません",
              file=sys.stderr)
        return 1

    # sweep + resume 実行 (CQ-0101)
    if args.sweep_file and args.resume:
        if seeds is None:
            print("エラー: --sweep-file + --resume には --seeds または"
                  " --seed-start/--num-seeds が必要です",
                  file=sys.stderr)
            return 1
        sweep_config = _load_sweep_config(Path(args.sweep_file))
        if sweep_config is None:
            return 1
        return run_sweep_resume(config, seeds, sweep_config,
                                Path(args.resume),
                                stop_on_error=stop_on_error)

    # sweep 実行 (CQ-0095)
    if args.sweep_file:
        if seeds is None:
            print("エラー: --sweep-file には --seeds または --seed-start/--num-seeds が必要です",
                  file=sys.stderr)
            return 1
        sweep_config = _load_sweep_config(Path(args.sweep_file))
        if sweep_config is None:
            return 1
        return run_sweep(config, seeds, sweep_config, base_dir,
                         stop_on_error=stop_on_error)

    # resume 実行 (CQ-0096)
    if args.resume:
        if seeds is None:
            print("エラー: --resume には --seeds または --seed-start/--num-seeds が必要です",
                  file=sys.stderr)
            return 1
        return run_batch_resume(config, seeds, Path(args.resume),
                                stop_on_error=stop_on_error)

    # バッチ実行 or 単一実行
    if seeds is not None:
        return run_batch(config, seeds, base_dir, stop_on_error=stop_on_error)

    # phase 単位 resume (CQ-0111)
    if args.resume_run:
        runner = Stage1Runner(config=config, base_dir=base_dir,
                              resume_run_dir=args.resume_run)
        try:
            result = runner.run()
        except ValueError as e:
            print(f"エラー: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"予期しないエラー: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        if "error" in result:
            print(f"実験失敗: {result['error']}", file=sys.stderr)
            return 1
        print(f"resume 完了: {result['run_dir']}")
        return 0

    # 成果物再利用 (CQ-0110)
    if args.reuse_from:
        reuse_phases = (args.reuse_phases.split(",") if args.reuse_phases
                        else ["imitation", "selfplay", "eval_before"])
        reuse_from = {"run_dir": args.reuse_from, "phases": reuse_phases}
        runner = Stage1Runner(config=config, base_dir=base_dir,
                              reuse_from=reuse_from)
        try:
            result = runner.run()
        except ValueError as e:
            print(f"エラー: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"予期しないエラー: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        if "error" in result:
            print(f"実験失敗: {result['error']}", file=sys.stderr)
            return 1
        print(f"reuse 実行完了: {result['run_dir']}")
        return 0

    # 単一 run 経路（従来互換）
    try:
        result = runner.run()
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"予期しないエラー: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if "error" in result:
        print(f"実験失敗: {result['error']}", file=sys.stderr)
        return 1

    print(f"実験完了: {result['run_dir']}")
    return 0


def _resolve_seeds(args) -> tuple[list[int] | None, str | None]:
    """CLI 引数から seed リストを構築する

    Returns:
        (seed_list, error_message)
        seed_list が None → 単一 run 経路
    """
    has_seeds = args.seeds is not None
    has_seed_start = args.seed_start is not None
    has_num_seeds = args.num_seeds is not None

    if has_seeds and (has_seed_start or has_num_seeds):
        return None, "--seeds と --seed-start/--num-seeds は同時に指定できません"

    if has_seeds:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
        except ValueError:
            return None, "--seeds の値は整数のカンマ区切りで指定してください"
        if not seeds:
            return None, "--seeds に 1 つ以上の seed を指定してください"
        return seeds, None

    if has_seed_start or has_num_seeds:
        if not (has_seed_start and has_num_seeds):
            return None, "--seed-start と --num-seeds は両方指定してください"
        if args.num_seeds < 1:
            return None, "--num-seeds は 1 以上で指定してください"
        seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
        return seeds, None

    return None, None


def run_batch(
    config: ExperimentConfig,
    seeds: list[int],
    base_dir: Path,
    stop_on_error: bool = True,
) -> int:
    """マルチ seed バッチ実行

    各 seed を独立した run_dir で実行し、集約レポートを生成する。

    Returns:
        終了コード (0: 全成功, 1: 失敗あり)
    """
    from mahjong_rl.batch_report import generate_batch_report

    batch_id = uuid.uuid4().hex[:8]
    date_str = datetime.now().strftime("%Y%m%d")
    name = config.experiment.get("name", "batch")
    batch_dir = base_dir / f"{date_str}_{name}_batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"バッチ実行開始: {len(seeds)} seeds, batch_dir={batch_dir}")

    results: list[dict] = []
    for i, seed in enumerate(seeds):
        logger.info(f"[Seed {i + 1}/{len(seeds)}] seed={seed}")
        config_copy = copy.deepcopy(config)
        config_copy.experiment["global_seed"] = seed

        runner = Stage1Runner(config=config_copy, base_dir=batch_dir)
        try:
            result = runner.run()
            success = "error" not in result
            entry: dict = {
                "seed": seed,
                "success": success,
                "result": result,
            }
            if not success:
                entry["error"] = result.get("error", "unknown")
            results.append(entry)
            if not success:
                logger.warning(f"  seed={seed} 失敗: {result.get('error')}")
                if stop_on_error:
                    logger.info("stop-on-error: バッチ中断")
                    break
        except Exception as e:
            logger.error(f"  seed={seed} 例外: {type(e).__name__}: {e}")
            results.append({
                "seed": seed,
                "success": False,
                "error": str(e),
            })
            if stop_on_error:
                logger.info("stop-on-error: バッチ中断")
                break

    # 集約レポート生成
    generate_batch_report(batch_dir, results)

    all_success = all(r["success"] for r in results)
    completed = len(results)
    success_count = sum(1 for r in results if r["success"])
    logger.info(
        f"バッチ完了: {success_count}/{completed} 成功, batch_dir={batch_dir}")

    if all_success:
        print(f"バッチ実行完了: {batch_dir}")
    else:
        print(f"バッチ実行完了 (一部失敗あり): {batch_dir}", file=sys.stderr)

    return 0 if all_success else 1


def run_batch_resume(
    config: ExperimentConfig,
    seeds: list[int],
    batch_dir: Path,
    stop_on_error: bool = True,
) -> int:
    """バッチ再開: 完了済み seed をスキップして未完了 seed のみ実行する (CQ-0096)

    Returns:
        終了コード (0: 全成功, 1: 失敗あり)
    """
    from mahjong_rl.batch_report import generate_batch_report

    if not batch_dir.exists():
        print(f"エラー: batch_dir が見つかりません: {batch_dir}", file=sys.stderr)
        return 1

    # 完了済み seed を検出
    completed = _detect_completed_seeds(batch_dir)
    skip_seeds = set(completed.keys())
    pending_seeds = [s for s in seeds if s not in skip_seeds]

    logger.info(f"resume: 完了済み {len(skip_seeds)} seeds, 未完了 {len(pending_seeds)} seeds")
    if not pending_seeds:
        logger.info("全 seed 完了済み")
        print(f"全 seed 完了済み: {batch_dir}")
        # レポートを再生成
        results = [completed[s] for s in seeds if s in completed]
        generate_batch_report(batch_dir, results)
        return 0

    # 完了済み seed の結果を収集
    results: list[dict] = []
    for seed in seeds:
        if seed in completed:
            results.append(completed[seed])
            continue

        logger.info(f"[Resume] seed={seed}")
        config_copy = copy.deepcopy(config)
        config_copy.experiment["global_seed"] = seed

        runner = Stage1Runner(config=config_copy, base_dir=batch_dir)
        try:
            result = runner.run()
            success = "error" not in result
            entry: dict = {
                "seed": seed,
                "success": success,
                "result": result,
            }
            if not success:
                entry["error"] = result.get("error", "unknown")
            results.append(entry)
            if not success:
                logger.warning(f"  seed={seed} 失敗: {result.get('error')}")
                if stop_on_error:
                    logger.info("stop-on-error: バッチ中断")
                    break
        except Exception as e:
            logger.error(f"  seed={seed} 例外: {type(e).__name__}: {e}")
            results.append({
                "seed": seed,
                "success": False,
                "error": str(e),
            })
            if stop_on_error:
                logger.info("stop-on-error: バッチ中断")
                break

    # 集約レポート再生成
    generate_batch_report(batch_dir, results)

    all_success = all(r["success"] for r in results)
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"resume 完了: {success_count}/{len(results)} 成功, batch_dir={batch_dir}")

    if all_success:
        print(f"バッチ再開完了: {batch_dir}")
    else:
        print(f"バッチ再開完了 (一部失敗あり): {batch_dir}", file=sys.stderr)

    return 0 if all_success else 1


def _detect_completed_seeds(batch_dir: Path) -> dict[int, dict]:
    """batch_dir 内の完了済み seed を検出する (CQ-0096)

    Returns:
        seed → result entry dict のマッピング
    """
    completed: dict[int, dict] = {}
    for run_dir in sorted(batch_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not summary.get("success", False):
            continue
        seed = summary.get("global_seed")
        if seed is None:
            continue
        completed[seed] = _restore_result_from_summary(run_dir, summary)
    return completed


def _restore_result_from_summary(run_dir: Path, summary: dict) -> dict:
    """summary.json から batch result entry を復元する (CQ-0096)"""
    ps = summary.get("phase_stats", {})
    eval_stats = ps.get("eval", {})
    sp_stats = ps.get("selfplay", {})

    result: dict = {
        "run_dir": str(run_dir),
        "global_seed": summary.get("global_seed"),
    }
    if eval_stats:
        result["eval_metrics"] = eval_stats
    if sp_stats:
        result["selfplay_stats"] = sp_stats

    return {
        "seed": summary.get("global_seed"),
        "success": True,
        "result": result,
    }


def run_sweep(
    config: ExperimentConfig,
    seeds: list[int],
    sweep_config: dict,
    base_dir: Path,
    stop_on_error: bool = True,
) -> int:
    """sweep 実行: 複数条件を連続でバッチ実行する (CQ-0095)

    Returns:
        終了コード (0: 全成功, 1: 失敗あり)
    """
    import csv as csv_mod

    conditions = sweep_config.get("conditions", [])
    if not conditions:
        print("エラー: sweep 設定に conditions がありません", file=sys.stderr)
        return 1

    sweep_id = uuid.uuid4().hex[:8]
    date_str = datetime.now().strftime("%Y%m%d")
    name = config.experiment.get("name", "sweep")
    sweep_dir = base_dir / f"{date_str}_{name}_sweep_{sweep_id}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"sweep 開始: {len(conditions)} 条件, {len(seeds)} seeds, sweep_dir={sweep_dir}")

    all_success = True
    condition_results: list[dict] = []

    for cond in conditions:
        cond_id = cond.get("condition_id", "unnamed")
        overrides = cond.get("overrides", {})
        logger.info(f"[Condition] {cond_id}: {overrides}")

        cond_config = copy.deepcopy(config)
        for key, value in overrides.items():
            try:
                _apply_override(cond_config, key, str(value))
            except ValueError as e:
                logger.error(f"  override 適用失敗: {e}")
                all_success = False
                condition_results.append({
                    "condition_id": cond_id,
                    "success": False,
                    "error": str(e),
                })
                continue

        cond_dir = sweep_dir / cond_id
        ret = run_batch(cond_config, seeds, cond_dir, stop_on_error=stop_on_error)
        if ret != 0:
            all_success = False

        # batch_summary から結果を読み取り
        cond_entry = {"condition_id": cond_id, "batch_dir": str(cond_dir)}
        batch_summary_path = _find_batch_summary(cond_dir)
        if batch_summary_path is not None:
            with open(batch_summary_path) as f:
                bs = json.load(f)
            cond_entry["success_rate"] = bs.get("success_rate", 0.0)
            cond_entry["aggregate"] = bs.get("aggregate", {})
        else:
            cond_entry["success_rate"] = 0.0
            cond_entry["aggregate"] = {}
        condition_results.append(cond_entry)

    # ランキング表生成
    _generate_sweep_ranking(sweep_dir, condition_results)

    logger.info(f"sweep 完了: sweep_dir={sweep_dir}")
    if all_success:
        print(f"sweep 完了: {sweep_dir}")
    else:
        print(f"sweep 完了 (一部失敗あり): {sweep_dir}", file=sys.stderr)

    return 0 if all_success else 1


def run_sweep_resume(
    config: ExperimentConfig,
    seeds: list[int],
    sweep_config: dict,
    sweep_dir: Path,
    stop_on_error: bool = True,
) -> int:
    """sweep の condition 単位 resume (CQ-0101)

    既存 sweep_dir の各 condition を resume し、未完了 seed のみ再実行する。
    sweep_dir にない condition は新規実行する。

    Returns:
        終了コード (0: 全成功, 1: 失敗あり)
    """
    if not sweep_dir.exists():
        print(f"エラー: sweep_dir が見つかりません: {sweep_dir}", file=sys.stderr)
        return 1

    conditions = sweep_config.get("conditions", [])
    if not conditions:
        print("エラー: sweep 設定に conditions がありません", file=sys.stderr)
        return 1

    logger.info(f"sweep resume 開始: {len(conditions)} 条件, {len(seeds)} seeds, "
                f"sweep_dir={sweep_dir}")

    all_success = True
    condition_results: list[dict] = []

    for cond in conditions:
        cond_id = cond.get("condition_id", "unnamed")
        overrides = cond.get("overrides", {})
        logger.info(f"[Condition Resume] {cond_id}: {overrides}")

        cond_config = copy.deepcopy(config)
        for key, value in overrides.items():
            try:
                _apply_override(cond_config, key, str(value))
            except ValueError as e:
                logger.error(f"  override 適用失敗: {e}")
                all_success = False
                condition_results.append({
                    "condition_id": cond_id,
                    "success": False,
                    "error": str(e),
                })
                continue

        cond_dir = sweep_dir / cond_id
        batch_dir = _find_batch_dir(cond_dir) if cond_dir.exists() else None

        if batch_dir is not None:
            # 既存 condition を resume
            logger.info(f"  resume: batch_dir={batch_dir}")
            ret = run_batch_resume(cond_config, seeds, batch_dir,
                                   stop_on_error=stop_on_error)
        else:
            # 新規 condition を実行
            logger.info(f"  新規実行: cond_dir={cond_dir}")
            ret = run_batch(cond_config, seeds, cond_dir,
                            stop_on_error=stop_on_error)

        if ret != 0:
            all_success = False

        # batch_summary から結果を読み取り
        cond_entry: dict = {"condition_id": cond_id, "batch_dir": str(cond_dir)}
        batch_summary_path = _find_batch_summary(cond_dir)
        if batch_summary_path is not None:
            with open(batch_summary_path) as f:
                bs = json.load(f)
            cond_entry["success_rate"] = bs.get("success_rate", 0.0)
            cond_entry["aggregate"] = bs.get("aggregate", {})
        else:
            cond_entry["success_rate"] = 0.0
            cond_entry["aggregate"] = {}
        condition_results.append(cond_entry)

    # ランキング表再生成
    _generate_sweep_ranking(sweep_dir, condition_results)

    logger.info(f"sweep resume 完了: sweep_dir={sweep_dir}")
    if all_success:
        print(f"sweep resume 完了: {sweep_dir}")
    else:
        print(f"sweep resume 完了 (一部失敗あり): {sweep_dir}", file=sys.stderr)

    return 0 if all_success else 1


def _find_batch_dir(cond_dir: Path) -> Path | None:
    """condition ディレクトリ内の batch_dir を探す (CQ-0101)

    run_batch は cond_dir 配下に日付付き batch_dir を作るため、
    1 階層下を探す。batch_summary.json がある dir を優先。
    """
    if not cond_dir.exists():
        return None
    # batch_summary.json がある dir を優先
    for d in sorted(cond_dir.iterdir()):
        if d.is_dir() and (d / "batch_summary.json").exists():
            return d
    # なければ最初のサブディレクトリを返す
    for d in sorted(cond_dir.iterdir()):
        if d.is_dir():
            return d
    return None


def _find_batch_summary(cond_dir: Path) -> Path | None:
    """条件ディレクトリ内の batch_summary.json を探す"""
    # run_batch は cond_dir 配下に batch_dir を作る
    for d in sorted(cond_dir.iterdir()):
        if d.is_dir():
            p = d / "batch_summary.json"
            if p.exists():
                return p
    return None


def _generate_sweep_ranking(sweep_dir: Path, condition_results: list[dict]) -> None:
    """条件間ランキング表を生成する (CQ-0095)"""
    import csv as csv_mod

    metrics_keys = ["avg_rank", "avg_score", "win_rate", "deal_in_rate"]
    rows = []
    for cr in condition_results:
        row: dict = {"condition_id": cr.get("condition_id", "")}
        row["success_rate"] = cr.get("success_rate", 0.0)
        agg = cr.get("aggregate", {})
        for key in metrics_keys:
            m = agg.get(key, {})
            row[f"{key}_mean"] = m.get("mean", "")
            row[f"{key}_ci_95_lower"] = m.get("ci_95_lower", "")
            row[f"{key}_ci_95_upper"] = m.get("ci_95_upper", "")
        rows.append(row)

    # avg_rank_mean 昇順ソート（空文字は末尾）
    rows.sort(key=lambda r: (
        r["avg_rank_mean"] if isinstance(r["avg_rank_mean"], (int, float)) else float("inf")
    ))

    # JSON
    with open(sweep_dir / "sweep_ranking.json", "w") as f:
        json.dump({"conditions": rows}, f, indent=2, ensure_ascii=False)

    # CSV
    fieldnames = ["condition_id", "success_rate"]
    for key in metrics_keys:
        fieldnames.extend([f"{key}_mean", f"{key}_ci_95_lower", f"{key}_ci_95_upper"])

    with open(sweep_dir / "sweep_ranking.csv", "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_sweep_config(path: Path) -> dict | None:
    """sweep 設定 YAML を読み込む (CQ-0095)"""
    import yaml

    if not path.exists():
        print(f"エラー: sweep ファイルが見つかりません: {path}", file=sys.stderr)
        return None
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"エラー: sweep ファイル読み込み失敗: {e}", file=sys.stderr)
        return None
    if not isinstance(data, dict) or "conditions" not in data:
        print("エラー: sweep ファイルに conditions キーが必要です", file=sys.stderr)
        return None
    return data


def _apply_override(config: ExperimentConfig, key: str, value: str) -> None:
    """ドット区切りのキーで config を上書きする"""
    parts = key.split(".")
    if len(parts) != 2:
        raise ValueError(f"override キーは section.key 形式で指定してください: {key}")

    section_name, field_name = parts
    section = getattr(config, section_name, None)
    if section is None or not isinstance(section, dict):
        raise ValueError(f"不正な config セクション: {section_name}")

    # 型推定
    parsed = _parse_value(value)
    section[field_name] = parsed


def _parse_value(value: str):
    """文字列を適切な型に変換する"""
    # JSON として解析を試みる (リスト、数値、bool 対応)
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


if __name__ == "__main__":
    sys.exit(main())
