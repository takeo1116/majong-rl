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

    # バッチ実行 or 単一実行
    if seeds is not None:
        stop_on_error = not args.continue_on_error
        return run_batch(config, seeds, base_dir, stop_on_error=stop_on_error)

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
