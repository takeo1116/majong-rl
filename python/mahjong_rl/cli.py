"""Stage 1 実験の CLI エントリポイント"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mahjong_rl.experiment import ExperimentConfig
from mahjong_rl.runner import Stage1Runner


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
        "--verbose", "-v", action="store_true",
        help="詳細ログを標準出力に表示する")

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

    # 実行
    runner = Stage1Runner(config=config, base_dir=Path(args.base_dir))
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
