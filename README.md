# majong-rl

麻雀の強化学習をゲームエンジンから作る

## 概要

日本式リーチ麻雀（4人打ち）のゲームエンジンを C++ で構築し、その上で動作する強化学習基盤を Python で整備するプロジェクト。

- **ゲームエンジン**: C++20 製。半荘進行・合法手列挙・和了判定・符計算・点数計算・精算を実装
- **学習基盤**: PyTorch ベース。pybind11 経由でエンジンを呼び出し、self-play → shard 保存 → PPO/Imitation 学習 → 評価のパイプラインを提供
- **段階的設計**: Stage 1 (打牌のみ学習) から段階的に拡張する構成

## ビルド方法

```bash
# C++ エンジンのビルド
mkdir build && cd build
cmake ..
make -j$(nproc)
```

```bash
# Python パッケージのインストール（開発モード）
pip install -e ".[dev]"
```

## テスト実行

```bash
# C++ テスト (361 tests)
cd build
ctest --output-on-failure
```

```bash
# Python テスト (169 tests)
python3 -m pytest tests/python/ -v
```

## サンプル実行

```bash
cd build
./mahjong_example
```

## ディレクトリ構成

```
src/
  core/       - 牌・副露・状態・行動・イベントの基礎構造
  rules/      - 和了判定、役判定、符計算、点数計算
  engine/     - reset/step、合法手列挙、応答解決
  rl/         - Observation、Reward、環境ラッパー (C++)
  io/         - ログ、文字列表現、CLI

bindings/     - pybind11 バインディング

python/mahjong_rl/
  env/        - Stage1Env (打牌専用環境ラッパー)
  encoders/   - FlatFeatureEncoder, ChannelTensorEncoder
  models/     - MLPPolicyValueModel
  baseline/   - 向聴数ベースライン
  shard.py    - 学習サンプル Parquet 入出力
  selfplay_worker.py  - Self-play データ生成
  learner.py          - PPO / Imitation 学習
  evaluator.py        - 評価対戦ランナー
  experiment.py       - 実験設定 YAML・Run ディレクトリ管理
  action_selector.py  - 行動選択 (argmax / sampling)

configs/      - 実験設定 YAML テンプレート
tests/
  unit/         - C++ 単体テスト
  integration/  - C++ 統合テスト
  replay/       - C++ 再現テスト
  python/       - Python テスト
docs/         - GAME_RULE.md, GAME_SPEC.md, RL_RULE.md, RL_SPEC.md
examples/     - サンプル実行
```

## ドキュメント

| ファイル | 内容 |
|---|---|
| `docs/GAME_RULE.md` | 採用する麻雀ルールの定義 |
| `docs/GAME_SPEC.md` | ゲームエンジン実装仕様 |
| `docs/RL_RULE.md` | 学習方針・研究方針 |
| `docs/RL_SPEC.md` | 学習システム実装仕様 |

## 技術スタック

- **ゲームエンジン**: C++20, CMake, GoogleTest
- **学習基盤**: Python, PyTorch, pybind11
- **データ保存**: Apache Parquet (PyArrow)
- **実験設定**: YAML (PyYAML)
