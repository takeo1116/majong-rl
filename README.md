# majong-rl

日本式リーチ麻雀の C++ ゲームエンジンと、Python/PyTorch ベースの強化学習基盤を統合したプロジェクト。

現時点の中心スコープは Stage 1（打牌学習）で、以下の実験ループを提供する。

- self-play（multi-process 対応）
- learner（PPO / imitation）
- evaluation（single / rotation、multi-process 対応）
- multi-seed batch / sweep / resume

## 1. セットアップ

前提:

- Python 3.10+
- CMake / C++20 コンパイラ

```bash
# 開発用依存込みでインストール
pip install -e ".[dev]"
```

```bash
# C++ 側をビルド
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

## 2. テスト

```bash
# C++ テスト
cd build
ctest --output-on-failure
```

```bash
# Python smoke（推奨: 日常確認）
python3 -m pytest tests/python/ -m smoke -v
```

```bash
# Python 全テスト
python3 -m pytest tests/python/ -v
```

テストマーカー:

- `smoke`: 軽量・日常確認向け
- `slow`: 重い統合系（`-m "not slow"` で除外可能）

## 3. 実験実行（CLI）

エントリポイント: `python3 -m mahjong_rl.cli`

### 3.1 config バリデーションのみ

```bash
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --validate-only
```

### 3.2 単一 run

```bash
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml
```

### 3.3 multi-seed batch

```bash
# 明示 seed
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --seeds 42,43,44

# 範囲指定
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --seed-start 42 \
  --num-seeds 5
```

### 3.4 batch resume

```bash
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --seeds 42,43,44,45,46 \
  --resume runs/<existing_batch_dir>
```

### 3.5 sweep（条件比較）

```bash
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --sweep-file sweep.yaml \
  --seeds 42,43,44
```

### 3.6 sweep resume（condition 単位）

```bash
python3 -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --sweep-file sweep.yaml \
  --seeds 42,43,44 \
  --resume runs/<existing_sweep_dir>
```

## 4. 主な config

- `configs/stage1_full_flat_mlp_ppo.yaml`
- `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- `configs/stage1_full_flat_mlp_ppo_rotation_eval.yaml`
- `configs/stage1_full_flat_mlp_ppo_eval_fast.yaml`
- `configs/stage1_full_flat_mlp_ppo_eval_strict.yaml`

## 5. 生成物

- `runs/`: 実験の生データ（run_dir、batch_dir、sweep_dir）
  - `summary.json`, `run.log`, `notes.md`, `checkpoints/`, `selfplay/`, `eval/`, `profile.json` など
- `experiments/`: runbook と report（Git 管理）
  - `exp_XXX/runbook.md`
  - `exp_XXX/report.md`

運用ルールは `experiments/README.md` を参照。

## 6. ディレクトリ概要

```text
src/                 C++ ゲームエンジン
bindings/            pybind11 バインディング
python/mahjong_rl/   RL 実装（env/encoders/models/runner/cli など）
configs/             実験設定 YAML
tests/               C++/Python テスト
docs/                仕様・ルール・運用ドキュメント
experiments/         runbook/report（実験ごとの記録）
runs/                実験生データ（Git 管理外）
```

## 7. 仕様ドキュメント

- `docs/GAME_RULE.md`: ゲームルール定義
- `docs/GAME_SPEC.md`: ゲーム実装仕様
- `docs/RL_RULE.md`: RL 側ルール
- `docs/RL_SPEC.md`: RL 実装仕様

## 8. 補足

- `python` コマンドがない環境では `python3` を使用する。
- GPU 利用時は PyTorch 側の CUDA 可用性に依存する（`torch.cuda.is_available()`）。
