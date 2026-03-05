# EXPERIMENT_STAGE1_RUNBOOK.md（並列化対応版）

最終更新: 2026-03-04  
対象: Stage 1 (DiscardOnly) / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel / PPO  
目的: Stage 1 の学習・評価実験を、単機 multi-process（1 worker = 1 process = 1 thread 相当）で高速に回す

---

## 0. 前提

- 実験は CLI で起動する
- self-play と evaluation は multi-process を利用できる
- learner は当面 single-process のまま
- 1 worker = 1 thread 相当運用を想定（内部スレッドは抑制される）
- 最初の目的は「強いモデルを得ること」ではなく、**学習実験が安定して回り、比較可能な結果が取れること**を確認すること

---

## 1. セットアップ

### 1.1 依存導入
```bash
pip install -e ".[dev]"
```

### 1.2 smoke test（軽量）
```bash
python3 -m pytest tests/python/ -m smoke -v
```

### 1.3 config validation（CLIに validate-only がない場合の代替）
```bash
python - <<'PY'
from pathlib import Path
from mahjong_rl.experiment import ExperimentConfig
from mahjong_rl.runner import Stage1Runner
cfg = ExperimentConfig.from_yaml(Path("configs/stage1_full_flat_mlp_ppo.yaml"))
print(Stage1Runner(cfg).validate_config())
PY
```

---

## 2. 実験の起動（基本形）

### 2.1 最小の起動例（verbose）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --verbose
```

### 2.2 override の基本例（推奨: seed + device 明示）
※ `--override` は「2階層まで」しか使えない（`a.b.c` は不可）。

```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

### 2.3 終了判定
- CLI 終了コード `0`: 成功
- CLI 終了コード `1`: 失敗
- run 後は `summary.json.success` と `summary.json.error` で再確認する

---

## 3. 実験の推奨プリセット

### 3.1 Run A（warm startなし / PPO直学習）
第一候補:
- `configs/stage1_full_flat_mlp_ppo.yaml`

### 3.2 Run B（warm startあり / imitation → PPO）
第二候補:
- `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`

### 3.3 rotation evalプリセット
比較実験フェーズ用:
- `configs/stage1_full_flat_mlp_ppo_rotation_eval.yaml`

### 3.4 推奨順
- **最初は Run A（warm startなし）**
- その後に Run B（warm startあり）を同予算で比較する
- rotation eval は最初から必須ではなく、比較フェーズで使う

---

## 4. デバイス設定

### 4.1 推奨の初期値
- `training.device="cuda"`
- `selfplay.inference_device="cpu"`
- `evaluation.inference_device="cpu"`

### 4.2 方針
- learner は GPU 効果が大きいので `cuda` を優先する
- self-play / eval は環境ステップ主体なので、まずは `cpu` を基準にする
- `auto` も使えるが、**最初の実験では明示指定推奨**
- run 後は `summary.json.device_info` と `summary.json.env_info` を確認する

---

## 5. 並列化の設定（重要）

### 5.1 並列化の基本方針
- まずは **単機 multi-process**
- 1 worker = 1 process = 1 thread 相当
- learner は当面 single-process
- **eval を先に並列化して効果確認**
- 次に self-play を並列化する

### 5.2 推奨開始点（10コア20スレッド想定）
- `evaluation.num_workers=8`
- `selfplay.num_workers=1`（まずは eval 並列だけ有効）
- 慣れたら `selfplay.num_workers=8` を試す

### 5.3 worker 数の考え方
- 最初は **8〜10 worker**
- いきなり 20 worker を前提にしない
- まずは 1 / 4 / 8 / 10 で比較する

---

## 6. 実験規模（小 / 中）

### 6.1 動作確認用（小）
- `selfplay.num_matches=5`
- `training.epochs=1`
- `evaluation.num_matches=5`
- `evaluation.num_workers=4`（または1）

### 6.2 比較用（中）
- `selfplay.num_matches=50`
- `training.epochs=2`
- `evaluation.num_matches=20`
- `evaluation.num_workers=8`

### 6.3 補足
- 最初は小さい run で動作確認
- その後、中規模 run で比較する
- いきなり大規模 run に行かない

---

## 7. 実行コマンド例

### 7.1 実験1: PPO直学習（小）+ parallel eval
目的: 並列込みで最後まで回ることを確認する

```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    selfplay.num_matches=5 \
    selfplay.num_workers=1 \
    evaluation.num_matches=5 \
    evaluation.num_workers=4 \
    training.epochs=1 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

### 7.2 実験2: PPO直学習（中）+ parallel eval
目的: 指標のブレを少し減らし、改善傾向を見る

```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    selfplay.num_matches=50 \
    selfplay.num_workers=1 \
    evaluation.num_matches=20 \
    evaluation.num_workers=8 \
    training.epochs=2 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

### 7.3 実験3: warm start比較（小）
目的: imitation → PPO が回ることを確認する

```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    selfplay.num_matches=5 \
    selfplay.num_workers=1 \
    evaluation.num_matches=5 \
    evaluation.num_workers=4 \
    training.epochs=1 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

### 7.4 実験4: warm start比較（中）
目的: PPO直学習との比較用

```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    selfplay.num_matches=50 \
    selfplay.num_workers=1 \
    evaluation.num_matches=20 \
    evaluation.num_workers=8 \
    training.epochs=2 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

### 7.5 実験5: parallel self-play + parallel eval
目的: 並列 self-play の効果確認

```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    selfplay.num_matches=50 \
    selfplay.num_workers=8 \
    evaluation.num_matches=20 \
    evaluation.num_workers=8 \
    training.epochs=2 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

---

## 8. 成功判定（run後に見る順番）

run ディレクトリ（例: `runs/20260304_stage1_.../`）で確認する。

### 8.1 `summary.json`（最優先）
確認点:
- `success == true`
- `phase_status` で各 phase が success
- `device_info`（requested / resolved）
- `env_info`（torch / cuda / gpu名）
- 使用 seed
- worker 数
- 並列時は partial 保存状況や worker seed 情報が追えること

### 8.2 `run.log`
- 失敗時にどのフェーズで落ちたか確認
- 並列 worker 実行のログが混線していないか確認

### 8.3 `notes.md`
- 人間向け概要（実行フェーズ、主要指標など）

### 8.4 `eval/eval_metrics.json`
最低限確認する指標:
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 8.5 `eval/eval_diff.json`
- 学習前後差分がある場合は確認する

### 8.6 `eval/partials/`
- parallel eval 時の partial 出力が存在するか確認する

### 8.7 `selfplay/worker_*/`
- parallel self-play 時に worker ごとに shard が分離されているか確認する

### 8.8 `checkpoints/*.pt`
- learner を走らせた場合に checkpoint が保存されているか確認する

---

## 9. 失敗時の切り分け

### 9.1 CLI 終了コード
- `0`: 成功
- `1`: 失敗

### 9.2 どこで失敗したか
- `summary.json.error`
- `summary.json.phase_status`
- `run.log`

### 9.3 並列特有の失敗例
- eval partial の欠落 / 集約失敗
  - `eval/partials/` を確認
- self-play shard の衝突 / 欠落
  - `selfplay/worker_*/` を確認
- seed 記録不足 / 再現性不安
  - `summary.json` を確認
- worker 数を増やしすぎて遅い
  - worker 数を下げる
  - まず eval だけ並列に戻す

---

## 10. 推奨実験順序

### 10.1 ステップ1
- smoke test
- config validation

### 10.2 ステップ2
- 実験1（PPO直学習・小）

### 10.3 ステップ3
- 実験2（PPO直学習・中）

### 10.4 ステップ4
- 実験3 / 実験4（warm start 比較）

### 10.5 ステップ5
- 実験5（parallel self-play + parallel eval）

---

## 11. ベンチマーク（optional）

### 11.1 eval 並列のスケーリング確認
同じ config で `evaluation.num_workers` を変える。

候補:
- 1
- 2
- 4
- 8
- 10

目的:
- 評価時間の短縮率を見る
- worker 数の最適点を探す

### 11.2 self-play 並列のスケーリング確認
同じ config で `selfplay.num_workers` を変える。

候補:
- 1
- 2
- 4
- 8
- 10

目的:
- データ生成速度の最適点を探す

### 11.3 ベンチマークで残したい項目
- worker 数
- num_matches
- 総実行時間
- 1 match あたり時間
- 1 worker 比の speedup

---

## 12. 実験中に守るルール

- 比較実験では、**比較対象以外の条件は固定**する
- 実験中にコードや config の追加修正が必要そうでも、**勝手に変更しない**
- 追加修正が必要だと判断した場合は、そのまま進めず、まずレポートする
- 最初の比較では **warm startなし** を基準 run にする
- rotation eval は最初の run では必須ではない
- まずは **eval 並列の効果確認**を優先する

---

## 13. Codex に実験を依頼するときの運用ルール

- この RUNBOOK に記載された手順と設定方針に従って実験を進める
- 実験中にコードや config の追加修正が必要だと判断した場合は、勝手に変更せず、まずその旨をレポートする
- 比較実験では、比較対象以外の条件を原則固定する
- 各実験について、論理名（例: Run A, Run B, Benchmark-Eval-8workers）と対応する run ディレクトリを明記する
- 実験完了後は、以下の形式でレポートを作成する

### 13.1 レポートに必ず含める項目
1. 対象実験
   - 実験名
   - run ディレクトリ
   - seed
   - 使用 config
   - override
2. 実行条件
   - `training.device`
   - `selfplay.inference_device`
   - `evaluation.inference_device`
   - `selfplay.num_workers`
   - `evaluation.num_workers`
3. 実行結果
   - success / failure
   - 実行時間
   - 主指標（`avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`）
   - 学習前後差分（可能なら）
4. 補足
   - `run.log` / `summary.json` から分かる警告や注意点
   - partial 出力や worker 出力の異常有無
5. 結論
   - この実験から何が言えるか
   - 次にやるべき実験や修正提案

### 13.2 ベンチマーク実験で追加してほしい項目
- worker 数
- num_matches
- 総実行時間
- 1 match あたり時間
- 1 worker 実行比の speedup

---

## 14. メモ（運用のコツ）

- まずは **PPO直学習 + parallel eval** で動作確認する
- warm start 比較はその次でよい
- worker 数は 8〜10 から試す
- いきなり最大 worker 数にしない
- 評価が速くなったら、次に self-play 並列の効果を見る
- learner は当面 single-process のままでよい

---