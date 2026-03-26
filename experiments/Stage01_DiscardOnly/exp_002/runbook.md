# EXPERIMENT_STAGE2_RUNBOOK.md（Runbook 2）

最終更新: 2026-03-05  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel / PPO  
目的:  
1. `evaluation.num_workers` の実用的な最適点を決める  
2. PPO直学習の多seed挙動を確認する  
3. warm start（imitation → PPO）の有効性を多seedで比較する

---

## 0. この Runbook の位置づけ

Runbook 1 で確認できたこと:

- Stage 1 実験基盤は安定して動作する
- parallel self-play / parallel eval は成立している
- warm start の導線は成立している
- ただし性能差はまだ不安定で、評価分散が大きい
- parallel self-play の速度効果は確認できた

この Runbook 2 では、  
**「動く」から「比較できる」へ進む** ことを目的とする。

---

## 1. 前提

- 実験は CLI で起動する
- self-play / evaluation は multi-process を利用可能
- multi-seed バッチ実行が利用可能
- 各 seed は **別 run_dir** に保存される
- batch 単位で集約レポートを出力できる

---

## 2. セットアップ

### 2.1 依存導入
```bash
pip install -e ".[dev]"
```

### 2.2 smoke test
```bash
python3 -m pytest tests/python/ -m smoke -v
```

### 2.3 config validation
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

## 3. 共通方針

### 3.1 基本デバイス
最初の比較では以下を固定する。

- `training.device="cuda"`
- `selfplay.inference_device="cpu"`
- `evaluation.inference_device="cpu"`

### 3.2 比較で固定するもの
比較対象以外の条件は固定する。

- Observation: `full`
- Encoder: `flat`
- Model: `mlp`
- Algorithm: `ppo`
- reward / legal mask / baseline など既存設定も固定
- seed群をそろえる

### 3.3 多seed比較の基本
- まず **5 seed**
- 必要なら 10 seed へ拡張
- 各 seed は **独立run**
- batch 集約レポートを見る

---

## 4. 実験A: parallel eval の worker 数ベンチ

### 4.1 目的
- `evaluation.num_workers` の実用的な最適点を決める
- 20戦評価がどこまで短縮できるか確認する
- worker 数を増やしたときに headroom がまだあるかを見る

### 4.2 比較対象
以下を比較する。

- `evaluation.num_workers=1`
- `evaluation.num_workers=4`
- `evaluation.num_workers=8`
- `evaluation.num_workers=12`
- `evaluation.num_workers=16`
- `evaluation.num_workers=20`

### 4.3 条件
- config: `configs/stage1_full_flat_mlp_ppo.yaml`
- seed: 42
- `selfplay.num_matches=50`
- `selfplay.num_workers=1`
- `training.epochs=2`
- `evaluation.num_matches=20`
- それ以外は固定

### 4.4 実行コマンド例

#### worker=1
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    selfplay.num_matches=50 \
    selfplay.num_workers=1 \
    evaluation.num_matches=20 \
    evaluation.num_workers=1 \
    training.epochs=2 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

#### worker=4
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    selfplay.num_matches=50 \
    selfplay.num_workers=1 \
    evaluation.num_matches=20 \
    evaluation.num_workers=4 \
    training.epochs=2 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

#### worker=8
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

#### worker=12
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    selfplay.num_matches=50 \
    selfplay.num_workers=1 \
    evaluation.num_matches=20 \
    evaluation.num_workers=12 \
    training.epochs=2 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

#### worker=16
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    selfplay.num_matches=50 \
    selfplay.num_workers=1 \
    evaluation.num_matches=20 \
    evaluation.num_workers=16 \
    training.epochs=2 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

#### worker=20
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    selfplay.num_matches=50 \
    selfplay.num_workers=1 \
    evaluation.num_matches=20 \
    evaluation.num_workers=20 \
    training.epochs=2 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

### 4.5 確認したい項目
- run 全体時間
- eval_before / eval の時間
- 1 match あたり時間
- 1 worker 比 speedup
- 指標の整合性（worker 数で評価値が不自然に壊れないか）

### 4.6 結論の出し方
- 最も速い worker 数を標準候補にする
- ただし speedup が頭打ちなら、少し少ない worker 数を標準にしてよい
- 以後の Runbook 2 実験では、その worker 数を採用する

---

## 5. 実験B: PPO直学習の多seed比較

### 5.1 目的
- 現行 Stage 1 PPO が平均的に改善するか確認する
- 学習分散（seed差）を測る

### 5.2 推奨条件
- config: `configs/stage1_full_flat_mlp_ppo.yaml`
- seed: 5本
- 推奨 seed 例: `42,43,44,45,46`
- `selfplay.num_matches=50`
- `selfplay.num_workers=8`（Runbook 1 で効果確認済み）
- `evaluation.num_matches=20`
- `evaluation.num_workers=<実験Aで決めた最適値>`
- `training.epochs=2`

### 5.3 実行コマンド例
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    selfplay.num_matches=50 \
    selfplay.num_workers=8 \
    evaluation.num_matches=20 \
    evaluation.num_workers=8 \
    training.epochs=2 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

※ `evaluation.num_workers=8` は例。実験Aの結果で置き換える。

### 5.4 確認したい項目
batch 集約レポートで以下を見る。

- 平均 `avg_rank`
- 平均 `avg_score`
- 平均 `win_rate`
- 平均 `deal_in_rate`
- 各指標の標準偏差
- 成功率
- 学習前後差分の平均

### 5.5 期待
- 改善が平均として見えるか
- それとも seed によって大きく揺れるだけか
- warm start 比較に進む価値があるか

---

## 6. 実験C: warm start の多seed比較

### 6.1 目的
- imitation → PPO が、PPO直学習に対して有利か確認する
- その性能差が imitation の高コストに見合うかを確認する

### 6.2 推奨条件
- config:
  - `configs/stage1_full_flat_mlp_ppo.yaml`
  - `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- 同じ seed 群
- 同じ `selfplay.num_matches`
- 同じ `evaluation.num_matches`
- 同じ `training.epochs`
- 同じ worker 数
- 同じ device 設定

### 6.3 実行コマンド例（PPO）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    selfplay.num_matches=50 \
    selfplay.num_workers=8 \
    evaluation.num_matches=20 \
    evaluation.num_workers=8 \
    training.epochs=2 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

### 6.4 実行コマンド例（imitation → PPO）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    selfplay.num_matches=50 \
    selfplay.num_workers=8 \
    evaluation.num_matches=20 \
    evaluation.num_workers=8 \
    training.epochs=2 \
    training.device=\"cuda\" \
    selfplay.inference_device=\"cpu\" \
    evaluation.inference_device=\"cpu\"
```

### 6.5 見るべき項目
- 主指標4種の平均・分散
- run 全体時間
- imitation フェーズ時間
- 改善幅あたりのコスト
- 失敗率 / ばらつき

### 6.6 判断基準
- 性能が同程度なら、まずは **PPO直学習を優先**
- warm start が明確に平均改善しないなら、後回しでもよい
- warm start が性能向上しても、時間コストが大きすぎるなら再設計を検討する

---

## 7. run 後に確認するもの

### 7.1 単独 run
- `summary.json`
- `run.log`
- `notes.md`
- `eval/eval_metrics.json`
- `eval/eval_diff.json`
- `checkpoints/*.pt`

### 7.2 multi-seed batch
- `batch_summary.json`
- `batch_table.csv` または `batch_table.jsonl`
- seed ごとの run_dir
- 失敗 run があればその `summary.json` / `run.log`

---

## 8. レポートに必ず含める項目

### 8.1 parallel eval ベンチ
- worker 数
- run_dir
- 総時間
- eval_before 時間
- eval 時間
- 1 match あたり時間
- 1 worker 比 speedup
- 指標の整合性コメント

### 8.2 多seed PPO比較
- batch_dir
- seed 群
- 共通 config
- worker 設定
- 主指標の平均 / 標準偏差
- 学習前後差分の平均
- 失敗 run の有無

### 8.3 warm start 比較
- PPO batch_dir
- imitation→PPO batch_dir
- 共通 seed 群
- 主指標比較
- imitation フェーズ時間
- 総時間比較
- コスト対効果のコメント

---

## 9. この Runbook でまだやらないこと

この Runbook 2 では、以下はまだ対象外とする。

- モデル大型化比較
- CNN 導入
- PartialObservation 比較
- teacher/student 蒸留
- learner 分散化
- 複数サーバ分散実行

---

## 10. 推奨順序

1. **実験A: parallel eval ベンチ**
2. worker 数の標準値を決める
3. **実験B: PPO直学習の多seed比較**
4. **実験C: warm start の多seed比較**

---

## 11. メモ

- まずは `evaluation.num_workers` の標準値を決める
- その後の多seed実験では worker 数を固定する
- warm start は現時点で高コストなので、PPO直学習との差が見えないなら後回しにしてよい
- 実験中に仕様変更や追加実装が必要と判断した場合は、勝手に変更せずレポートする

---