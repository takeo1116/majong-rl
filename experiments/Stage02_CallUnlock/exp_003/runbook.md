# Experiment Runbook: exp_003

作成日: 2026-03-30  
Stage: `Stage02_CallUnlock`  
参照:
- `experiments/Stage02_CallUnlock/exp_001/runbook.md`
- `experiments/Stage02_CallUnlock/exp_001/report.md`
- `experiments/Stage02_CallUnlock/exp_002/runbook.md`
- `experiments/Stage02_CallUnlock/exp_002/report.md`
- `configs/stage2a_core_minimal_separated_baseline.yaml`

## 1. 背景

`exp_001` では、当初 Stage02a の feature 比較として

1. A `core_minimal`
2. B `stage1style_context`
3. C `stage1style_context_plus_danger`

を比較する予定だった。

しかし実際には、

- imitation / selfplay / eval の correctness 問題
- parallel shard 読み込み不整合
- worker thread cap 欠落
- Stage02a learner の大きな速度問題
- mixed PPO の不安定性

が先に露出し、feature 比較に入る前に PPO 安定化が主課題となった。

そのため `exp_002` で、

- A `core_minimal` 固定
- `policy_ratio=0.25`
- `baseline_sample_weight=0.5`
- `policy_anchor(reference="imitation_fixed", coef=0.5)`
- `num_cycles=20`

を共通条件として、

- `ppo_mode="mixed"`
- `ppo_mode="separated"`

を比較した。

結果として、

- `mixed`: 後半 cycle で discard drift が再発し不採用
- `separated`: 20 cycle を通して安定し、採用候補

という結論が得られた。

また、Stage02a learner の throughput も

- tensor path
- discard best-set ベクトル化
- optional candidate 前計算
- direct parquet path

により、比較実験を回せる水準まで改善した。

したがって `exp_003` では、**`separated` を安定 baseline として、`exp_001` の本来の問いだった A/B/C feature 比較を再開する。**

## 2. 問い

Stage02a を `separated` PPO 条件で安定に回したとき、次の 3 条件のうちどれを Stage02 の暫定 feature baseline に採用すべきか。

1. A `core_minimal`
2. B `stage1style_context`
3. C `stage1style_context_plus_danger`

具体的には次を確認する。

1. A/B/C のうち、どれが最も素直に学習して eval 成績が良いか
2. `opponent_current_shanten` / `opponent_tenpai_flag` が Stage02a でも有効か
3. `danger_mask` を Stage02a に入れる価値があるか
4. `separated` 条件でも、feature 差として意味のある差が出るか

## 3. この実験の位置づけ

この `exp_003` は、Stage02 の **最初の正式 feature 比較実験** である。

- `exp_001` の本来の問いを、安定化済み PPO 条件でやり直す
- PPO 安定性そのものを比較する実験ではない
- `exp_002` で決めた `separated` 条件を共通土台とする
- ここで決めた feature baseline を、以後の Stage02 実験の参照点にする

## 4. 共通条件

### PPO / learner baseline

`exp_002` の結論に従い、全条件で次を固定する。

- `training.rule_mix_learner.ppo_mode = "separated"`
- `training.rule_mix.policy_ratio = 0.25`
- `training.rule_mix_learner.baseline_sample_weight = 0.5`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.policy_anchor.coef = 0.5`
- `training.multi_cycle.num_cycles = 20`

### ベース config

共通設定は以下の config にまとめた。

- `configs/stage2a_core_minimal_separated_baseline.yaml`

この config には、A `core_minimal` の feature set と、
`exp_002` で安定した `separated` baseline 条件が入っている。

## 5. 比較条件

### A `core_minimal`

ベース config をそのまま使用する。

有効:

- `feature_encoder.shanten_hint.enabled = true`
- `feature_encoder.discard_ukeire_hint.enabled = true`
- `feature_encoder.current_shanten.enabled = true`
- `feature_encoder.shape_hint.enabled = true`
- `feature_encoder.turn_context.enabled = true`

無効:

- `feature_encoder.opponent_current_shanten.enabled = false`
- `feature_encoder.opponent_tenpai_flag.enabled = false`
- `feature_encoder.danger_mask.enabled = false`

狙い:

- Stage02a の最小有望条件
- call 解放段階で必要最小限の情報だけでどこまで行けるかを見る

### B `stage1style_context`

A に追加:

- `feature_encoder.opponent_current_shanten.enabled = true`
- `feature_encoder.opponent_tenpai_flag.enabled = true`

据え置き:

- `feature_encoder.danger_mask.enabled = false`

狙い:

- Stage01 で補助的に効いた opponent 文脈が Stage02a でも有効かを見る

### C `stage1style_context_plus_danger`

B に追加:

- `feature_encoder.danger_mask.enabled = true`

狙い:

- Stage01 current best の本命だった `danger_mask` を Stage02a に持ち込む価値があるかを見る
- ただし Stage02a では optional / call 系の分布も変わるため、Stage01 と全く同じ意味ではないことに注意する

## 6. 既存 run の流用

`AGENTS.md` の流用ルールに従い、`exp_003` では A `core_minimal` を新規実行せず、`exp_002` の既存 run を control として流用する。

流用対象:

- `exp_002` の A2 `separated` control

流用理由:

- `configs/stage2a_core_minimal_separated_baseline.yaml` と同じ feature 条件
- `separated`
- `policy_ratio=0.25`
- `baseline_sample_weight=0.5`
- `anchor(imitation_fixed, coef=0.5)`
- `num_cycles=20`

が一致しており、`exp_003` の A control としてそのまま使えるため。

今回新しく回すのは B/C のみとする。

## 7. 実行順

1. A `core_minimal` は `exp_002` の既存 run を流用
2. B `stage1style_context`
3. C `stage1style_context_plus_danger`

理由:

- A は既に同条件 run が存在し、再実行コストに対する追加情報が小さい
- B/C の差分だけ新たに取得すれば、`exp_003` の問いには十分答えられる
- run 名と対応表は `run_map.json` と driver で明示的に管理する

## 8. 実行コマンド

実行前提:

- C++ 側 (`bindings/`, `src/engine/`, `src/rules/`, `src/core/`) を触った場合は、先に `.venv` を rebuild する

```bash
./.venv/bin/python -m pip install -e . --no-build-isolation
```

### A `core_minimal`

新規実行は行わず、以下を control として流用する。

- `exp_002` の A2 `separated` control

### B `stage1style_context`

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_separated_baseline.yaml \
  --base-dir runs \
  --override \
  'experiment.name="stage2a_exp003_B_stage1style_context_seed42"' \
  'experiment.global_seed=42' \
  'feature_encoder.opponent_current_shanten.enabled=true' \
  'feature_encoder.opponent_tenpai_flag.enabled=true'
```

### C `stage1style_context_plus_danger`

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_separated_baseline.yaml \
  --base-dir runs \
  --override \
  'experiment.name="stage2a_exp003_C_stage1style_context_plus_danger_seed42"' \
  'experiment.global_seed=42' \
  'feature_encoder.opponent_current_shanten.enabled=true' \
  'feature_encoder.opponent_tenpai_flag.enabled=true' \
  'feature_encoder.danger_mask.enabled=true'
```

### B/C driver

```bash
./.venv/bin/python scripts/local/stage2/exp_003_driver.py
```

この driver は

- A を `exp_002` 既存 run から流用
- B/C を順番に実行
- 実行ログと対応表は `experiments/Stage02_CallUnlock/exp_003/run_map.json` を中心に管理する

する。

## 9. 観測ポイント

### imitation

各 chunk で次を確認する。

- `data_generation_sec`
- `train_sec`
- `diagnostics_sec`
- `chunk_total_sec`

目的:

- feature 差で learner throughput が極端に悪化しないかを見る

### multi-cycle learner

各 cycle で次を確認する。

- learner loss
- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- `anchor_kl_optional`

目的:

- `separated` 条件でも feature によって不安定化しないか確認する

### eval

各 cycle と final で次を確認する。

- `avg_rank`
- `win_rate`

目的:

- feature 差が最終性能にどう出るかを見る

## 10. 成功判定

最低条件:

1. A/B/C がすべて hard crash なしで完走する
2. `ratio_mean` / `clip_fraction` / `anchor_kl_discard` が `exp_002 separated` と同程度の安定域に収まる
3. eval が後半で一方的に崩れない

比較条件としてほしいもの:

1. A/B/C の final `avg_rank` に差が出る
2. `win_rate` にも概ね整合した差が出る
3. 少なくとも 1 条件は A より明確に良い、または A が最も素直であると判断できる

## 11. 次アクション判定

### A が最良

- 当面の Stage02 feature baseline は A `core_minimal`
- B/C の opponent 文脈や danger は、現段階では見送る

### B が最良

- Stage02 でも `stage1style_context` が有効
- opponent 文脈を baseline に採用する

### C が最良

- `danger_mask` は Stage02a でも価値がある
- Stage01 current best の思想を Stage02 にも持ち込める可能性が高い

### 差が不明瞭

- seed を増やす
- あるいは `num_cycles` / `num_matches` を増やして再比較する

## 12. 補足

この `exp_003` は、まず **feature 比較を前に進めること** を優先する。

- `mixed` parity の再挑戦
- baseline 打牌を PPO に再び混ぜる試み

は、別テーマとして後で扱う。

現時点では、`exp_002` で確立した `separated` baseline の上で、
Stage02 の特徴量比較を進めることが最も自然である。
