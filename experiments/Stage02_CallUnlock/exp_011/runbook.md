# Experiment Runbook: exp_011

作成日: 2026-04-02  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_010/report.md`
- `experiments/Stage02_CallUnlock/exp_010/runbook.md`
- `experiments/Stage02_CallUnlock/exp_009_bugfix/bug_report.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

## 1. 背景

`exp_010` では、open-hand bugfix 後に semantic auxiliary 比較を取り直した。
その結果、

- imitation / selfplay の `win_called` support 自体は復活した
- ただし `A1_semaux_default` の優位はかなり弱まった
- semantic diagnostics では `win_called` は support を持つのに、`terminal_head` は `win_called` をほぼ学べていなかった
- `terminal_head` は `ron_bystander` 偏重、`yaku_head` は `Riichi` 偏重が強かった

ここまでで、teacher supply 側の致命 bug は概ね解消した一方、**full observation の base feature 自体にまだ learning を不利にする欠損が残っていた**ことが分かった。

具体的には、`exp_010` 時点の full observation には次の問題があった。

- 4 家 block は seat-fixed で、`self / shimo / toimen / kamicha` の意味が index に固定されていなかった
- full path の `riichi` は 0 埋めだった
- full path に `menzen` が明示特徴として入っていなかった

これらはその後の修正で対処済みである。

## 2. 今回の再実験前に入った修正

今回の `exp_011` は、`exp_010` と**同じ学習条件**を維持したまま、以下の特徴量修正を入れた上で再実行する。

### 2.1 actor-relative full observation (`CQ-0263`)

full observation の 4 家 block を actor-relative に回転し、

- `0 = self`
- `1 = shimo`
- `2 = toimen`
- `3 = kamicha`

で固定した。

対象:
- hands
- discards
- melds
- scores
- dealer scalar も actor-relative 化

これにより、`discard / optional / semantic` の全 branch が共有する base feature で、feature index と麻雀上の意味が一致するようになった。

### 2.2 full observation への `riichi` / `menzen` 追加 (`CQ-0264`)

`FullObservation` と `FlatFeatureEncoder` を拡張し、full path に actor-relative の

- `riichi` block (4 dim)
- `menzen` block (4 dim)

を追加した。

これにより、少なくとも base feature 上で

- `win_menzen / win_called`
- `Riichi` 系役
- `optional` の面前維持判断

に必要な基本状態が明示的に入るようになった。

### 2.3 open-hand semantics は引き続き修正済み

`exp_010` と同様に、今回も以下の修正済み前提を維持する。

- rule-based teacher の open-hand shanten / acceptance
- evaluator baseline seat の open-hand discard
- encoder の `current_shanten` / `shanten_hint` / `discard_ukeire_hint`
- Python fallback
- opponent `danger_mask`

したがって、`exp_011` は

- open-hand bugfix 済み
- actor-relative 化済み
- `riichi` / `menzen` 実値化済み

の feature / teacher / evaluator semantics 上での比較になる。

## 3. 問い

今回の runbook で答えたい問いは次の 5 つである。

1. actor-relative + `riichi/menzen` 追加後でも、`C0 / A1 / A2` は stable に回るか
2. `exp_010` で弱かった `A1_semaux_default` は、feature 修正後に再び優位を取り戻すか
3. `terminal_head` の `win_called` 学習は改善するか
4. `yaku_head` の `Riichi` 偏重は緩和するか
5. semantic trunk の改善が `discard / optional / final eval` に反映されるか

## 4. 基本方針

今回は **学習条件は `exp_010` と同一** にする。

つまり比較条件は

- `C0_r3_control_featurefix`
- `A1_semaux_default_featurefix`
- `A2_semaux_light_featurefix`

と同じ構成を使い、差分は **encoder / observation 側の修正だけ** に限定する。

この方針により、`exp_010 -> exp_011` の差はほぼ

- actor-relative full observation
- full path `riichi` / `menzen`

の影響として読める。

## 5. 比較条件

### C0: feature-fix 後 R3 control

- semantic aux 無効
- actor-relative + `riichi/menzen` 追加済み full observation を使う control

### A1: feature-fix 後 semantic aux default

- `model.semantic_aux.enabled = true`
- `training.semantic_aux.enabled = true`
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.2`
- `training.semantic_aux.yaku_loss_coef = 0.1`

### A2: feature-fix 後 semantic aux light

- semantic aux 有効
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`

## 6. 共通固定条件

全条件共通:

- `core_minimal`
- `full observation`
- mixed PPO
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.policy_anchor.coef = 0.75`
- `training.lr = 1e-4`
- `training.clip_epsilon = 0.15`
- `training.max_grad_norm = 0.50`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`
- seed `42`

encoder 側では、`exp_010` と同様に次を有効とする。

- `shanten_hint`
- `discard_ukeire_hint`
- `current_shanten`
- `shape_hint`
- `turn_context`

無効のままとするもの:

- `opponent_current_shanten`
- `opponent_tenpai_flag`
- `danger_mask`

## 7. 今回の追加観測

通常の eval 指標に加えて、今回は feature 修正の効果を見るために次を必ず確認する。

### 7.1 `win_called` support

各条件について、少なくとも次を確認する。

- imitation data に `win_called` が存在するか
- selfplay shard に `win_called` が存在するか
- `call` decision row の中にも `win_called` が存在するか

### 7.2 semantic diagnostics

`A1` については少なくとも

- imitation checkpoint
- final checkpoint

の 2 点で semantic diagnostics を取る。

特に見たいもの:

- terminal accuracy
- `win_menzen` / `win_called` support と recall
- terminal label-conditioned confidence
- `win_called` の `mean_p`, `p50`, `p90`, `top1_hit_rate`, `top3_hit_rate`
- `top1_confusers`
- yaku micro / macro F1
- `Riichi`, `Yakuhai`, `Tanyao` の positive-conditioned confidence

### 7.3 安定性

以下が `exp_010` と同程度に健全かを見る。

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- learner loss の暴れ方
- 20 cycle 完走可否

### 7.4 性能

主に次を比較する。

- `imitation_eval`
- `cycle_00`
- `final`
- `tail-5 average`

特に見たい差分:

- `final - imitation_eval`
- `cycle_00 - imitation_eval`
- `A1 - C0`
- `A2 - C0`

## 8. 成功判定

今回の再実験は、少なくとも次を満たせば成功とみなす。

1. 全条件が stable に完走する
2. `win_called` support が imitation / selfplay の両方で維持される
3. `A1` の semantic diagnostics で、`exp_010` より `win_called` の confidence か rank に改善が見える
4. `A1` または `A2` が `C0` より `imitation_eval` からの改善量で上回る

特に今回は、単純な final 1 点だけではなく、**semantic head が rare/open 系 class を少しでも掴み始めるか**を重視する。

## 9. 典型的な解釈

### ケース 1: `A1` が改善し、`win_called` confidence も上がる

解釈:

- これまでの弱さは feature 欠損の影響が大きかった
- semantic aux はなお有望
- 次は `A1` multi-seed に進める

### ケース 2: `A1` の semantic diagnostics は改善するが final は弱い

解釈:

- semantic trunk 自体は改善している
- ただし policy 利用か reward 側がまだ別問題
- 次は branch attribution や summary 利用を疑う

### ケース 3: `C0` も `A1` も同程度に改善

解釈:

- actor-relative + `riichi/menzen` は semantic aux 以前に共通基盤改善として効いた
- その上で semantic aux の追加価値はまだ限定的

### ケース 4: semantic diagnostics も性能もほぼ変わらない

解釈:

- feature 欠損は主因ではなかった
- 次は class imbalance / loss weighting / sampling を優先して検討する

## 10. 想定管理ファイル

- `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `experiments/Stage02_CallUnlock/exp_011/run_map.json`
- `experiments/Stage02_CallUnlock/exp_011/report.md`

想定ラベル:

- `C0_r3_control_featurefix`
- `A1_semaux_default_featurefix`
- `A2_semaux_light_featurefix`

## 11. 実行前確認

実行前に少なくとも次を確認する。

- `.venv` の通常 import で最新 `_mahjong_core` が使われていること
- actor-relative encoder + `riichi/menzen` 追加後のテストが通っていること
- Stage2 selfplay smoke で `win_called` が出ること

ここが崩れていると、`exp_011` の解釈が再び不安定になる。

## 12. 実行コマンド

全条件を順番に回す:

```bash
./.venv/bin/python scripts/local/stage2/exp_011_driver.py
```

1 条件だけ試す:

```bash
EXP011_ONLY=A1_semaux_default_featurefix ./.venv/bin/python scripts/local/stage2/exp_011_driver.py
```
