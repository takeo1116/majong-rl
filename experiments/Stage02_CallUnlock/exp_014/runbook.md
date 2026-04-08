# Experiment Runbook: exp_014

作成日: 2026-04-08  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_012/report.md`
- `experiments/Stage02_CallUnlock/exp_013/report.md`
- `experiments/Stage02_CallUnlock/exp_013/runbook.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

## 1. 背景

`exp_013` では、次の構造整理を反映した上で `C0 / A1 / A2` を比較した。

- `CQ-0265`: `shanten_hint` / `discard_ukeire_hint` を discard direct hint に戻す
- `CQ-0266`: terminal semantic を 5-class に整理する

その結果、

- `win_called` は旧 8-class terminal よりかなり素直に学べるようになった
- 一方で `draw_tenpai` は依然としてほぼ学べていない
- 実用基準としては `A2_semaux_light_vhalf_structfix` が最も安定

という状況になった。

その後、`draw_tenpai` 仮説を直接検証するために `CQ-0267` を実装した。

## 2. 今回の再実験前に入った修正

### 2.1 `self_tenpai_flag` の追加 (`CQ-0267`)

共有 encoder feature に `self_tenpai_flag` を追加した。

- `current_shanten == 0` のとき `1.0`
- それ以外は `0.0`

`current_shanten` は残したまま、テンパイかどうかを direct な 1bit で与える。

これは特に

- `draw_tenpai`
- 終盤の押し引き
- 形式聴牌判断

に効くことを期待する。

### 2.2 `remaining_draws_norm` の追加 (`CQ-0267`)

観測に `remaining_draws` を追加し、encoder では `remaining_draws_norm` を共有特徴として追加した。

- `remaining_draws / 70.0`
- 終盤ほど値が小さくなる

これにより、従来の

- `turn_number`
- `turn_context`

だけでは弱かった「残り局面長」の情報を direct に渡せるようになった。

### 2.3 partial path の self 基準修正

`CQ-0267` の仕上げとして、partial path の `meld_count` を self 基準に修正した。

これにより、

- `current_shanten`
- `self_tenpai_flag`
- shanten/ukeire 系 hint

が observer != 0 の partial sample でも self 基準で整合する。

## 3. 今回の問い

今回の `exp_014` で答えたい問いは次の 4 つである。

1. `CQ-0267` により `draw_tenpai` が以前より学べるようになるか
2. その改善は `C0` にも出るか、それとも `semantic_aux` ありの `A2` で主に出るか
3. `A2_semaux_light_vhalf_structfix` は、`tenpaifix` 後も実用基準として維持できるか
4. 必要なら `A1` も追加し、`draw_tenpai` に対して default semantic aux が有利か不利かを確認できるか

## 4. 実験方針

今回は `exp_013` の現基準をそのまま維持し、差分を `CQ-0267` の追加特徴だけに絞る。

固定するもの:

- `training.value_loss_coef = 0.125`
- latest direct hint branch (`CQ-0265`)
- terminal 5-class (`CQ-0266`)
- seed `42`

主比較は **`C0` と `A2` の 2 条件**とする。

理由:

- `A2` は `exp_013` 時点で最も stable
- `draw_tenpai` 改善を観測しやすい
- `C0` を入れることで、単なる特徴量追加か、semantic aux がその特徴をうまく使えているかを切り分けられる

必要なら `A1` を 3 条件目として追加する。

## 5. 比較条件

### C0: control + vhalf + tenpaifix

- semantic aux 無効
- `training.value_loss_coef = 0.125`
- 最新の
  - actor-relative full observation
  - full path `riichi` / `menzen`
  - Stage2a discard direct hints
  - terminal 5-class
  - `self_tenpai_flag`
  - `remaining_draws_norm`
  を使う control

### A2: semantic aux light + vhalf + tenpaifix

- semantic aux 有効
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `training.value_loss_coef = 0.125`

### A1: semantic aux default + vhalf + tenpaifix（任意追加）

- semantic aux 有効
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.2`
- `training.semantic_aux.yaku_loss_coef = 0.1`
- `training.value_loss_coef = 0.125`

これは default では回さず、必要なときだけ追加する。

## 6. 共通固定条件

全条件共通:

- `core_minimal`
- `full observation`
- mixed PPO
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.policy_anchor.coef = 0.75`
- `training.value_loss_coef = 0.125`
- `training.lr = 1e-4`
- `training.clip_epsilon = 0.15`
- `training.max_grad_norm = 0.50`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`
- seed `42`

encoder / model 側では、次の構造を前提にする。

- actor-relative full observation
- full path `riichi`
- full path `menzen`
- Stage2a discard direct hints (`shanten_hint`, `discard_ukeire_hint`)
- terminal semantic 5-class
- `self_tenpai_flag`
- `remaining_draws_norm`

## 7. 必須観測

### 7.1 通常性能

- `imitation_eval`
- `final`
- `best cycle`
- `tail-5 average`

特に見る差分:

- `final - imitation_eval`
- `A2 - C0`
- 必要なら `A1 - A2`

### 7.2 PPO 安定性

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- learner loss の暴れ方
- 20 cycle 完走可否

### 7.3 semantic diagnostics

少なくとも `A2` について、以下を取る。

- `checkpoint_imitation.pt`
- `best cycle checkpoint`
- `checkpoint_learner.pt`

見るもの:

- terminal accuracy
- `draw_tenpai` / `win_called` / `deal_in` の support と recall
- label-conditioned confidence
- relevant subset の平均予測分布

### 7.4 `draw_tenpai` 向け subset 診断

今回の本命はここである。

少なくとも次を見たい。

- `actual_draw_tenpai`
- その中でも
  - `self_tenpai_flag = 1`
  - `remaining_draws_norm` が低い終盤局面

比較相手:

- `actual_other_non_dealin`
- ただし同じく
  - self tenpai
  - 終盤

見たいもの:

- `p(draw_tenpai)`
- `p(win_called)`
- `p(other_non_dealin)`
- top1 / top3

問いは、**終盤テンパイ局面で `draw_tenpai` を `other_non_dealin` より高く置けるか** である。

## 8. 成功判定

今回の再実験は、少なくとも次のいずれかを満たせば前進とみなす。

1. `A2` の policy 性能が `exp_013` 比で維持または改善する
2. `draw_tenpai mean_p` が前回より上がる
3. `draw_tenpai top1_hit` が `0` から動く
4. relevant subset で `p(draw_tenpai)` が `other_non_dealin` より明確に高くなる

## 9. 典型的な解釈

### ケース 1: `A2` で `draw_tenpai` が改善

解釈:

- `self_tenpai_flag` / `remaining_draws_norm` は効いている
- 次は multi-seed か、必要なら `deal_in` の改善へ進む

### ケース 2: policy は維持されるが `draw_tenpai` は動かない

解釈:

- 特徴量追加だけでは不十分
- 次は sampling / weighting / anchor を優先する

### ケース 3: `C0` だけ改善し `A2` は改善しない

解釈:

- semantic aux 側の loss 設計や保持の問題が残っている
- 特徴量自体は悪くないが、使い方に問題がある

## 10. 実行コマンド

既定の 2 条件 (`C0 + A2`):

```bash
./.venv/bin/python scripts/local/stage2/exp_014_driver.py
```

`A1` も含めて 3 条件:

```bash
EXP014_INCLUDE_A1=1 ./.venv/bin/python scripts/local/stage2/exp_014_driver.py
```

1 条件だけ:

```bash
EXP014_ONLY=A2_semaux_light_vhalf_tenpaifix ./.venv/bin/python scripts/local/stage2/exp_014_driver.py
```
