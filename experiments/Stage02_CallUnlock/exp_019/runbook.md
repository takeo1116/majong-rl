# Experiment Runbook: exp_019

作成日: 2026-04-10  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_015/report.md`
- `experiments/Stage02_CallUnlock/exp_017/report.md`
- `experiments/Stage02_CallUnlock/exp_018/report.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

## 1. 背景

`exp_015` では

- `A2_semaux_light_vhalf_tenpaifix_prnorm`

が practical baseline になった。

その後 `exp_017` では、`tile_presence_flags` と `value_hidden_dims` の 2x2 比較を行い、

- `yakuflags on + wide`

が

- `yakuflags on + narrow`

より大きく改善することが確認された。

特に diagnostics では、

- `Tanyao` の `mean_p`
- `hit@0.2`

が明確に持ち上がっており、特徴量アイデア自体は有望だと判断した。

一方で、policy だけを見ると `exp_017 on_wide` は

- `exp_015 A2` baseline より少し悪い

という位置に留まっていた。

ただしこの比較はほぼ **1 seed** に依存しており、

- `exp_015 A2`: seed42
- `exp_017 on_wide`: seed42

だけで practical な採用判断を下している状態だった。

このため、次は

- baseline 側も
- `on_wide` 側も

**同じ seed セットで追加 2 seed** 回して、
結論をより頑健にする。

## 2. 今回の問い

`exp_019` で答えたい問いは次の 2 点である。

1. `exp_017 on_wide` は seed を増やしても `exp_015 A2` より一段劣るのか
2. それとも seed を揃えて見ると、baseline と同等、あるいは上回る可能性があるのか

言い換えると今回は、

- 新しい構造をさらに足す前に
- **既存の有望候補 (`on_wide`) を multi-seed で sanity-check する**

実験である。

## 3. 実験方針

今回は seed42 の既存結果をアンカーとして使い、
**seed43 / seed44 のみ新規に回す**。

比較する 2 条件:

1. `exp_015` 相当 baseline
2. `exp_017 on_wide` 相当条件

これを同じ seed で paired に比較する。

### 3.1 baseline 側

- `A2_semaux_light_vhalf_tenpaifix_prnorm`
- `tile_presence_flags = false`
- `value_hidden_dims = [128, 64]`

### 3.2 on_wide 側

- `A2_semaux_light_vhalf_tenpaifix_prnorm_on_widevalue`
- `tile_presence_flags = true`
- `value_hidden_dims = [256, 128]`
- `tile_presence_flags_semantic_only = false`

重要:

- `on_wide` は `exp_018 semantic_only` ではなく、
  あくまで `exp_017` で有望だった **all-trunks raw input** 条件を再検証する

## 4. 既存アンカー

seed42 の既存結果をアンカーとして使う。

### baseline anchor

- `exp_015 A2 seed42`
- `runs/20260408_stage2a_exp015_A2_semaux_light_vhalf_tenpaifix_prnorm_seed42_2b43d332`

### on_wide anchor

- `exp_017 on_wide seed42`
- `runs/20260409_stage2a_exp017_A2_semaux_light_vhalf_tenpaifix_prnorm_on_widevalue_seed42_999ef54a`

## 5. 新規実行条件

### 5.1 baseline seed43

- `A2_semaux_light_vhalf_tenpaifix_prnorm_seed43`

### 5.2 baseline seed44

- `A2_semaux_light_vhalf_tenpaifix_prnorm_seed44`

### 5.3 on_wide seed43

- `A2_semaux_light_vhalf_tenpaifix_prnorm_on_widevalue_seed43`

### 5.4 on_wide seed44

- `A2_semaux_light_vhalf_tenpaifix_prnorm_on_widevalue_seed44`

## 6. 共通固定条件

全条件共通:

- `training.value_loss_coef = 0.125`
- `training.policy_anchor.coef = 0.75`
- `training.multi_cycle.num_cycles = 20`
- semantic aux 有効
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`

前提として保持するもの:

- direct hint branch (`CQ-0265`)
- terminal 5-class (`CQ-0266`)
- `self_tenpai_flag / remaining_draws_norm` (`CQ-0267`)
- terminal player-round normalization (`CQ-0268`)
- `deal_in` risk diagnostics (`CQ-0269`)
- illegal discard snapshot fix (`CQ-0271`)
- `tile_presence_flags` on/off flag (`CQ-0272`)
- `tile_presence_flags_semantic_only` 実装 (`CQ-0273`)
  ただし今回は **false** のまま使う

## 7. 必須観測

### 7.1 通常性能

各 seed について:

- imitation eval
- final
- best cycle
- tail-5 average

比較単位:

- seed42 / 43 / 44 ごとの paired 比較
- 3 seed 平均

### 7.2 PPO 安定性

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- retain

### 7.3 diagnostics

今回の主目的は multi-seed の policy 比較なので、
diagnostics はまず最小限でよい。

ただし、もし `on_wide` が 3 seed で baseline にかなり近づくか上回るなら、
その時点で改めて

- `Tanyao`
- yaku macro F1
- terminal

の詳細 diagnostics を取りに行く。

## 8. 成功判定

今回の成功判定は次の通り。

1. `on_wide` の 3 seed 平均が baseline と同等以上
2. paired seed 比較で `on_wide` が 3 回のうち 2 回以上 baseline に迫る
3. 少なくとも「seed42 だけの偶然ではない」と言える

逆に、

- 追加 2 seed を入れても `on_wide` が一貫して baseline 未満

なら、`on_wide` は「有望だが practical baseline 更新にはまだ足りない」と判断しやすくなる。

## 9. 実行コマンド

全条件:

```bash
./.venv/bin/python scripts/local/stage2/exp_019_driver.py
```

1 条件だけ:

```bash
EXP019_ONLY=A2_semaux_light_vhalf_tenpaifix_prnorm_on_widevalue_seed43 \
  ./.venv/bin/python scripts/local/stage2/exp_019_driver.py
```

## 10. 期待アウトプット

- `experiments/Stage02_CallUnlock/exp_019/run_map.json`
- `experiments/Stage02_CallUnlock/exp_019/driver_logs/*.log`
- 新規 4 run の `summary.json`
- seed42 を含めた paired 3-seed 比較表

