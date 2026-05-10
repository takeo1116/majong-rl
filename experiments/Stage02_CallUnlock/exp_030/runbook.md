# Experiment Runbook: exp_030

作成日: 2026-05-08  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_030` の目的は、`exp_029` で有効だった `value_semantic` 側 learning rate split をさらに上げ、どこまで性能改善するか、どこで破綻するかを 1seed probe で確認することである。

`exp_029` では、CQ-0285 base coefficient のまま `value_semantic_lr=3e-3` (`30x`) まで上げても学習は壊れず、性能も大きく改善した。

```text
VS_LR30x seed42:
final  2.055
best   1.985
best10 2.047
tail10 2.170
tail20 2.149
```

ただし `TERM50x` の tail には届かなかった。

```text
TERM50x seed42:
final  2.075
best   1.955
best10 2.039
tail10 2.131
tail20 2.112
```

今回は hybrid に進む前に、`value_semantic_lr` 単独でまだ改善余地があるかを確認する。

## 2. 背景

### 2.1 exp029 の読み

`VS_LR30x` では PPO ratio 的な破綻は見えなかった。

```text
VS_LR30x late:
clip      0.0562
ratio_max 6.09
max_prob  0.9126
```

これは `policy_lr=1e-4` を固定し、`value_semantic_lr` だけを上げる設計が、global lr を上げるより安定しやすいことを示している。

### 2.2 今回見る仮説

仮説:

```text
TERM50x に届いていない理由は、value_semantic_lr がまだ足りないだけかもしれない。
```

もし `VS_LR50x` / `VS_LR100x` が `TERM50x` に並ぶなら、`terminal_loss_coef=5.0` に頼らず、より綺麗な lr split 設計で性能を戻せる。

もし高lrで壊れる、または伸びないなら、`value_semantic` 側の更新速度だけでは不十分で、terminal-specific signal が必要だと判断できる。

## 3. 実験条件

固定:

- seed: `42`
- `policy_ratio = 1.0`
- `ppo_mode = "separated"`
- `policy_anchor.enabled = false`
- `reward.point_delta_scale = 0.0001`
- `training.lr = 0.0001`
- `training.lr_groups.enabled = true`
- `training.lr_groups.policy = 0.0001`
- `clip_epsilon = 0.15`
- `entropy_coef = 0.0`
- `value_loss_coef = 0.125`
- `terminal_loss_coef = 0.1`
- `yaku_loss_coef = 0.05`
- `multi_cycle.num_cycles = 60`
- `selfplay_matches_per_cycle = 200`
- `gradient_norms.enabled = true`
- `gradient_norms.max_batches_per_epoch = 4`

Sweep:

| label | policy_lr | value_semantic_lr | multiplier | intent |
|---|---:|---:|---:|---|
| `VS_LR50x` | `1e-4` | `5e-3` | 50x | natural next point after 30x |
| `VS_LR100x` | `1e-4` | `1e-2` | 100x | high lr candidate |
| `VS_LR300x` | `1e-4` | `3e-2` | 300x | break boundary probe |

## 4. 比較対象

```text
exp026 seed42:
runs/20260503_stage2a_rewardscale_probe_P100_seed42_dd0b0c5d

CQ0285 base seed42:
runs/20260507_stage2a_gradnorm60_cq0285_P100_scaled_seed42_5aaf163a

TERM50x seed42:
runs/20260507_stage2a_exp028_terminal50x_seed42_df9be40a

VS_LR30x seed42:
runs/20260508_stage2a_exp029_vs_lr30x_seed42_09e5e39d
```

## 5. 実行コマンド

driver:

```bash
./.venv/bin/python scripts/local/stage2/exp_030_driver.py
```

単発:

```bash
EXP030_ONLY=VS_LR50x ./.venv/bin/python scripts/local/stage2/exp_030_driver.py
EXP030_ONLY=VS_LR100x ./.venv/bin/python scripts/local/stage2/exp_030_driver.py
EXP030_ONLY=VS_LR300x ./.venv/bin/python scripts/local/stage2/exp_030_driver.py
```

## 6. 見るべき指標

### 6.1 Performance

- final avg_rank
- best avg_rank
- best10
- tail10
- tail20
- final win_rate
- final deal_in_rate
- final avg_score

特に `tail10 / tail20` を `TERM50x` と比較する。

### 6.2 Stability

高lr sweepなので、以下を重視する。

- `clip_fraction`
- `ratio_max`
- `log_ratio_p01 / p99`
- `max_prob`
- `entropy`
- value / terminal / yaku loss の異常値

壊れる場合は、policy lr 固定でも semantic summary 分布が急に動き、policy input が不安定になる可能性がある。

### 6.3 Gradient diagnostics

- `T/Y`, `T/V`, `Y/V`
- `weighted_terminal_loss.value_semantic.mean`
- `weighted_yaku_loss.value_semantic.mean`
- `weighted_value_loss.value_semantic.mean`

lr split は loss coefficient を変えないため、`TERM50x` のような terminal支配にはならないはずである。

## 7. 判定

### 成功

`VS_LR50x` または `VS_LR100x` が `TERM50x seed42` に近づく、または上回る。

目安:

```text
best10 <= 2.04
tail10 <= 2.13
tail20 <= 2.12
```

この場合、最良条件を3seed化する価値がある。

### 部分成功

`VS_LR30x` より改善するが `TERM50x` には届かない。

この場合、lr split は有効だが、terminal-specific shaping も必要と判断する。次は hybrid:

```yaml
value_semantic_lr: best lr
terminal_loss_coef: 0.3 or 1.0
```

### 破綻

`VS_LR100x` / `VS_LR300x` で以下が出る:

- performance 大幅悪化
- clip_fraction 急増
- ratio_max / log_ratio tail 悪化
- max_prob 急上昇
- value/semantic loss 異常

この場合、lr単独の限界が見えたと判断する。

## 8. 次アクション

1. `VS_LR50x/100x` が強ければ、最良条件を 3seed 化
2. `VS_LR30x` 付近が最良なら、`VS_LR30x + terminal_loss_coef 0.3/1.0` を試す
3. 高lrで壊れるなら、`TERM50x` を強い暫定baselineとして扱いつつ、semantic_proj 整理や alpha 正規化へ進む
