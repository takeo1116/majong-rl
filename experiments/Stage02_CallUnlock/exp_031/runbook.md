# Experiment Runbook: exp_031

作成日: 2026-05-08  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_031` の目的は、`exp_030 VS_LR100x` を土台に、terminal / yaku / value の gradient scale を terminal 基準でオーダーレベルに揃えたとき、性能を維持できるかを確認することである。

`exp_030 VS_LR100x` は `value_semantic_lr=0.01` で、terminal coef を大きくしなくても `TERM50x` に近い tail を出した。

```text
VS_LR100x seed42:
final  2.140
best   1.980
best10 2.063
tail10 2.128
tail20 2.129
```

ただし late gradient ratio はまだ terminal が強い。

```text
VS_LR100x late:
T/Y ~= 5.43
T/V ~= 2.02
Y/V ~= 0.40
```

今回は terminal を基準として、yaku と value の係数を丸めて引き上げる。

```text
terminal_loss_coef: 0.10  # keep
 yaku_loss_coef:    0.05 -> 0.25
 value_loss_coef:   0.125 -> 0.25
```

完全一致ではなく、学習中の揺れを前提に「桁を揃える」ことを狙う。

## 2. 背景

### 2.1 TERM50x

`terminal_loss_coef=5.0` は3seedで強かった。

```text
TERM50x exp028 3seed:
best   1.9333
best10 2.0263
tail10 2.1095
tail20 2.1017
```

ただし、terminal signal への依存が大きい。

### 2.2 VS_LR100x

`terminal_loss_coef=0.1` のまま、`value_semantic_lr=0.01` にした `VS_LR100x` は seed42 で `TERM50x` に近い tail を出した。

```text
TERM50x seed42:
tail10 2.131
tail20 2.112

VS_LR100x seed42:
tail10 2.128
tail20 2.129
```

これは、terminal coef を極端に大きくしなくても、value_semantic 側の lr を上げればかなり学習できることを示す。

### 2.3 今回の問い

本当に綺麗な説明を目指すなら、terminal だけが強い状態を残したくない。

ただし、terminal を弱めると性能が落ちる可能性があるため、今回は「一番強い terminal を基準に、弱い yaku/value を上げる」方針を取る。

```text
T/Y ~= 1
T/V ~= 1
Y/V ~= 1
```

完全一致は不要で、オーダーレベルで揃えばよい。

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
- `training.lr_groups.value_semantic = 0.01`
- `clip_epsilon = 0.15`
- `entropy_coef = 0.0`
- `value_loss_coef = 0.25`
- `terminal_loss_coef = 0.10`
- `yaku_loss_coef = 0.25`
- `multi_cycle.num_cycles = 60`
- `selfplay_matches_per_cycle = 200`
- `gradient_norms.enabled = true`
- `gradient_norms.max_batches_per_epoch = 4`

Condition:

| label | policy_lr | value_semantic_lr | terminal_loss_coef | yaku_loss_coef | value_loss_coef | intent |
|---|---:|---:|---:|---:|---:|---|
| `VS100_BALANCED_TBASE` | `1e-4` | `1e-2` | `0.10` | `0.25` | `0.25` | terminal基準でyaku/valueを上げる |

## 4. 比較対象

```text
VS_LR100x:
runs/20260508_stage2a_exp030_vs_lr100x_seed42_f1c3da06

TERM50x:
runs/20260507_stage2a_exp028_terminal50x_seed42_df9be40a

CQ0285 base:
runs/20260507_stage2a_gradnorm60_cq0285_P100_scaled_seed42_5aaf163a
```

## 5. 実行コマンド

driver:

```bash
./.venv/bin/python scripts/local/stage2/exp_031_driver.py
```

単発指定:

```bash
EXP031_ONLY=VS100_BALANCED_TBASE ./.venv/bin/python scripts/local/stage2/exp_031_driver.py
```

validate-only:

```bash
EXP031_VALIDATE_ONLY=1 ./.venv/bin/python scripts/local/stage2/exp_031_driver.py
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

特に `VS_LR100x` と `TERM50x` に対して、tail10 / tail20 が保てるかを見る。

### 6.2 Gradient balance

最重要:

- `T/Y`
- `T/V`
- `Y/V`
- `weighted_terminal_loss.value_semantic.mean`
- `weighted_yaku_loss.value_semantic.mean`
- `weighted_value_loss.value_semantic.mean`

成功条件:

```text
T/Y, T/V, Y/V が 0.3-3.0 程度に収まり、
performance が VS_LR100x と大きく崩れない。
```

### 6.3 Stability

- `clip_fraction`
- `ratio_max`
- `log_ratio_p01 / p99`
- `max_prob`
- `value_loss`
- `terminal_loss`
- `yaku_loss`

## 7. 判定

### 成功

gradient ratio が揃い、性能が `VS_LR100x` に近い。

この場合:

```text
terminal強化ではなく、value_semantic高lr + balanced aux で説明できる。
```

次は `VS100_BALANCED_TBASE` を 3seed 化する価値がある。

### 部分成功

gradient ratio は綺麗になるが、性能がやや落ちる。

この場合:

```text
信号バランスは改善したが、terminal signal を少し強めに残す必要がある。
```

次は `yaku/value` の上げ幅を少し戻す、または `terminal_loss_coef=0.2-0.3` を試す。

### 失敗

性能が大きく落ちる。

この場合:

```text
terminal signal は相対的に強めに残す必要がある。
```

次は `VS_LR100x` または `TERM50x` 系を本命にする。
