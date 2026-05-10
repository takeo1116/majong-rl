# Experiment Runbook: exp_032

作成日: 2026-05-08  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_032` の目的は、`VS_LR100x` を土台に policy 側 learning rate を上げたとき、学習速度・性能・安定性が改善するかを 1seed probe で確認することである。

これまでの結果では、`value_semantic_lr` を `100x` まで上げても学習は大きく崩壊せず、`TERM50x` に近い性能が得られた。

```text
VS_LR100x seed42:
final  2.140
best   1.980
best10 2.063
tail10 2.128
tail20 2.130
```

一方、`VS_LR300x` も best/best10 は強く、`value_semantic_lr` をさらに上げても壊れるだけではなかった。

```text
VS_LR300x seed42:
final  2.155
best   1.935
best10 2.007
tail10 2.154
tail20 2.141
```

ここから、現状は `value_semantic` 側だけでなく policy 側 lr が律速になっている可能性がある。

```text
value_semantic は速く動いているが、policy が 1e-4 のままなので action distribution への反映が遅いかもしれない。
```

## 2. 仮説

PPO の policy 更新は value/advantage を参照して action probability を動かす。  
`value_semantic_lr` を大きくして value/semantic 表現が改善しても、policy lr が低すぎると、その改善が行動方策へ十分に反映されない可能性がある。

仮説:

```text
policy_lr を 3x-10x 程度に上げると、VS_LR100x の value/semantic 改善に policy がより速く追従し、best/tail が改善する可能性がある。
```

ただし policy lr は action distribution を直接動かすため、value_semantic lr より危険である。  
崩壊する場合は、`clip_fraction`, `ratio_max`, `max_prob`, `entropy` に出るはずである。

## 3. 実験条件

固定:

- seed: `42`
- `policy_ratio = 1.0`
- `ppo_mode = "separated"`
- `policy_anchor.enabled = false`
- `reward.point_delta_scale = 0.0001`
- `training.lr_groups.enabled = true`
- `training.lr_groups.value_semantic = 0.01`
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

| label | policy_lr | value_semantic_lr | intent |
|---|---:|---:|---|
| `P3x_VS100x` | `0.0003` | `0.01` | conservative useful jump |
| `P5x_VS100x` | `0.0005` | `0.01` | main aggressive probe |
| `P10x_VS100x` | `0.0010` | `0.01` | break-boundary probe |

`P2x` は `P3x` と近いため今回は省略する。  
`P5x` で明確に壊れた場合、`P10x` は実行せず、必要なら後で `P2x` を単体で試す。

## 4. 実行方針

Driver は `P3x -> P5x -> P10x` の順で実行する。

途中停止は手動判断とする。  
ログを見て明確に壊れていると判断した場合は、driver を止めて残り条件は実行しない。

## 5. 実行コマンド

全条件を順番に実行:

```bash
./.venv/bin/python scripts/local/stage2/exp_032_driver.py
```

単発実行:

```bash
EXP032_ONLY=P3x_VS100x ./.venv/bin/python scripts/local/stage2/exp_032_driver.py
EXP032_ONLY=P5x_VS100x ./.venv/bin/python scripts/local/stage2/exp_032_driver.py
EXP032_ONLY=P10x_VS100x ./.venv/bin/python scripts/local/stage2/exp_032_driver.py
```

validate-only:

```bash
EXP032_VALIDATE_ONLY=1 ./.venv/bin/python scripts/local/stage2/exp_032_driver.py
```

## 6. 監視ポイント

途中で見るべき値:

- eval avg_rank
- `clip_fraction`
- `ratio_max`
- `log_ratio_p01 / p99`
- `max_prob_mean`
- entropy

危険サイン:

```text
clip_fraction が大きく跳ねる
ratio_max が極端に大きくなる
max_prob が急上昇する
entropy が急低下する
eval avg_rank が連続して悪化する
```

## 7. 比較対象

```text
VS_LR100x:
runs/20260508_stage2a_exp030_vs_lr100x_seed42_f1c3da06

VS_LR300x:
runs/20260508_stage2a_exp030_vs_lr300x_seed42_d8ecf0c9

TERM50x seed42:
runs/20260507_stage2a_exp028_terminal50x_seed42_df9be40a
```

## 8. 判定

### 成功

`P3x` または `P5x` で、`VS_LR100x` より best/tail が改善し、PPO diagnostics も破綻しない。

目安:

```text
best10 <= 2.04
tail10 <= 2.12
tail20 <= 2.12
clip_fraction <= 0.15 程度
```

この場合、最良条件を 3seed 化する。

### 部分成功

best は改善するが tail が悪化する。

この場合、policy lr を上げると探索・短期改善は強くなるが、安定性は落ちると判断する。  
次は中間の `P2x` または entropy/target_kl 系を検討する。

### 失敗

`P3x` から明確に壊れる。

この場合、policy lr は既に十分高く、`VS_LR100x` のまま 3seed 化する方がよい。
