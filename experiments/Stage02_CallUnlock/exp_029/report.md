# Experiment Report: exp_029

作成日: 2026-05-08  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_029` は、CQ-0286 の optimizer parameter group lr split を使い、`terminal_loss_coef=5.0` に頼らず `value_semantic` 側の学習速度を上げることで、`TERM50x` 相当の性能に近づけるかを確認する 1seed probe である。

`exp_028 TERM50x` は強かったが、terminal loss だけを大きくする設計には違和感が残る。

本実験では、loss coefficient は CQ-0285 base に戻し、optimizer lr だけを分ける。

```text
policy lr: fixed 1e-4
value_semantic lr: 3e-4 / 1e-3 / 3e-3
```

## 2. 背景

### 2.1 CQ0285 base

CQ-0285 後、terminal loss を平均化したまま base coefficient で回すと、gradient balance は改善したが性能は悪化した。

```text
CQ0285 base seed42:
final  2.295
best   2.165
best10 2.248
tail10 2.374
tail20 2.361
```

### 2.2 exp028 TERM50x

`terminal_loss_coef` だけを `0.1 -> 5.0` に上げた `TERM50x` は 3seed で再現した。

```text
TERM50x exp028 3seed:
final  2.1417
best   1.9333
best10 2.0263
tail10 2.1095
tail20 2.1017
```

これは terminal-driven value_trunk shaping が効くことを示す。一方で、`terminal_loss_coef=5.0` は設定として大きい。

### 2.3 CQ0286

CQ-0286 では Stage2a optimizer に parameter group lr split を追加した。

```yaml
training:
  lr: 0.0001
  lr_groups:
    enabled: true
    policy: 0.0001
    value_semantic: 0.001
```

group:

```text
policy:
  discard_trunk / discard_head / optional_trunk / candidate_encoder / optional_scorer

value_semantic:
  value_trunk / value_head / terminal_head / yaku_head / semantic_proj
```

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

| label | policy_lr | value_semantic_lr | multiplier |
|---|---:|---:|---:|
| `VS_LR3x` | `1e-4` | `3e-4` | 3x |
| `VS_LR10x` | `1e-4` | `1e-3` | 10x |
| `VS_LR30x` | `1e-4` | `3e-3` | 30x |

## 4. Run 一覧

| label | run_dir |
|---|---|
| VS_LR3x | `runs/20260508_stage2a_exp029_vs_lr3x_seed42_8f9ae192` |
| VS_LR10x | `runs/20260508_stage2a_exp029_vs_lr10x_seed42_b5f5442a` |
| VS_LR30x | `runs/20260508_stage2a_exp029_vs_lr30x_seed42_09e5e39d` |

Run map:

```text
experiments/Stage02_CallUnlock/exp_029/run_map.json
```

## 5. 主結果

### 5.1 Performance

| condition | final | best | best10 | tail10 | tail20 | final win | final deal | final score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp026 seed42 | 1.970 | 1.970 | 2.052 | 2.124 | 2.137 | 0.2368 | 0.1675 | 30757.5 |
| CQ0285 base | 2.295 | 2.165 | 2.248 | 2.374 | 2.361 | 0.2576 | 0.1842 | 26556.0 |
| TERM50x seed42 | 2.075 | 1.955 | 2.039 | 2.131 | 2.112 | 0.2085 | 0.1808 | 30150.5 |
| VS_LR3x | 2.190 | 2.060 | 2.091 | 2.161 | 2.162 | 0.2517 | 0.1853 | 27875.5 |
| VS_LR10x | 2.180 | 2.025 | 2.065 | 2.153 | 2.143 | 0.2057 | 0.2029 | 29093.0 |
| VS_LR30x | 2.055 | 1.985 | 2.047 | 2.170 | 2.149 | 0.2085 | 0.1824 | 30465.0 |

`VS_LR30x` が本 sweep の最良。

`VS_LR30x` は:

- final は `TERM50x` より良い
- best / best10 は `exp026 seed42` とほぼ同等
- tail10 / tail20 は `TERM50x` より弱い

```text
VS_LR30x:
final  2.055
best   1.985
best10 2.047
tail10 2.170
tail20 2.149
```

### 5.2 Diagnostics

Late = last 10 cycles average.

| condition | clip | max_prob | ratio_max | T/Y | T/V | Y/V | weighted_terminal | weighted_yaku | weighted_value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CQ0285 base | 0.0577 | 0.9180 | 9.08 | 6.75 | 7.15 | 1.07 | 0.1082 | 0.0163 | 0.0159 |
| TERM50x seed42 | 0.0516 | 0.9333 | 10.11 | 487.81 | 303.00 | 0.62 | 7.0523 | 0.0145 | 0.0244 |
| VS_LR3x | 0.0600 | 0.9103 | 13.24 | 7.19 | 6.22 | 0.87 | 0.1067 | 0.0149 | 0.0174 |
| VS_LR10x | 0.0554 | 0.9211 | 5.88 | 7.54 | 2.65 | 0.35 | 0.0950 | 0.0126 | 0.0382 |
| VS_LR30x | 0.0562 | 0.9126 | 6.09 | 11.90 | 1.21 | 0.10 | 0.0831 | 0.0071 | 0.0748 |

`VS_LR30x` では value gradient がかなり強くなっている。

```text
T/V: 1.21
Y/V: 0.10
```

これは `TERM50x` のような terminal支配とはまったく違う状態である。

## 6. 解釈

### 6.1 lr split は有効

`VS_LR3x -> VS_LR10x -> VS_LR30x` で概ね改善傾向がある。

```text
final:
2.190 -> 2.180 -> 2.055

best:
2.060 -> 2.025 -> 1.985

best10:
2.091 -> 2.065 -> 2.047
```

したがって、CQ-0285 後の performance 悪化には、`value_semantic` 側の学習速度不足も関与していると読める。

### 6.2 ただし TERM50x の完全代替ではない

`VS_LR30x` は final / best10 ではかなり良いが、終盤平均は `TERM50x` に届かなかった。

```text
TERM50x tail10/tail20: 2.131 / 2.112
VS_LR30x tail10/tail20: 2.170 / 2.149
```

つまり、単に value_semantic 側を速くするだけでは、terminal-specific shaping を完全には置き換えられない。

### 6.3 TERM50x と VS_LR30x は効き方が違う

`TERM50x`:

```text
terminal gradient が value_semantic trunk を強く支配する
T/Y ~= 488
T/V ~= 303
```

`VS_LR30x`:

```text
value gradient が相対的に強い
T/Y ~= 12
T/V ~= 1.2
```

それでも `VS_LR30x` はかなり強い。これは「terminalだけ」ではなく、value_semantic 側の更新速度自体もボトルネックだったことを示す。

### 6.4 PPO ratio 的な破綻は見えない

`VS_LR30x` でも late clip は `0.056` 程度で、`ratio_max` も `6.09`。

```text
VS_LR30x:
clip      0.0562
ratio_max 6.09
max_prob  0.9126
```

policy lr は固定されているため、global lr を上げるより PPO update は荒れにくいと見てよい。

## 7. 結論

`VS_LR30x` は有望。

ただし、`TERM50x` の完全な代替にはまだ届かない。

整理:

```text
CQ0285 base:
  綺麗だが弱い

TERM50x:
  強いが terminal 依存が大きい

VS_LR30x:
  より綺麗でかなり強いが、tail は TERM50x に届かない
```

したがって、次は hybrid が本命。

```text
VS_LR30x + small terminal restoration
```

## 8. 次アクション

次の 1seed probe 候補:

```yaml
training:
  lr: 0.0001
  lr_groups:
    enabled: true
    policy: 0.0001
    value_semantic: 0.003
  value_loss_coef: 0.125
  semantic_aux:
    terminal_loss_coef: 0.3
    yaku_loss_coef: 0.05
```

もう少し強め:

```yaml
training.semantic_aux.terminal_loss_coef: 1.0
```

目的:

```text
value_semantic 側を速く育てつつ、
terminal-specific shaping を少しだけ戻す。
```

候補:

| label | value_semantic_lr | terminal_loss_coef |
|---|---:|---:|
| `VS30_TERM3x` | `0.003` | `0.3` |
| `VS30_TERM10x` | `0.003` | `1.0` |

この2本が `TERM50x` の tail を超えれば、`terminal_loss_coef=5.0` より説明しやすい本命条件になる。
