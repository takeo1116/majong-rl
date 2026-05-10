# Experiment Report: exp_027

作成日: 2026-05-07  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_027` は、CQ-0285 で terminal semantic loss を player-round 平均に正規化した後、`terminal / yaku / value` の loss coefficient を同倍率で上げる必要があるかを確認する 1seed probe である。

CQ-0285 後の 60-cycle probe では、gradient balance は改善した一方で policy performance が悪化した。

比較対象:

- `exp_026 seed42 P100 scaled`: `runs/20260503_stage2a_rewardscale_probe_P100_seed42_dd0b0c5d`
- `CQ0285 base coef`: `runs/20260507_stage2a_gradnorm60_cq0285_P100_scaled_seed42_5aaf163a`

## 2. 背景

CQ-0285 前は terminal aux が value/semantic trunk を極端に支配していた。

```text
weighted_terminal / weighted_yaku  ~= 128-318
weighted_terminal / weighted_value ~= 238-1344
```

CQ-0285 後は、60-cycle probe で以下まで下がった。

```text
all-cycle average:
weighted_terminal / weighted_yaku  ~= 8.0
weighted_terminal / weighted_value ~= 9.6

late_50_59 average:
weighted_terminal / weighted_yaku  ~= 6.7
weighted_terminal / weighted_value ~= 7.2
```

しかし性能は悪化した。

```text
exp026 seed42 baseline:
final  1.970
best   1.970
tail10 2.124
tail20 2.137

CQ-0285 + base coef:
final  2.295
best   2.165
tail10 2.374
tail20 2.361
```

このため、terminal だけではなく `terminal / yaku / value` 全体の loss signal が弱くなりすぎた可能性を検証した。

## 3. 実験条件

Base: CQ-0285 適用後の `P100 scaled`。

固定:

- seed: `42`
- `reward.point_delta_scale = 0.0001`
- `policy_ratio = 1.0`
- `ppo_mode = "separated"`
- `policy_anchor.enabled = false`
- `training.lr = 0.0001`
- `clip_epsilon = 0.15`
- `entropy_coef = 0.0`
- `multi_cycle.num_cycles = 60`
- `selfplay_matches_per_cycle = 200`
- `gradient_norms.enabled = true`
- `gradient_norms.max_batches_per_epoch = 4`

Sweep:

| label | multiplier | value_loss_coef | terminal_loss_coef | yaku_loss_coef |
|---|---:|---:|---:|---:|
| `COEF10x` | 10x | 1.25 | 1.0 | 0.5 |
| `COEF50x` | 50x | 6.25 | 5.0 | 2.5 |
| `COEF100x` | 100x | 12.5 | 10.0 | 5.0 |

## 4. Run 一覧

| label | run_dir |
|---|---|
| COEF10x | `runs/20260507_stage2a_exp027_cq0285_coef10x_seed42_b48da299` |
| COEF50x | `runs/20260507_stage2a_exp027_cq0285_coef50x_seed42_0b31039d` |
| COEF100x | `runs/20260507_stage2a_exp027_cq0285_coef100x_seed42_0fdab930` |

Driver map:

```text
experiments/Stage02_CallUnlock/exp_027/run_map.json
```

## 5. 主結果

### 5.1 Performance

| condition | final | best | best5 | best10 | tail5 | tail10 | tail20 | final win | final deal | final score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp026 seed42 base | 1.970 | 1.970 | 2.098 | 2.124 | 2.098 | 2.124 | 2.137 | 0.2368 | 0.1675 | 30757.5 |
| CQ0285 base coef | 2.295 | 2.165 | 2.236 | 2.270 | 2.337 | 2.374 | 2.361 | 0.2576 | 0.1842 | 26556.0 |
| COEF10x | 2.515 | 2.155 | 2.239 | 2.275 | 2.451 | 2.472 | 2.388 | 0.2772 | 0.1743 | 25333.0 |
| COEF50x | 2.245 | 2.055 | 2.189 | 2.220 | 2.327 | 2.292 | 2.259 | 0.3046 | 0.1822 | 27282.5 |
| COEF100x | 2.300 | 2.105 | 2.209 | 2.227 | 2.251 | 2.275 | 2.253 | 0.2485 | 0.1887 | 27116.5 |

### 5.2 Late diagnostics

Late = last 10 cycles average.

| condition | entropy | clip | max_prob | log_ratio_p01 | ratio_max | T/Y | T/V | Y/V |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp026 seed42 base | 0.2531 | 0.0591 | 0.8981 | -0.3222 | 8.8905 | - | - | - |
| CQ0285 base coef | 0.2056 | 0.0577 | 0.9180 | -0.3460 | 9.0773 | 6.75 | 7.15 | 1.07 |
| COEF10x | 0.1917 | 0.0605 | 0.9222 | -0.3673 | 6.5208 | 5.96 | 8.23 | 1.39 |
| COEF50x | 0.1675 | 0.0611 | 0.9319 | -0.4150 | 20.2041 | 7.15 | 7.50 | 1.05 |
| COEF100x | 0.1626 | 0.0592 | 0.9356 | -0.4133 | 8.6781 | 7.43 | 7.44 | 1.00 |

### 5.3 Late gradient norm means

`value_semantic` group, last 10 cycles average.

| condition | weighted_terminal | weighted_yaku | weighted_value | terminal_loss | yaku_loss | value_loss |
|---|---:|---:|---:|---:|---:|---:|
| CQ0285 base coef | 0.1082 | 0.0163 | 0.0159 | 0.8536 | 0.1319 | 0.00927 |
| COEF10x | 1.0501 | 0.1775 | 0.1337 | 0.8269 | 0.1119 | 0.00698 |
| COEF50x | 5.9540 | 0.8336 | 0.8153 | 0.8799 | 0.1312 | 0.01042 |
| COEF100x | 11.7656 | 1.5940 | 1.7244 | 0.9042 | 0.1376 | 0.01125 |

## 6. 解釈

### 6.1 係数同倍率上げは部分的に有効

`COEF50x` は `CQ0285 base coef` より明確に改善した。

```text
best:   2.165 -> 2.055
tail10: 2.374 -> 2.292
tail20: 2.361 -> 2.259
final:  2.295 -> 2.245
```

したがって、CQ-0285 後に `terminal / yaku / value` 全体の押す力が弱くなりすぎた、という仮説は部分的に支持される。

### 6.2 ただし exp026 seed42 baseline には戻らない

`COEF50x` は改善したが、`exp026 seed42 base` には届かなかった。

```text
exp026 seed42 base final: 1.970
COEF50x final:           2.245
```

単純に auxiliary/value coefficient を大きくすれば旧性能に戻る、という状況ではない。

### 6.3 COEF10x は弱い、COEF100x はCOEF50xよりやや弱い

`COEF10x` は後半の drift が大きく、tail が悪い。

`COEF100x` は `COEF50x` と近いが、best / final はやや劣る。係数を大きくしすぎても明確な追加改善はない。

今回の範囲では `COEF50x` が最も妥当だった。

### 6.4 PPO ratio 的な破綻は見えない

`COEF50x` / `COEF100x` でも `clip_fraction` は late で `~0.06` 程度であり、明確な PPO ratio explosion は見えない。

一方、係数を上げるほど entropy は低下し、max_prob は上がる。

```text
CQ0285 base coef entropy late: 0.2056
COEF50x entropy late:         0.1675
COEF100x entropy late:        0.1626
```

aux/value を強めると policy がよりdeterministicになる傾向がある。

## 7. 結論

`COEF50x` が本probeの最良条件。

ただし、`exp026 seed42 base` には戻らないため、CQ-0285 の単純な平均化 + coefficient一括増強だけでは不十分。

現時点の見立て:

```text
CQ0285前:
terminalが強すぎるが、その強いterminal signalが性能に効いていた

CQ0285後:
gradient balanceは改善したが、terminal/value/semantic全体の押しが弱くなり性能悪化

COEF50x:
押しは戻ったが、旧terminal支配の性能までは戻らない
```

## 8. 次アクション

次は `COEF50x` をベースに、terminalだけを追加で少し強める probe が妥当。

候補:

```yaml
training.value_loss_coef: 6.25
training.semantic_aux.yaku_loss_coef: 2.5
training.semantic_aux.terminal_loss_coef: 10.0
```

または少し強め:

```yaml
training.value_loss_coef: 6.25
training.semantic_aux.yaku_loss_coef: 2.5
training.semantic_aux.terminal_loss_coef: 15.0
```

目的は、`COEF50x` の value/yaku/value_semantic 全体の押しを維持しつつ、terminal signal を旧条件に少し寄せること。

ただし、terminal-only 強化に進む前に、semantic eval で `COEF50x` が terminal/yaku head をどう学んでいるかを確認する価値がある。
