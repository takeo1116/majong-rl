# Experiment Report: exp_028

作成日: 2026-05-08  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_028` は、CQ-0285 後に悪化した性能が **terminal semantic loss の絶対スケール低下**で説明できるかを検証する実験である。

CQ-0285 では terminal semantic loss を `sum / weight_sum` に正規化した。これにより gradient balance は改善したが、60-cycle probe では performance が悪化した。

本実験では、CQ-0285 の式は維持したまま、`terminal_loss_coef` だけを上げる。

狙い:

```text
terminal-driven value_trunk shaping が policy performance を支えていたなら、
terminal_loss_coef を戻すことで性能も戻るはず。
```

## 2. 背景

### 2.1 exp026 baseline

`reward.point_delta_scale=0.0001` を入れた `P100 scaled` は、exp026 で 3seed 改善が確認された。

```text
P100 scaled exp026 3seed:
final  2.0817
best   1.9600
best10 2.0422
tail10 2.1203
tail20 2.1292
win    0.2348
deal   0.1835
score  30066.3
```

### 2.2 CQ0285 base coef

CQ-0285 後、base coefficient のまま seed42 60-cycle probe を回すと性能が悪化した。

```text
runs/20260507_stage2a_gradnorm60_cq0285_P100_scaled_seed42_5aaf163a

final  2.295
best   2.165
tail10 2.374
tail20 2.361
```

late gradient ratio:

```text
T/Y ~= 6.75
T/V ~= 7.15
```

### 2.3 exp027 coef sweep

exp027 では `terminal / yaku / value` の coef を同倍率で 10x / 50x / 100x 上げた。

最良は `COEF50x` だったが、exp026 baseline には届かなかった。

```text
COEF50x seed42:
final  2.245
best   2.055
tail10 2.292
tail20 2.259
```

同倍率で上げても `T/Y` はほぼ動かなかったため、terminal-driven shaping 仮説を直接見るには terminal だけを上げる必要があった。

## 3. 実験条件

Base: CQ-0285 適用後の `P100 scaled`。

固定:

- `policy_ratio = 1.0`
- `ppo_mode = "separated"`
- `policy_anchor.enabled = false`
- `reward.point_delta_scale = 0.0001`
- `training.lr = 0.0001`
- `clip_epsilon = 0.15`
- `entropy_coef = 0.0`
- `value_loss_coef = 0.125`
- `yaku_loss_coef = 0.05`
- `multi_cycle.num_cycles = 60`
- `selfplay_matches_per_cycle = 200`
- `gradient_norms.enabled = true`
- `gradient_norms.max_batches_per_epoch = 4`

Sweep:

| label | terminal_loss_coef | seeds | intent |
|---|---:|---|---|
| `TERM30x` | 3.0 | 42 | main 1seed probe |
| `TERM50x` | 5.0 | 42, 43, 44 | stronger terminal restoration; 3seed validation |

## 4. Run 一覧

| label | seed | run_dir |
|---|---:|---|
| TERM30x | 42 | `runs/20260507_stage2a_exp028_terminal30x_seed42_17b712e3` |
| TERM50x | 42 | `runs/20260507_stage2a_exp028_terminal50x_seed42_df9be40a` |
| TERM50x | 43 | `runs/20260507_stage2a_exp028_terminal50x_seed43_bccf983a` |
| TERM50x | 44 | `runs/20260508_stage2a_exp028_terminal50x_seed44_ee04dff5` |

Run maps:

```text
experiments/Stage02_CallUnlock/exp_028/run_map.json
experiments/Stage02_CallUnlock/exp_028/term50x_run_map.json
```

## 5. 主結果

### 5.1 seed42 probe

| condition | final | best | best10 | tail10 | tail20 | final win | final deal | final score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp026 seed42 base | 1.970 | 1.970 | 2.052 | 2.124 | 2.137 | 0.2368 | 0.1675 | 30757.5 |
| CQ0285 base coef | 2.295 | 2.165 | 2.270 | 2.374 | 2.361 | 0.2576 | 0.1842 | 26556.0 |
| exp027 COEF50x | 2.245 | 2.055 | 2.220 | 2.292 | 2.259 | 0.3046 | 0.1822 | 27282.5 |
| TERM30x | 2.155 | 2.030 | 2.104 | 2.162 | 2.167 | 0.2321 | 0.1862 | 28406.5 |
| TERM50x | 2.075 | 1.955 | 2.039 | 2.131 | 2.112 | 0.2085 | 0.1808 | 30150.5 |

`TERM50x` は seed42 で exp026 base とほぼ同等まで回復した。

特に:

```text
best:   1.970 -> 1.955
best10: 2.052 -> 2.039
tail20: 2.137 -> 2.112
```

final は exp026 seed42 より少し悪いが、終盤平均では同等以上。

### 5.2 TERM50x 3seed 平均

| condition | n | final | best | best10 | tail10 | tail20 | win | deal-in | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P100 scaled exp026 | 3 | 2.0817 | 1.9600 | 2.0422 | 2.1203 | 2.1292 | 0.2348 | 0.1835 | 30066.3 |
| TERM50x exp028 | 3 | 2.1417 | 1.9333 | 2.0263 | 2.1095 | 2.1017 | 0.2088 | 0.1851 | 29447.0 |

差分 (`TERM50x - exp026`):

| metric | diff |
|---|---:|
| final avg_rank | +0.0600 |
| best avg_rank | -0.0267 |
| best10 avg_rank | -0.0159 |
| tail10 avg_rank | -0.0108 |
| tail20 avg_rank | -0.0275 |
| win_rate | -0.0260 |
| deal_in_rate | +0.0016 |
| avg_score | -619.3 |

`TERM50x` は final では exp026 より悪いが、best / best10 / tail10 / tail20 では exp026 を上回った。

### 5.3 TERM50x seed 別

| seed | final | best | best10 | tail10 | tail20 | win | deal-in | score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.075 | 1.955 | 2.039 | 2.131 | 2.112 | 0.2085 | 0.1808 | 30150.5 |
| 43 | 2.235 | 1.930 | 2.012 | 2.124 | 2.111 | 0.1999 | 0.1876 | 28571.5 |
| 44 | 2.115 | 1.915 | 2.028 | 2.074 | 2.083 | 0.2179 | 0.1870 | 29619.0 |

全 seed で best は `1.955` 以下、tail20 は `2.112` 以下。終盤平均は安定している。

## 6. Diagnostics

### 6.1 Late PPO diagnostics

Late = last 10 cycles average.

| condition | clip | max_prob | ratio_max | T/Y | T/V | Y/V |
|---|---:|---:|---:|---:|---:|---:|
| CQ0285 base coef seed42 | 0.0577 | 0.9180 | 9.08 | 6.75 | 7.15 | 1.07 |
| exp027 COEF50x seed42 | 0.0611 | 0.9319 | 20.20 | 7.15 | 7.50 | 1.05 |
| TERM30x seed42 | 0.0592 | 0.9058 | 5.42 | 208.58 | 309.75 | 1.49 |
| TERM50x seed42 | 0.0516 | 0.9333 | 10.11 | 487.81 | 303.00 | 0.62 |
| TERM50x 3seed mean | 0.0474 | 0.9338 | 8.45 | 492.64 | 354.43 | 0.72 |

`TERM50x` は terminal gradient ratio を CQ-0285 前に近い水準まで戻した。

一方、`clip_fraction` は下がっており、PPO ratio 的な破綻は見えない。

### 6.2 解釈上の注意

`TERM50x` は final win_rate が exp026 より低い。

```text
exp026:  0.2348
TERM50x: 0.2088
```

それでも tail10 / tail20 / best は改善しているため、単純に和了率を上げる方向ではなく、局面選択や失点抑制を含めた順位性能が改善している可能性がある。

deal-in はほぼ同等。

```text
exp026:  0.1835
TERM50x: 0.1851
```

## 7. 解釈

### 7.1 terminal-driven shaping 仮説は強く支持された

CQ-0285 base coef では terminal gradient ratio が下がり、performance も悪化した。

```text
CQ0285 base:
T/Y ~= 6.75
T/V ~= 7.15
final 2.295
tail10 2.374
```

`TERM50x` では terminal ratio を戻すことで performance も戻った。

```text
TERM50x:
T/Y ~= 492.6
T/V ~= 354.4
final 2.1417
tail10 2.1095
```

これは、今の Stage2a では terminal prediction が単なる補助タスクではなく、`value_trunk` を outcome-sensitive に育てる主信号として機能していることを示す。

### 7.2 exp027 の同倍率 coef sweep とは違う

exp027 の `COEF50x` は value/yaku/terminal を同倍率で上げたため、`T/Y` と `T/V` はほぼ変わらなかった。

```text
COEF50x seed42:
T/Y ~= 7.15
T/V ~= 7.50
tail10 2.292
```

一方、`TERM50x` は terminal だけを上げ、ratio を大きく戻した。

```text
TERM50x seed42:
T/Y ~= 487.8
T/V ~= 303.0
tail10 2.131
```

性能差は大きく、terminal-specific signal が効いていると読める。

### 7.3 CQ-0285 は式として正しいが、coef再設計が必要

CQ-0285 は terminal loss を平均lossスケールに揃える修正だった。gradient balanceとしては自然だが、既存の `terminal_loss_coef=0.1` は CQ-0285 後には小さすぎた。

現状の強い条件は:

```yaml
training.semantic_aux.terminal_loss_coef: 5.0
training.semantic_aux.yaku_loss_coef: 0.05
training.value_loss_coef: 0.125
```

これは見た目には大きいが、CQ-0285 前の `sum` 定義が batch内 player-round group 数で terminal loss を膨らませていたことを考えると、実効scaleを戻す補正として解釈できる。

## 8. 結論

`TERM50x` は 3seed で再現した。

評価:

- `final` は exp026 より悪い
- `best`, `best10`, `tail10`, `tail20` は exp026 より良い
- `deal-in` はほぼ同等
- PPO ratio 的な破綻は見えない
- terminal gradient ratio は CQ-0285 前に近い水準まで戻った

したがって、`TERM50x` は次の強い候補として扱う価値がある。

ただし、設計としてはまだ気持ち悪さが残る。

```text
今のモデルでは value_trunk を terminal が主に育てている。
これは性能上は効いているが、value が主役であるという直感とはズレる。
```

## 9. 次アクション

### 9.1 optimizer parameter group lr split

次は、`terminal_loss_coef=5.0` に頼らず、`value_semantic` 側の learning rate を上げて同じ効果が出るかを試す価値がある。

目的:

```text
terminalを異常に重くするのではなく、
value/outcome representation 側を速く学習させることで性能を戻せるか
```

候補:

```yaml
training.lr: 0.0001
training.lr_groups.enabled: true
training.lr_groups.policy: 0.0001
training.lr_groups.value_semantic: 0.0003 / 0.001 / 0.003
```

最初は CQ-0285 base coef のまま sweep するのが情報量が高い。

```yaml
training.value_loss_coef: 0.125
training.semantic_aux.terminal_loss_coef: 0.1
training.semantic_aux.yaku_loss_coef: 0.05
```

これで性能が戻るなら、`TERM50x` より設計としてきれいな説明になる。

### 9.2 semantic_proj の整理

`semantic_proj` は summary に入っているが、`proj.detach()` され、直接lossもないため dead weight になっている可能性が高い。

これは `TERM50x` のあとに片づける価値がある。

候補:

1. `semantic_proj` を summary から外す
2. `proj` だけ detach を外す
3. `semantic_proj` に明示的な補助lossを与える

ただし、まずは lr split 実験を優先する。
