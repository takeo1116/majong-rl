# Experiment Report: exp_032

作成日: 2026-05-09  
Stage: `Stage02_CallUnlock`

## Summary

`exp_032` では、`VS_LR100x` を土台に policy 側 learning rate を上げ、policy lr が律速になっているかを調べた。

結論:

- `P5x_VS100x seed42` は非常に強く、policy lr を上げる価値があることを示した。
- しかし `P5x_VS100x` の3seed平均は seed44 の悪化で弱く、現時点で本命確定にはできない。
- `P10x_VS100x` は明確に collapse 寄りで、policy lr の上限を超えている。
- `P5x_VS300x` は伸びず、`value_semantic_lr=300x` と `policy_lr=5x` の組み合わせは少なくとも seed42 では不採用。
- 次は `P3x/P4x/P5x` の中間を詰めるか、`P5x` に entropy/target_kl など安定化を組み合わせるのが自然。

## Background

`exp_030` では、`value_semantic_lr=0.01` (`VS_LR100x`) が `TERM50x` に近い tail を出した。

```text
VS_LR100x seed42:
final  2.140
best   1.980
best10 2.063
tail10 2.128
tail20 2.130
```

一方、`VS_LR300x` は best/best10 は強いが tail は弱かった。

```text
VS_LR300x seed42:
final  2.155
best   1.935
best10 2.006
tail10 2.154
tail20 2.141
```

この結果から、`value_semantic` 側だけを速くしても、policy 側 lr が低いと action distribution への反映が律速になっている可能性があった。

そこで `exp_032` では、`value_semantic_lr=0.01` を固定し、policy lr を上げる probe を行った。

## Conditions

共通条件:

- `policy_ratio = 1.0`
- `ppo_mode = "separated"`
- `policy_anchor.enabled = false`
- `reward.point_delta_scale = 0.0001`
- `training.lr_groups.enabled = true`
- `training.lr_groups.value_semantic = 0.01` unless noted
- `clip_epsilon = 0.15`
- `entropy_coef = 0.0`
- `value_loss_coef = 0.125`
- `terminal_loss_coef = 0.1`
- `yaku_loss_coef = 0.05`
- `multi_cycle.num_cycles = 60`
- `selfplay_matches_per_cycle = 200`
- `gradient_norms.enabled = true`

Initial sweep:

| label | seed | policy_lr | value_semantic_lr | run |
|---|---:|---:|---:|---|
| `P3x_VS100x` | 42 | 0.0003 | 0.01 | `runs/20260508_stage2a_exp032_p3x_vs100x_seed42_829db64f` |
| `P5x_VS100x` | 42 | 0.0005 | 0.01 | `runs/20260509_stage2a_exp032_p5x_vs100x_seed42_3e85cde2` |
| `P10x_VS100x` | 42 | 0.0010 | 0.01 | `runs/20260509_stage2a_exp032_p10x_vs100x_seed42_310de194` |

Follow-up:

| label | seed | policy_lr | value_semantic_lr | run |
|---|---:|---:|---:|---|
| `P5x_VS100x` | 43 | 0.0005 | 0.01 | `runs/20260509_stage2a_exp032_p5x_vs100x_seed43_a9c04555` |
| `P5x_VS100x` | 44 | 0.0005 | 0.01 | `runs/20260509_stage2a_exp032_p5x_vs100x_seed44_2083784d` |
| `P5x_VS300x` | 42 | 0.0005 | 0.03 | `runs/20260509_stage2a_exp032_p5x_vs300x_seed42_7149fc19` |

## Performance

Lower avg_rank is better.

| condition | final | best | best_cycle | best10 | tail10 | tail20 | final win | final deal-in | final avg_score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exp026 P100 scaled seed42` | 1.970 | 1.970 | 59 | 2.052 | 2.124 | 2.137 | 0.2368 | 0.1675 | 30757.5 |
| `TERM50x seed42` | 2.075 | 1.955 | 28 | 2.039 | 2.131 | 2.112 | 0.2085 | 0.1808 | 30150.5 |
| `VS_LR100x seed42` | 2.140 | 1.980 | 51 | 2.062 | 2.128 | 2.130 | 0.2147 | 0.2050 | 29369.5 |
| `VS_LR300x seed42` | 2.155 | 1.935 | 49 | 2.006 | 2.154 | 2.141 | 0.2060 | 0.1915 | 29146.0 |
| `P3x_VS100x seed42` | 2.155 | 1.955 | 57 | 2.081 | 2.147 | 2.162 | 0.2272 | 0.1949 | 29485.0 |
| `P5x_VS100x seed42` | 2.105 | 1.885 | 45 | 2.044 | 2.098 | 2.105 | 0.2116 | 0.1914 | 29309.0 |
| `P5x_VS100x seed43` | 2.160 | 2.050 | 52 | 2.082 | 2.164 | 2.159 | 0.2106 | 0.1837 | 29055.0 |
| `P5x_VS100x seed44` | 2.340 | 2.195 | 30 | 2.224 | 2.293 | 2.280 | 0.2274 | 0.1927 | 26378.5 |
| `P10x_VS100x seed42` | 2.215 | 2.110 | 52 | 2.200 | 2.252 | 2.269 | 0.2291 | 0.1994 | 28651.5 |
| `P5x_VS300x seed42` | 2.245 | 2.115 | 36 | 2.163 | 2.245 | 2.237 | 0.1847 | 0.2047 | 27783.0 |

## P5x_VS100x 3seed Aggregate

| metric | mean | std | seed42 | seed43 | seed44 |
|---|---:|---:|---:|---:|---:|
| final | 2.2017 | 0.1229 | 2.1050 | 2.1600 | 2.3400 |
| best | 2.0433 | 0.1551 | 1.8850 | 2.0500 | 2.1950 |
| best10 | 2.1165 | 0.0950 | 2.0440 | 2.0815 | 2.2240 |
| tail10 | 2.1852 | 0.0991 | 2.0980 | 2.1645 | 2.2930 |
| tail20 | 2.1811 | 0.0894 | 2.1050 | 2.1587 | 2.2795 |
| final win | 0.2165 | 0.0094 | 0.2116 | 0.2106 | 0.2274 |
| final deal-in | 0.1892 | 0.0049 | 0.1914 | 0.1837 | 0.1927 |
| final avg_score | 28247.5 | 1623.6 | 29309.0 | 29055.0 | 26378.5 |

読み:

- seed42 は非常に強い。
- seed43 は `VS_LR100x seed42` に近いが、明確な改善とは言いにくい。
- seed44 はかなり悪い。
- 3seed平均では `P5x_VS100x` は本命確定できない。

## PPO Diagnostics

late cycle の代表値。

| condition | clip_fraction | max_prob_mean | entropy | ratio_max | log_ratio_p01 | log_ratio_p99 |
|---|---:|---:|---:|---:|---:|---:|
| `VS_LR100x seed42` | 0.0572 | 0.9100 | 0.2145 | 16.02 | -0.321 | 0.243 |
| `P3x_VS100x seed42` | 0.0775 | 0.9307 | 0.1653 | 10.02 | -0.510 | 0.351 |
| `P5x_VS100x seed42` | 0.0781 | 0.9423 | 0.1401 | 15.30 | -0.525 | 0.410 |
| `P5x_VS100x seed43` | 0.0760 | 0.9574 | 0.1039 | 27.03 | -0.761 | 0.408 |
| `P5x_VS100x seed44` | 0.0729 | 0.9384 | 0.1525 | 6.21 | -0.498 | 0.335 |
| `P10x_VS100x seed42` | 0.0530 | 0.9823 | 0.0429 | 5399.11 | -1.327 | 0.317 |
| `P5x_VS300x seed42` | 0.0720 | 0.9335 | 0.1619 | 6.91 | -0.451 | 0.327 |

読み:

- `P5x` は `VS_LR100x` より policy が強く動いている。
- `clip_fraction` は 0.07-0.08 程度で、clipping に張り付きすぎてはいない。
- `P10x` は entropy が `0.0429`, max_prob が `0.9823`, ratio_max が `5399` で、明確に collapse 寄り。
- `P5x_VS300x` は diagnostics 上は極端に壊れていないが、performance が弱い。

## Interpretation

### 1. policy lr は確かに性能に効く

`P5x_VS100x seed42` は、`VS_LR100x seed42` より全体的に改善した。

```text
VS_LR100x seed42:
best   1.980
best10 2.062
tail10 2.128
tail20 2.130

P5x_VS100x seed42:
best   1.885
best10 2.044
tail10 2.098
tail20 2.105
```

このため、policy lr が律速になっていたという仮説はかなり妥当である。

### 2. ただし P5x は seed 安定性が弱い

3seed 化すると、seed44 が大きく悪化した。

```text
P5x_VS100x 3seed:
tail10 mean 2.1852
tail20 mean 2.1811
```

これは本命としては弱い。  
少なくとも「P5x が常に強い」とは言えない。

### 3. P10x は上げすぎ

`P10x` は avg_rank も悪く、diagnostics も collapse 寄りである。

```text
entropy   0.0429
max_prob  0.9823
ratio_max 5399
```

policy lr の上限は `5x` と `10x` の間にありそうで、`10x` は明確に過剰。

### 4. P5x_VS300x は不採用

`P5x_VS300x` は、`VS_LR300x` の best の強さを tail に反映する狙いだったが、結果は弱い。

```text
P5x_VS300x seed42:
final  2.245
best   2.115
best10 2.163
tail10 2.245
tail20 2.237
```

少なくともこの構成では、value_semantic lr を `300x` まで上げる利点は見えない。

## Decision

`P5x_VS100x` は上振れ力があるが、現時点では本命確定ではない。

次の候補:

1. `P3x/P4x_VS100x` 周辺を追加で見る
2. `P5x_VS100x + entropy_coef` または `target_kl` で安定化する
3. `VS_LR100x` を保守的な本命として3seed化する

現時点の実験的読みは以下。

```text
policy lr を上げる価値はある。
ただし P5x は強すぎる seed があり、seed 安定性が足りない。
P10x は上げすぎ。
次は P3x-P5x の中間、または P5x に安定化を入れるのが自然。
```
