# Experiment Report: exp_031

作成日: 2026-05-08  
Stage: `Stage02_CallUnlock`

## Summary

`exp_031` では、`exp_030 VS_LR100x` を土台に、terminal / yaku / value の gradient scale を terminal 基準でオーダーレベルに揃えた。

結論:

- gradient balance はかなり狙い通り改善した。
- しかし性能は `VS_LR100x` より明確に悪化した。
- 「terminal / yaku / value の信号を揃えること」自体は、少なくとも現構成では性能改善につながらなかった。
- terminal が相対的に強い状態は、単なる不均衡ではなく、今の shared value_semantic trunk にとって有効な shaping signal になっている可能性が高い。

## Background

`exp_030 VS_LR100x` は、`value_semantic_lr=0.01` によって `TERM50x` に近い tail を出した。

```text
VS_LR100x seed42:
final  2.140
best   1.980
best10 2.063
tail10 2.128
tail20 2.130
```

ただし late gradient ratio はまだ terminal が強かった。

```text
VS_LR100x late:
T/Y ~= 5.36
T/V ~= 2.91
Y/V ~= 0.54
```

そこで `exp_031` では terminal を基準にし、yaku/value 側の coefficient を上げた。

```yaml
training:
  value_loss_coef: 0.25
  semantic_aux:
    terminal_loss_coef: 0.10
    yaku_loss_coef: 0.25
  lr_groups:
    enabled: true
    policy: 0.0001
    value_semantic: 0.01
```

## Conditions

Run:

```text
runs/20260508_stage2a_exp031_vs100_balanced_tbase_seed42_1dfe4fcb
```

Fixed:

- seed: `42`
- `policy_ratio = 1.0`
- `ppo_mode = "separated"`
- `policy_anchor.enabled = false`
- `reward.point_delta_scale = 0.0001`
- `training.lr_groups.enabled = true`
- `training.lr_groups.policy = 0.0001`
- `training.lr_groups.value_semantic = 0.01`
- `clip_epsilon = 0.15`
- `entropy_coef = 0.0`
- `multi_cycle.num_cycles = 60`
- `selfplay_matches_per_cycle = 200`
- `gradient_norms.enabled = true`

Changed from `VS_LR100x`:

| key | VS_LR100x | exp031 |
|---|---:|---:|
| `terminal_loss_coef` | 0.10 | 0.10 |
| `yaku_loss_coef` | 0.05 | 0.25 |
| `value_loss_coef` | 0.125 | 0.25 |

## Performance

Lower avg_rank is better.

| condition | final | best | best10 | tail10 | tail20 | final win | final deal-in | final avg_score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `exp026 P100 scaled seed42` | 1.970 | 1.970 | 2.052 | 2.124 | 2.137 | 0.2368 | 0.1675 | 30757.5 |
| `TERM50x seed42` | 2.075 | 1.955 | 2.039 | 2.131 | 2.112 | 0.2085 | 0.1808 | 30150.5 |
| `VS_LR100x` | 2.140 | 1.980 | 2.062 | 2.128 | 2.130 | 0.2147 | 0.2050 | 29369.5 |
| `VS_LR300x` | 2.155 | 1.935 | 2.006 | 2.154 | 2.141 | 0.2060 | 0.1915 | 29146.0 |
| `exp031 balanced` | 2.125 | 2.040 | 2.125 | 2.208 | 2.206 | 0.2402 | 0.1909 | 29363.5 |

読み:

- `final=2.125` だけを見ると悪くないが、`tail10=2.208`, `tail20=2.206` は明確に悪い。
- `best=2.040`, `best10=2.125` で、良い checkpoint の質も `VS_LR100x` / `TERM50x` より落ちている。
- cycle 51 に `avg_rank=2.04` は出ているが、一貫した改善にはなっていない。

## Gradient Diagnostics

late cycle の代表値。

| condition | T/Y | T/V | Y/V | weighted_terminal | weighted_yaku | weighted_value |
|---|---:|---:|---:|---:|---:|---:|
| `VS_LR100x` | 5.36 | 2.91 | 0.54 | 0.0599 | 0.0112 | 0.0206 |
| `exp031 balanced` | 2.07 | 1.08 | 0.52 | 0.0808 | 0.0391 | 0.0746 |

読み:

- `T/V` は `1.08` まで揃った。
- `T/Y` も `5.36 -> 2.07` まで改善した。
- `Y/V` は `0.52` で、完全一致ではないが同じオーダーに入っている。
- したがって、gradient balance の狙い自体は概ね成功している。

PPO stability:

| condition | clip_fraction | max_prob_mean | ratio_max | entropy |
|---|---:|---:|---:|---:|
| `VS_LR100x` | 0.0572 | 0.9100 | 16.02 | 0.2145 |
| `exp031 balanced` | 0.0550 | 0.9138 | 6.09 | 0.2078 |

読み:

- `clip_fraction` は悪化していない。
- `ratio_max` はむしろ小さい。
- PPO の ratio 的な崩壊ではない。
- 性能低下は「policy更新が荒れた」よりも、「aux/value 側の信号配分が policy 改善に不利になった」可能性が高い。

## Interpretation

`exp_031` は、かなり重要な反証になった。

当初の仮説は以下だった。

```text
terminal / yaku / value の gradient norm を揃えると、より綺麗で安定した学習になるのではないか。
```

結果は逆で、gradient norm を揃えるほど性能は落ちた。

これは、今の Stage2a 構成では terminal signal が単なる補助 head ではなく、value_semantic trunk を policy に有用な方向へ押す主要な shaping signal になっていることを示唆する。

特に重要なのは、`exp031` で `clip_fraction` や `ratio_max` が悪化していない点である。  
つまり、学習が壊れたわけではなく、より「整った」gradient balance が、結果として弱い方策を作った可能性がある。

## Decision

`exp031 balanced` は本命から外す。

次の本命候補は以下:

1. `VS_LR100x`
2. `VS_LR300x`
3. `TERM50x`

ただし `VS_LR100x` と `VS_LR300x` の差を考えると、次は value_semantic 側ではなく policy 側 lr が律速になっているかを見る価値がある。

そのため、次実験 `exp_032` では `VS_LR100x` を土台に policy lr を上げる。

```text
P3x_VS100x:  policy_lr=0.0003, value_semantic_lr=0.01
P5x_VS100x:  policy_lr=0.0005, value_semantic_lr=0.01
P10x_VS100x: policy_lr=0.0010, value_semantic_lr=0.01
```

`exp032` で policy lr を上げても崩壊せず性能が改善するなら、`VS_LR100x` の 3seed 化より先に policy/value_semantic lr の組み合わせを詰める価値がある。
