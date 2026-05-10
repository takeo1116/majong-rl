# Experiment Report: exp_030

作成日: 2026-05-08  
Stage: `Stage02_CallUnlock`

## Summary

`exp_030` では、CQ-0286 の optimizer lr group を使い、policy 側 lr を `1e-4` に固定したまま `value_semantic_lr` だけを `50x / 100x / 300x` に上げた。

結論:

- `VS_LR100x` が最も有望。`TERM50x` に近い tail を、terminal coef を極端に上げずに達成した。
- `VS_LR300x` は best/best10 は強いが、tail は `VS_LR100x` より悪い。
- `VS_LR50x` は明確に弱く、lr sweep は単調ではなかった。
- 高 lr でも policy 側が壊れた兆候は限定的で、`value_semantic` 側の学習速度はまだ主要な調整軸になり得る。

## Conditions

共通条件:

- seed: `42`
- `policy_ratio = 1.0`
- `ppo_mode = "separated"`
- `policy_anchor.enabled = false`
- `reward.point_delta_scale = 0.0001`
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

Sweep:

| label | value_semantic_lr | multiplier | run |
|---|---:|---:|---|
| `VS_LR50x` | `0.005` | 50x | `runs/20260508_stage2a_exp030_vs_lr50x_seed42_d58c5d55` |
| `VS_LR100x` | `0.010` | 100x | `runs/20260508_stage2a_exp030_vs_lr100x_seed42_f1c3da06` |
| `VS_LR300x` | `0.030` | 300x | `runs/20260508_stage2a_exp030_vs_lr300x_seed42_d8ecf0c9` |

## Performance

Lower avg_rank is better.

| condition | final | best | best10 | tail10 | tail20 | final win | final deal-in | final avg_score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `exp026 P100 scaled seed42` | 1.970 | 1.970 | 2.052 | 2.124 | 2.137 | 0.2368 | 0.1675 | 30757.5 |
| `TERM50x seed42` | 2.075 | 1.955 | 2.039 | 2.131 | 2.112 | 0.2085 | 0.1808 | 30150.5 |
| `VS_LR30x seed42` | 2.055 | 1.985 | 2.047 | 2.170 | 2.149 | 0.2085 | 0.1824 | 30465.0 |
| `VS_LR50x` | 2.205 | 2.130 | 2.181 | 2.248 | 2.256 | 0.2284 | 0.1828 | 28703.5 |
| `VS_LR100x` | 2.140 | 1.980 | 2.063 | 2.128 | 2.130 | 0.2147 | 0.2050 | 29369.5 |
| `VS_LR300x` | 2.155 | 1.935 | 2.007 | 2.154 | 2.141 | 0.2060 | 0.1915 | 29146.0 |

読み:

- `VS_LR100x` は `TERM50x seed42` と同程度の tail を出した。
- `VS_LR300x` は best/best10 が強く、短期的にはかなり良い checkpoint を作るが、終盤の安定性では `VS_LR100x` に劣る。
- `VS_LR50x` は今回の seed では悪く、`30x -> 50x -> 100x` のような単純な単調改善ではない。

## PPO / Gradient Diagnostics

late cycle 近傍の代表値。

| condition | clip_fraction | max_prob_mean | ratio_max | T/Y | T/V | Y/V | weighted_terminal | weighted_yaku | weighted_value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `VS_LR50x` | 0.0530 | 0.9222 | 7.11 | 8.37 | 1.73 | 0.21 | 0.0759 | 0.0092 | 0.0459 |
| `VS_LR100x` | 0.0563 | 0.9091 | 7.27 | 5.43 | 2.02 | 0.40 | 0.0567 | 0.0109 | 0.0299 |
| `VS_LR300x` | 0.0543 | 0.9224 | 17.65 | 14.62 | 3.03 | 0.21 | 0.1369 | 0.0095 | 0.0475 |

読み:

- `clip_fraction` は 50x/100x/300x で大きく悪化しておらず、policy 更新そのものは破綻していない。
- `VS_LR300x` は `ratio_max` が大きくなり、semantic 側の急な変化が一部 state で policy input を強く動かしている可能性がある。
- `VS_LR100x` でも terminal gradient は yaku/value より強く、完全に balanced とは言えない。
- ただし `VS_LR100x` は性能と安定性のバランスが最も良い。

## Interpretation

`exp_030` は、CQ-0285 後に弱くなった学習を `value_semantic_lr` でかなり戻せることを示した。

重要なのは、`terminal_loss_coef=5.0` のように terminal だけを極端に強くしなくても、`value_semantic_lr=0.01` で `TERM50x` に近い性能が出た点である。これは、性能低下の一部が「terminal signal の係数そのもの」ではなく、「value_semantic trunk/head 群を十分に動かせていない」ことから来ていた可能性を示す。

一方で、`VS_LR100x` でも gradient ratio はまだ terminal 側が強い。

```text
VS_LR100x late:
T/Y ~= 5.43
T/V ~= 2.02
Y/V ~= 0.40
```

そのため、次は terminal を基準にして yaku/value を上げ、signal balance をオーダーレベルで揃えられるかを見る価値がある。

## Decision

`VS_LR100x` を次の主候補とする。

次の実験 `exp_031` では、`value_semantic_lr=0.01` を維持しつつ、terminal を基準に yaku/value coefficient を上げる。

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

狙いは、`T/Y`, `T/V`, `Y/V` を 0.3-3.0 程度に収めながら、`VS_LR100x` の性能を維持できるか確認することである。
