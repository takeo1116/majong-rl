# exp_037: Optional Action Family Ablation (1 seed)

## Purpose

CQ-0291/CQ-0292/CQ-0294 で optional action を本格的に開放した後、exp_036 の `optional_all` が性能を落とした原因を family 別に切り分ける。

対象は seed42 の 1seed probe。各条件は 30 cycle、selfplay 200 matches/cycle。

## Conditions

共通設定:

- base: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- policy_ratio: 1.0
- reward.point_delta_scale: 0.0001
- policy lr: 0.0001
- value/semantic lr: 0.01
- target_kl: enabled, target=0.03, stop_multiplier=1.5
- semantic aux: enabled
- tile_presence_flags: enabled
- riichi_discard_mask: enabled

Ablation:

| label | riichi | tsumo | ron | ankan | kakan | kyuushu |
|---|---:|---:|---:|---:|---:|---:|
| WIN_ONLY | off | on | on | off | off | off |
| KAN_ONLY | off | off | off | on | on | on |
| RII_ONLY | on | off | off | off | off | off |

Run map: `experiments/Stage02_CallUnlock/exp_037/run_map.json`

## Results

### Main Eval Metrics

Lower avg_rank is better.

| run | cycles | final | best | best_cycle | mean | tail5 | tail10 | tail20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp034 optional_off seed42, first 30 cycles | 30 | 2.310 | 2.030 | 24 | 2.322 | 2.238 | 2.215 | 2.290 |
| exp036 optional_all seed42, first 30 cycles | 30 | 2.315 | 2.125 | 28 | 2.353 | 2.204 | 2.259 | 2.344 |
| exp037 WIN_ONLY | 30 | 2.355 | 2.075 | 28 | 2.330 | 2.241 | 2.297 | 2.326 |
| exp037 KAN_ONLY | 30 | 2.210 | 2.210 | 29 | 2.347 | 2.269 | 2.272 | 2.319 |
| exp037 RII_ONLY | 30 | 2.315 | 2.140 | 6 | 2.323 | 2.278 | 2.277 | 2.307 |

### Cycle Sequences

| run | avg_rank by cycle |
|---|---|
| WIN_ONLY | 2.515, 2.270, 2.380, 2.450, 2.400, 2.170, 2.320, 2.280, 2.280, 2.330, 2.330, 2.355, 2.405, 2.360, 2.385, 2.385, 2.320, 2.440, 2.330, 2.235, 2.285, 2.520, 2.370, 2.225, 2.365, 2.250, 2.375, 2.150, 2.075, 2.355 |
| KAN_ONLY | 2.565, 2.435, 2.435, 2.410, 2.390, 2.405, 2.375, 2.355, 2.280, 2.385, 2.400, 2.325, 2.385, 2.425, 2.520, 2.315, 2.240, 2.385, 2.390, 2.270, 2.270, 2.285, 2.305, 2.240, 2.280, 2.265, 2.290, 2.275, 2.305, 2.210 |
| RII_ONLY | 2.405, 2.420, 2.410, 2.415, 2.325, 2.340, 2.140, 2.270, 2.285, 2.520, 2.455, 2.315, 2.290, 2.385, 2.420, 2.350, 2.160, 2.355, 2.380, 2.270, 2.195, 2.365, 2.200, 2.280, 2.340, 2.315, 2.335, 2.285, 2.140, 2.315 |

### Decision Family Diagnostics

Totals across 30 cycles.

| run | optional_total | riichi | tsumo | ron | ankan | kakan | kyuushu |
|---|---:|---:|---:|---:|---:|---:|---:|
| WIN_ONLY | 63,811 | 0 | 14,926 | 48,885 | 0 | 0 | 0 |
| KAN_ONLY | 40,017 | 0 | 0 | 0 | 11,612 | 27,643 | 762 |
| RII_ONLY | 51,797 | 51,797 | 0 | 0 | 0 | 0 | 0 |

Riichi opportunity diagnostics:

| run | opportunities | opened | bypassed | bypass_rate |
|---|---:|---:|---:|---:|
| WIN_ONLY | 48,864 | 0 | 0 | 0.0000 |
| KAN_ONLY | 46,158 | 0 | 0 | 0.0000 |
| RII_ONLY | 51,820 | 51,797 | 23 | 0.0004 |

Notes:

- RII_ONLY almost always opens riichi optional when the opportunity exists. The bypass rate is effectively zero.
- WIN_ONLY introduces about 2.1k optional win decisions/cycle.
- KAN_ONLY introduces about 1.3k optional kan/kyuushu decisions/cycle.

## Interpretation

The ablation does not identify a single catastrophic family.

- `WIN_ONLY` has the best single checkpoint among the three (`2.075` at cycle 28), but its final/tail metrics are still weaker than exp034 optional-off seed42 first-30-cycle tail10.
- `KAN_ONLY` is the smoothest late improver and ends at its best checkpoint (`2.210`), but the absolute level remains below the optional-off reference.
- `RII_ONLY` is not obviously broken after CQ-0294. The policy almost never bypasses riichi opportunities, and performance is comparable to the other ablations. However, it does not recover the optional-off baseline.

The most likely reading is that the degradation in `optional_all` is not caused by one isolated family bug. It is more likely a distribution/decision-space expansion effect: the model now has to learn several rare optional branches whose labels are mostly trivial or highly imbalanced, and this adds optimization burden before yielding meaningful strategic benefit.

## Recommendation

For moving toward rule expansion, avoid jumping directly to `optional_all` as the next stable baseline.

Recommended next experiment:

1. Use `RII_ONLY` or `WIN_ONLY` only if the goal is specifically to debug/validate that family.
2. For a stable next training baseline, keep optional actions off for performance continuity, or enable only one family and run 3seed if we decide that family is strategically important enough.
3. If we continue optional unlock work, the next most informative probe is not another family ablation but label/behavior diagnostics for optional branches: action distribution, teacher entropy, positive action rate, and policy agreement per `decision_family`.

My current preference: do not 3seed all family ablations yet. First decide whether the next stage prioritizes rule fidelity or best playing strength. If rule fidelity is the priority, start with `RII_ONLY` because it is common and semantically important. If playing strength is the priority, stay with exp034-style optional-off while proceeding to the next rule expansion outside optional decisions.
