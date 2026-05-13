# exp_038: Model Capacity Probe after Optional Unlock

## Purpose

`exp_038` tested whether the performance drop seen after optional-action rule expansion can be explained by insufficient model capacity.

The probe ran three seed42, 30-cycle conditions:

| label | optional flags | value_hidden_dims | intent |
|---|---|---:|---|
| `OFF_WIDE1_seed42` | all off | `[384,192]` | Does optional-off improve from larger value/semantic trunk? |
| `OFF_WIDE2_seed42` | all off | `[512,256]` | Does an even larger value/semantic trunk help? |
| `RII_WIDE1_seed42` | riichi only | `[384,192]` | Does riichi optional recover with more capacity? |

Common settings:

- seed: 42
- cycles: 30
- selfplay matches/cycle: 200
- policy_ratio: 1.0
- ppo_mode: separated
- reward.point_delta_scale: 0.0001
- policy lr: 0.0001
- value/semantic lr: 0.01
- target_kl: enabled, target=0.03, multiplier=1.5, skip enabled
- semantic aux: enabled
- tile_presence_flags: enabled
- riichi_discard_mask: enabled

Run map: `experiments/Stage02_CallUnlock/exp_038/run_map.json`

## Main Results

Lower avg_rank is better.

| run | cycles | final | best | best_cycle | tail5 | tail10 | tail20 | final win_rate | tail10 win_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp034 optional-off seed42, first 30 | 30 | 2.310 | 2.030 | 24 | 2.201 | 2.215 | 2.266 | 0.221 | 0.234 |
| exp037 RII_ONLY seed42 | 30 | 2.315 | 2.140 | 6 | 2.278 | 2.277 | 2.307 | 0.227 | 0.236 |
| exp038 OFF_WIDE1 | 30 | 2.300 | 2.170 | 12 | 2.271 | 2.269 | 2.282 | 0.229 | 0.240 |
| exp038 OFF_WIDE2 | 30 | 2.265 | 2.140 | 27 | 2.274 | 2.272 | 2.311 | 0.228 | 0.242 |
| exp038 RII_WIDE1 | 30 | 2.300 | 2.205 | 28 | 2.296 | 2.304 | 2.325 | 0.227 | 0.237 |

## Decision Diagnostics

Totals across 30 cycles.

| run | optional_total | riichi opportunities | riichi opened | riichi bypassed | bypass_rate |
|---|---:|---:|---:|---:|---:|
| OFF_WIDE1 | 0 | 49,556 | 0 | 0 | 0.0000 |
| OFF_WIDE2 | 0 | 52,070 | 0 | 0 | 0.0000 |
| RII_WIDE1 | 51,221 | 51,241 | 51,221 | 20 | 0.0004 |

RII_WIDE1 again shows that the policy almost always opens the riichi optional branch when available. The problem is not primarily riichi bypass.

## Interpretation

Capacity increase alone did not recover the optional-off baseline.

- `OFF_WIDE2` produced a reasonable best checkpoint (`2.140` at cycle 27), but its tail metrics are still weaker than exp034 seed42 first-30-cycle reference (`tail10=2.215`).
- `OFF_WIDE1` is not clearly better than the original optional-off model.
- `RII_WIDE1` is worse than exp037 `RII_ONLY` on tail metrics and does not support the hypothesis that riichi optional degradation is mainly value/semantic trunk capacity shortage.

The cleanest reading is that value/semantic trunk capacity is not the main bottleneck. The optional-action degradation still looks more like a decision-space / curriculum / supervision-budget issue than a simple representational capacity issue.

One caveat: all exp038 runs used `riichi_discard_mask=true`, so the optional-off WIDE runs are not a perfectly identical reproduction of exp034. Still, the lack of a clear improvement is enough to avoid making WIDE the next default on this evidence alone.

## Recommendation

Do not 3seed WIDE/RII_WIDE immediately.

Preferred next direction:

1. Keep the stable optional-off configuration as the strength baseline.
2. For rule-fidelity work, investigate optional-action training design rather than simply increasing value/semantic capacity.
3. If using a wider model later, treat `OFF_WIDE2` as a possible engineering option, not as a proven improvement.

A more promising rule-fidelity path is to reduce the learning burden of nearly deterministic optional families, or introduce optional decisions with curriculum/automation rather than asking PPO to absorb all optional branches at once.
