# exp_039: Optional-Off Regression Check against exp034

## Purpose

`exp_039` checks whether the current post-CQ0296 codebase can reproduce the strong `exp_034` optional-off baseline when all optional actions are disabled.

This is a sanity check before continuing rule expansion. If optional-off performance is already weaker under the current code, then the degradation seen in `exp_036` / `exp_037` / `exp_038` is not caused only by enabling optional decisions.

## Conditions

The experiment was first run as seed42, then extended to seed43/44 after seed42 looked weaker than exp034.

Common settings:

| item | value |
|---|---:|
| cycles | 60 |
| selfplay_matches_per_cycle | 200 |
| optional_riichi / tsumo / ron / ankan / kakan / kyuushu | all false |
| feature_encoder.tile_presence_flags | true |
| feature_encoder.riichi_discard_mask | false |
| model.value_hidden_dims | `[256,128]` |
| policy lr | 0.0001 |
| value/semantic lr | 0.01 |
| target_kl | enabled, target=0.03, multiplier=1.5 |
| reward.point_delta_scale | 0.0001 |

Runs:

| seed | run |
|---:|---|
| 42 | `runs/20260512_20260512_stage2b_exp039_exp034_repro_off_seed42_1a12f2d7` |
| 43 | `runs/20260512_20260512_stage2b_exp039_exp034_repro_off_seed43_e3e2bc77` |
| 44 | `runs/20260512_20260512_stage2b_exp039_exp034_repro_off_seed44_a48d65a8` |

Run maps:

- `experiments/Stage02_CallUnlock/exp_039/run_map.json`
- `experiments/Stage02_CallUnlock/exp_039/followup_run_map.json`

## Result

Lower avg_rank is better.

### Per Seed

| experiment | seed | final | best | best_cycle | tail10 | tail20 |
|---|---:|---:|---:|---:|---:|---:|
| exp034 | 42 | 1.960 | 1.960 | 59 | 2.098 | 2.098 |
| exp034 | 43 | 2.180 | 2.005 | 31 | 2.123 | 2.136 |
| exp034 | 44 | 2.095 | 2.030 | 53 | 2.119 | 2.132 |
| exp039 | 42 | 2.145 | 2.070 | 53 | 2.171 | 2.191 |
| exp039 | 43 | 2.125 | 1.995 | 51 | 2.143 | 2.160 |
| exp039 | 44 | 2.115 | 1.990 | 49 | 2.155 | 2.156 |

### Aggregate

| experiment | final mean | final std | best mean | best std | tail10 mean | tail10 std | tail20 mean | tail20 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp034 | 2.078 | 0.091 | 1.998 | 0.029 | 2.113 | 0.011 | 2.122 | 0.017 |
| exp039 | 2.128 | 0.012 | 2.018 | 0.037 | 2.156 | 0.011 | 2.169 | 0.016 |
| exp039 - exp034 | +0.050 | - | +0.020 | - | +0.044 | - | +0.047 | - |

### Per-Seed Difference (`exp039 - exp034`)

| seed | final | best | tail10 | tail20 |
|---:|---:|---:|---:|---:|
| 42 | +0.185 | +0.110 | +0.073 | +0.094 |
| 43 | -0.055 | -0.010 | +0.021 | +0.024 |
| 44 | +0.020 | -0.040 | +0.037 | +0.024 |

## Learning Curve

10-cycle block mean:

| experiment | seed | 00-09 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp034 | 42 | 2.437 | 2.316 | 2.215 | 2.182 | 2.097 | 2.098 |
| exp034 | 43 | 2.409 | 2.285 | 2.199 | 2.142 | 2.149 | 2.123 |
| exp034 | 44 | 2.416 | 2.345 | 2.267 | 2.214 | 2.144 | 2.119 |
| exp039 | 42 | 2.364 | 2.366 | 2.302 | 2.234 | 2.212 | 2.171 |
| exp039 | 43 | 2.422 | 2.263 | 2.141 | 2.140 | 2.176 | 2.143 |
| exp039 | 44 | 2.336 | 2.331 | 2.298 | 2.261 | 2.156 | 2.155 |

## Optional Diagnostics

All exp039 runs kept optional decisions disabled.

| seed | optional_decision_count | riichi_optional_opened_count | riichi_bypassed_by_non_riichi_discard_count |
|---:|---:|---:|---:|
| 42 | 0 | 0 | 0 |
| 43 | 0 | 0 | 0 |
| 44 | 0 | 0 | 0 |

The observed gap is therefore not caused by optional branches being sampled.

## Interpretation

The seed42 result looked substantially weaker than exp034 seed42, but the 3seed follow-up softens the conclusion.

What the 3seed result says:

- This is not a catastrophic regression.
- `best` is broadly comparable: exp039 mean `2.018` vs exp034 mean `1.998`.
- `final` is also within the broad exp034 seed spread, though exp039 mean is worse by `+0.050`.
- The consistent signal is late/tail performance: `tail10` and `tail20` are worse by about `+0.04-0.05` across 3seed.

So the right reading is not "current optional-off is broken". It is more precise to say:

> The current codebase reproduces most of the exp034 optional-off strength, but late-cycle performance is consistently a little weaker.

This matters because optional-enabled experiments are being compared to exp034. If the post-CQ0296 optional-off baseline itself is now around `tail10=2.16` rather than `2.11`, then part of the optional-enabled gap comes from general code drift rather than optional actions alone.

## Likely Implications

The optional unlock degradation remains real, but its baseline should probably be updated.

- Compare optional-on experiments against exp039, not only against exp034.
- Still, optional-on runs such as exp036/037/038 are usually worse than exp039, so optional action learning/design is still a problem.
- The residual exp034 -> exp039 tail gap is small but consistent enough to investigate if we want a clean rule-expansion baseline.

Likely suspects remain optional-off-visible code/model changes:

1. Candidate/action type embedding expansion to 11 action types.
2. Optional summary action-type presence expansion in the value path.
3. CQ-0290 red-tile discard priority, probably small but behavioral.
4. Seed/initialization changes caused by model structure changes even when optional flags are off.

## Recommendation

Do not block all progress on this, but do not ignore it either.

Recommended next step:

1. Treat exp039 as the current-code optional-off reference.
2. If we continue optional action work, compare against exp039 rather than exp034.
3. For a cleaner baseline, run one targeted ablation around optional-summary/action-type expansion with optional flags off.

My preference: add a small CQ/probe to make optional summary/action-type expansion configurable or legacy-compatible when all optional families are disabled. That would directly test whether the `+0.04-0.05` late tail gap is architectural drift or just random/codebase noise.
