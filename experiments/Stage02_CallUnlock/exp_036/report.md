# exp_036: Optional All 60-cycle Probe (seed42)

## Purpose

`exp_036` は Stage02b optional action unlock の最初の本格 probe。`Riichi / Tsumo / Ron / Ankan / Kakan / Kyuushu` をすべて optional decision として有効化し、60 cycle で学習が壊れないか、また既存の Stage02a stable baseline に対して性能がどの程度落ちるかを確認した。

この実験は CQ-0294 前の実行であり、以下はまだ未反映だった。

- `riichi_discard_mask` feature
- optional_riichi 有効時の teacher / baseline discard mask 分離
- riichi opportunity / bypass diagnostics
- optional summary の action-type presence 拡張

そのため、本 report は「pre-CQ0294 optional_all の挙動」として読む。

## Run

- run_dir: `runs/20260510_20260510_stage2b_optional_all_probe_seed42_f5e9ee5a`
- runbook: `experiments/Stage02_CallUnlock/exp_036/runbook.md`
- seed: 42
- cycles: 60
- selfplay matches/cycle: 200
- policy_ratio: 1.0
- PPO mode: separated
- reward.point_delta_scale: 0.0001
- policy lr: 0.0001
- value/semantic lr: 0.01
- target_kl: enabled, target=0.03, stop_multiplier=1.5
- gradient_norms: enabled
- optional flags: all enabled

Encoder summary:

| feature | value |
|---|---:|
| input_dim | 606 |
| tile_presence_flags | true |
| riichi_discard_mask | not available in this run |

## Main Result

Lower avg_rank is better.

| metric | value |
|---|---:|
| final avg_rank | 2.360 |
| best avg_rank | 2.125 |
| best cycle | 28 |
| mean avg_rank | 2.332 |
| tail5 avg_rank | 2.322 |
| tail10 avg_rank | 2.330 |
| tail20 avg_rank | 2.319 |

Cycle avg_rank sequence:

```text
2.415, 2.540, 2.420, 2.355, 2.370, 2.405, 2.370, 2.510, 2.365, 2.400,
2.380, 2.480, 2.405, 2.535, 2.485, 2.290, 2.175, 2.330, 2.360, 2.410,
2.140, 2.370, 2.240, 2.315, 2.250, 2.220, 2.305, 2.305, 2.125, 2.315,
2.220, 2.345, 2.420, 2.435, 2.155, 2.360, 2.295, 2.275, 2.205, 2.240,
2.230, 2.260, 2.425, 2.240, 2.410, 2.295, 2.280, 2.345, 2.430, 2.170,
2.400, 2.395, 2.250, 2.415, 2.225, 2.350, 2.275, 2.265, 2.360, 2.360
```

## Comparison

| run | cycles | final | best | best_cycle | mean | tail10 | tail20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp034 optional-off seed42 | 60 | 1.960 | 1.960 | 59 | 2.224 | 2.098 | 2.098 |
| exp036 optional-all seed42 | 60 | 2.360 | 2.125 | 28 | 2.332 | 2.330 | 2.319 |

Same 30-cycle window comparison:

| run | cycles | final30 | best30 | best_cycle | mean30 | tail10_30 |
|---|---:|---:|---:|---:|---:|---:|
| exp034 optional-off seed42 first30 | 30 | 2.310 | 2.030 | 24 | 2.322 | 2.215 |
| exp036 optional-all seed42 first30 | 30 | 2.315 | 2.125 | 28 | 2.353 | 2.259 |

Interpretation:

- optional_all did not crash and did learn into the low 2.1s at best.
- It did not reach exp034 seed42. The gap is visible both over 60 cycles and in the first-30-cycle matched comparison.
- The late phase did not recover; tail20 stayed around 2.32.

## Optional Decision Family Counts

Totals across 60 cycles:

| family | count |
|---|---:|
| discard | 5,677,691 |
| response | 1,525,089 |
| riichi | 89,507 |
| tsumo | 30,386 |
| ron | 97,601 |
| ankan | 22,462 |
| kakan | 65,658 |
| kyuushu | 1,515 |
| optional total | 307,129 |

Approximate per-cycle means:

| family | mean/cycle |
|---|---:|
| riichi | 1,492 |
| tsumo | 506 |
| ron | 1,627 |
| ankan | 374 |
| kakan | 1,094 |
| kyuushu | 25 |
| optional total | 5,119 |

Notes:

- Optional decisions are not rare as a whole: about 5.1k optional samples/cycle.
- `ron`, `riichi`, and `kakan` dominate optional samples.
- `kyuushu` is extremely rare and unlikely to explain the full degradation by itself.
- This run lacks riichi opportunity/bypass counters, so it cannot distinguish whether `optional_riichi` harmed learning through pre-optional discard avoidance. That gap motivated CQ-0294.

## Representative Cycles

| cycle | avg_rank | win_rate | updates | optional_count | riichi | tsumo | ron | ankan | kakan | kyuushu |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.415 | 0.246 | 489 | 4,642 | 1,000 | 497 | 1,670 | 365 | 1,089 | 21 |
| 10 | 2.380 | 0.239 | 477 | 4,928 | 1,342 | 440 | 1,668 | 317 | 1,139 | 22 |
| 20 | 2.140 | 0.254 | 474 | 5,095 | 1,533 | 491 | 1,610 | 453 | 989 | 19 |
| 28 | 2.125 | 0.247 | 487 | 5,213 | 1,560 | 503 | 1,617 | 423 | 1,085 | 25 |
| 40 | 2.230 | 0.250 | 504 | 5,443 | 1,718 | 484 | 1,652 | 452 | 1,106 | 31 |
| 50 | 2.400 | 0.228 | 484 | 5,080 | 1,520 | 496 | 1,580 | 422 | 1,040 | 22 |
| 59 | 2.360 | 0.232 | 494 | 5,206 | 1,503 | 498 | 1,584 | 344 | 1,252 | 25 |

## Interpretation

`optional_all` is playable but not yet a good default.

The result does not look like PPO collapse: updates remain high, the run reaches 2.125 at cycle 28, and eval fluctuates rather than monotonically degrading. The issue looks more like the expanded action/decision distribution makes the learning problem harder or noisier.

Three likely contributors:

1. Pre-CQ0294 riichi semantics were incomplete.
   - Policy discard mask widened under optional_riichi, but teacher/baseline semantics and riichi opportunity diagnostics were not yet separated.
   - This makes exp036 partially obsolete as a diagnostic for final optional_riichi behavior.
2. Optional branches add many samples with highly imbalanced labels.
   - Tsumo/Ron are likely mostly action-positive when presented.
   - Kan/Kyuushu are likely mostly skip-positive.
   - Without family-wise teacher/policy agreement, it is hard to know whether these branches are useful learning signal or mostly regularization/noise.
3. Value-side optional summary and encoder inputs were weaker than post-CQ0294.
   - `riichi_discard_mask` was missing.
   - action-type presence did not fully represent all new optional action types.

## Follow-up

Completed after this run:

- CQ-0294 fixed riichi teacher/baseline semantics, added riichi diagnostics, added `riichi_discard_mask`, and expanded optional summary action-type presence.
- exp_037 family ablation after CQ-0294 showed no single catastrophic family, but also no clear optional-off baseline recovery.

Recommended next step:

- CQ-0295: add optional decision family diagnostics and offline audit.
- Reuse exp036/exp037 shards/checkpoints to inspect family-wise teacher label distribution, policy agreement, entropy/max_prob, and PPO contribution before running another long optional-all experiment.

## CQ-0295 Optional Family Audit (20k sample)

After CQ-0295, an offline audit was run on final cycle shard with `--max-samples 20000`.

Outputs:

- `experiments/Stage02_CallUnlock/exp_036/optional_family_audit_final_20k/optional_family_audit_exp036_final_20k.json`
- `experiments/Stage02_CallUnlock/exp_036/optional_family_audit_final_20k/optional_family_audit_exp036_final_20k_summary.md`

Command:

```bash
./.venv/bin/python scripts/local/stage2/optional_family_audit.py \
  --config runs/20260510_20260510_stage2b_optional_all_probe_seed42_f5e9ee5a/config.yaml \
  --checkpoint runs/20260510_20260510_stage2b_optional_all_probe_seed42_f5e9ee5a/checkpoints/checkpoint_learner.pt \
  --shard-dir runs/20260510_20260510_stage2b_optional_all_probe_seed42_f5e9ee5a/cycle_59/selfplay \
  --output-dir experiments/Stage02_CallUnlock/exp_036/optional_family_audit_final_20k \
  --label exp036_final_20k \
  --device cpu \
  --batch-size 512 \
  --max-samples 20000
```

Summary:

| family | samples | teacher positive/action rate | policy agreement | entropy mean | max_prob mean | teacher_action_prob mean |
|---|---:|---:|---:|---:|---:|---:|
| discard | 15,231 | n/a | 0.617 | 0.371 | 0.844 | 0.609 |
| response | 3,955 | n/a | 0.812 | 0.078 | 0.972 | 0.808 |
| riichi | 246 | 1.000 | 1.000 | ~0.000 | 1.000 | 1.000 |
| tsumo | 83 | 1.000 | 1.000 | ~0.000 | 1.000 | 1.000 |
| ron | 248 | 1.000 | 1.000 | ~0.000 | 1.000 | 1.000 |
| ankan | 69 | 0.000 | 1.000 | ~0.000 | 1.000 | 1.000 |
| kakan | 164 | 0.000 | 1.000 | 0.005 | 0.999 | 0.999 |
| kyuushu | 4 | 0.000 | 1.000 | 0.181 | 0.954 | 0.954 |

Interpretation:

- Binary optional families are almost deterministic under the current teacher:
  - Riichi/Tsumo/Ron are always positive when presented.
  - Ankan/Kakan/Kyuushu are always Skip in this sample.
- The trained policy learns those binary labels almost perfectly. This suggests the optional branch itself is not failing to imitate the teacher.
- The degradation is therefore less likely to be “optional branch cannot learn its labels” and more likely one of:
  - expanded decision process changes trajectory distribution;
  - optional branches add many low-information samples and optimization load;
  - discard/response behavior is indirectly affected by changed state/action interfaces;
  - response branch remains confident but not perfectly aligned (`teacher_action_prob mean=0.808`, agreement=0.812), which may be worth deeper inspection.

This supports a next step focused on whether these deterministic optional decisions should be treated as environment automation / rule gates rather than learned policy branches, or whether their PPO weighting should be reduced/isolated.
