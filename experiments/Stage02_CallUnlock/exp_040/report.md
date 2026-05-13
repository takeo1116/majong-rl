# exp_040: Optional-All Gamma 0.95 Follow-up

作成日: 2026-05-14  
Stage: `Stage02_CallUnlock` / `Stage02b_OptionalAllGamma095`

## 1. Summary

`exp_040` では、optional action unlock 後の性能低下について、`gamma=0.5` が小さすぎるという仮説を検証した。

`gamma=0.95 / gae_lambda=0.95` に変更したところ、旧 optional-all 条件 (`exp_036`, gamma=0.5) より明確に改善した。

一方で、ルール拡張前の optional-off baseline (`exp_034`) および現コード optional-off baseline (`exp_039`) にはまだ届いていない。

結論:

- `gamma=0.5` は optional action unlock 後には明らかに弱すぎる。
- optional branch 挿入で reward が 1 decision 遅れる影響を、`gamma=0.95` はかなり緩和している。
- ただし optional-all にはまだ residual cost があり、tail performance は optional-off より弱い。
- それでも gamma 修正により大きな性能劣化はかなり回復したため、次のルール忠実度拡張へ進める水準と判断する。

## 2. Background

ClaudeCode review (`experiments/Stage02_CallUnlock/exp_039/claude_code_review.md`) で、optional branch 挿入により reward attribution が従来より 1 decision 遅れる可能性が指摘された。

特に `RIICHI_OPTIONAL` では、従来なら discard decision に直接紐づいていた engine step reward が、optional response decision を挟んだ後に発生する。

`gamma=0.5` では 1 decision 遅れるだけで reward signal が半減する。これは optional unlock 前には許容できていたとしても、optional unlock 後には過剰に短期化された discount になっている可能性がある。

そのため、今回は `gamma=0.95 / gae_lambda=0.95` を試した。

## 3. Conditions

| parameter | value |
|---|---:|
| seeds | 42, 43, 44 |
| cycles | 60 |
| selfplay_matches_per_cycle | 200 |
| optional_riichi / tsumo / ron / ankan / kakan / kyuushu | all true |
| gamma | 0.95 |
| gae_lambda | 0.95 |
| policy_ratio | 1.0 |
| ppo_mode | separated |
| reward.point_delta_scale | 0.0001 |
| policy lr | 0.0001 |
| value/semantic lr | 0.01 |
| model.value_hidden_dims | `[256,128]` |
| target_kl | 0.03 |
| target_kl stop_multiplier | 1.5 |
| target_kl skip | true |
| feature_encoder.tile_presence_flags | true |
| feature_encoder.riichi_discard_mask | true |
| semantic aux | enabled |

## 4. Runs

| label | seed | run_dir |
|---|---:|---|
| `GAMMA095_OPTIONAL_ALL_seed42` | 42 | `runs/20260513_20260513_stage2b_optional_all_gamma095_probe_seed42_cf98d62a` |
| `GAMMA095_OPTIONAL_ALL_seed43` | 43 | `runs/20260513_20260513_stage2b_optional_all_gamma095_probe_seed43_be3a2915` |
| `GAMMA095_OPTIONAL_ALL_seed44` | 44 | `runs/20260513_20260513_stage2b_optional_all_gamma095_probe_seed44_a677ca53` |

Driver:

```text
scripts/local/stage2/exp_040_driver.py
```

Run map:

```text
experiments/Stage02_CallUnlock/exp_040/run_map.json
```

## 5. Results

| seed | final | best | best_cycle | tail10 | tail20 | final win | final deal-in | final avg_score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.235 | 2.065 | 54 | 2.246 | 2.252 | 0.2258 | 0.2041 | 27515.5 |
| 43 | 2.325 | 2.110 | 18 | 2.319 | 2.285 | 0.2388 | 0.1943 | 27248.0 |
| 44 | 2.220 | 2.070 | 57 | 2.209 | 2.207 | 0.2365 | 0.1947 | 28688.0 |

Aggregate:

| metric | mean | std | min | max |
|---|---:|---:|---:|---:|
| final | 2.260 | 0.046 | 2.220 | 2.325 |
| best | 2.082 | 0.020 | 2.065 | 2.110 |
| tail10 | 2.258 | 0.046 | 2.209 | 2.319 |
| tail20 | 2.248 | 0.032 | 2.207 | 2.285 |
| final win | 0.2337 | 0.0057 | 0.2258 | 0.2388 |
| final deal-in | 0.1977 | 0.0045 | 0.1943 | 0.2041 |
| final avg_score | 27817.2 | 625.4 | 27248.0 | 28688.0 |

## 6. 10-Cycle Block Means

| seed | c00-09 | c10-19 | c20-29 | c30-39 | c40-49 | c50-59 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.423 | 2.345 | 2.327 | 2.252 | 2.257 | 2.246 |
| 43 | 2.413 | 2.339 | 2.262 | 2.276 | 2.251 | 2.319 |
| 44 | 2.455 | 2.340 | 2.308 | 2.275 | 2.206 | 2.209 |

seed42/44 は後半まで改善が残った。seed43 は cycle18 で best=2.110 まで行ったが、その後 tail が弱くなった。

## 7. Comparison

| experiment | condition | final | best | tail10 | tail20 |
|---|---|---:|---:|---:|---:|
| exp034 | optional-off old baseline, 3seed mean | 2.078 | 1.998 | 2.113 | 2.122 |
| exp039 | current optional-off, 3seed mean | 2.128 | 2.018 | 2.156 | 2.169 |
| exp036 | optional-all gamma=0.5, seed42 | 2.360 | 2.125 | 2.330 | 2.319 |
| exp040 | optional-all gamma=0.95, 3seed mean | 2.260 | 2.082 | 2.258 | 2.248 |

Interpretation:

- Compared with `exp036`, `gamma=0.95` clearly improves all major metrics.
- `best mean=2.082` is now relatively close to current optional-off (`exp039 best=2.018`).
- However, `tail20=2.248` remains weaker than `exp039 tail20=2.169` by about `+0.079`.
- The old `exp034` baseline is still stronger than both current optional-off and optional-all.

## 8. Optional Diagnostics

Final cycle family counts:

| seed | discard | response | riichi | tsumo | ron | ankan | kakan | kyuushu | optional_total | riichi_bypass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 92687 | 25277 | 1711 | 466 | 1657 | 284 | 539 | 30 | 4687 | 0.0012 |
| 43 | 92038 | 25068 | 1704 | 460 | 1592 | 374 | 528 | 29 | 4687 | 0.0012 |
| 44 | 89469 | 24394 | 1849 | 535 | 1539 | 330 | 805 | 23 | 5081 | 0.0005 |

Riichi bypass remains effectively zero. The degradation is therefore unlikely to be caused by the agent simply failing to open Riichi when available.

## 9. Interpretation

`gamma=0.95` strongly supports the hypothesis that `gamma=0.5` was too short-sighted after optional branch insertion.

The likely mechanism is:

1. Optional unlock inserts additional decisions between a strategic decision and the resulting reward.
2. With `gamma=0.5`, even a one-step delay halves the reward signal.
3. This weakens the PPO credit assignment for decisions immediately before optional actions.
4. `gamma=0.95` largely restores that signal.

However, the residual gap to optional-off suggests that gamma was not the only cost. Remaining candidates include:

- More complex action process increasing variance.
- Additional optional family samples diluting update composition.
- Late-cycle stability issues, especially visible in seed43.
- Remaining mismatch between optional decision learning and baseline/teacher semantics.

## 10. Decision

This result is good enough to proceed with rule fidelity expansion.

`exp040` does not fully recover `exp034`/`exp039`, but the large degradation seen in earlier optional-all experiments is substantially reduced. Continuing to optimize optional-all before moving on may yield incremental gains, but it risks spending too much time away from the main roadmap.

Recommended next baseline for rule-expanded experiments:

- `gamma=0.95`
- `gae_lambda=0.95`
- all optional decisions enabled
- `riichi_discard_mask=true`
- current stable optimizer settings from `exp034`/`exp039`

## 11. Next Actions

1. Move on to the next rule-fidelity expansion step.
2. Keep `gamma=0.95 / gae_lambda=0.95` as the default for optional-enabled Stage2b experiments.
3. If optional-all instability becomes a blocker later, revisit reward attribution / family-specific diagnostics instead of reverting optional actions.
