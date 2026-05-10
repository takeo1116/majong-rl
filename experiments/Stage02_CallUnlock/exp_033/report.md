# Experiment Report: exp_033

作成日: 2026-05-10  
Stage: `Stage02_CallUnlock`

## Summary

`exp_033` では、`policy_lr=5x` / `value_semantic_lr=100x` に `target_kl` を加え、さらに CQ-0288/CQ-0289 後の整理された構成で再確認した。

結論:

- seed42 は強く、`P5x + target_kl` が上振れ性能を持つことは確認できた。
- しかし seed43 が明確に弱く、序盤の大きな悪化と最終的な低性能が出た。
- seed44 は開始されたが、seed43 の結果を見て `P5x` を安定設定として採用しない判断に移ったため、実験としては完走させていない。
- `policy_lr=5x` は探索設定としては有用だが、ルール拡張前 baseline には不向き。
- 後続の `exp_034` で `policy_lr=1x / value_semantic_lr=100x / target_kl=on` を3seed確認し、そちらを final baseline として採用する判断になった。

Decision:

```text
P5x_TKL は不採用。
policy lr は 1x に戻す。
value_semantic_lr=100x と target_kl は維持する。
```

## Background

`exp_032` では `P5x_VS100x` が seed42 で非常に強かった。

```text
P5x_VS100x seed42:
final  2.105
best   1.885
best10 2.044
tail10 2.098
tail20 2.105
```

ただし3seed化すると seed44 が弱く、平均では安定設定として不十分だった。

```text
P5x_VS100x 3seed:
final  mean 2.2017
best   mean 2.0433
best10 mean 2.1165
tail10 mean 2.1852
tail20 mean 2.1811
```

その後、弱かった seed44 に `target_kl` を入れた probe では大きく改善した。

```text
P5 seed44 + target_kl:
final  2.115
best   2.070
best10 2.119
tail10 2.155
tail20 2.194
score  29554.0
```

このため `exp_033` では、`P5x` に `target_kl` を加え、さらに以下の修正後の構成で再確認した。

- CQ-0288: dead weight だった `semantic_proj` 削除
- CQ-0289: `lr_groups.apply_to=["ppo"]` により imitation warmstart には lr_groups を適用しない

## Conditions

共通条件:

- `policy_ratio = 1.0`
- `ppo_mode = "separated"`
- `policy_anchor.enabled = false`
- `reward.point_delta_scale = 0.0001`
- `feature_encoder.tile_presence_flags = true`
- `model.value_hidden_dims = [256, 128]`
- `training.lr_groups.enabled = true`
- `training.lr_groups.apply_to = ["ppo"]`
- `training.lr_groups.policy = 0.0005`
- `training.lr_groups.value_semantic = 0.01`
- `clip_epsilon = 0.15`
- `entropy_coef = 0.0`
- `value_loss_coef = 0.125`
- `terminal_loss_coef = 0.1`
- `yaku_loss_coef = 0.05`
- `ppo_target_kl.enabled = true`
- `ppo_target_kl.target = 0.03`
- `ppo_target_kl.stop_multiplier = 1.5`
- `ppo_target_kl.skip_minibatch_on_exceed = true`
- `multi_cycle.num_cycles = 60`
- `selfplay_matches_per_cycle = 200`
- `gradient_norms.enabled = true`

Runs:

| label | seed | status | run |
|---|---:|---|---|
| `FINAL_P5_TKL_seed42` | 42 | completed | `runs/20260509_stage2a_exp033_final_p5_tkl_seed42_955f8a2b` |
| `FINAL_P5_TKL_seed43` | 43 | completed | `runs/20260510_stage2a_exp033_final_p5_tkl_seed43_c6b164bc` |
| `FINAL_P5_TKL_seed44` | 44 | interrupted | `runs/20260510_stage2a_exp033_final_p5_tkl_seed44_8ccade43` |

Note:

`seed44` は `run_map.json` 上では `running` のままだが、実体は imitation warmstart 中に中断された。`summary.json` も checkpoint も存在しないため、performance 集計には含めない。

## Performance

Lower avg_rank is better.

| seed | final | best | best_cycle | best10 | tail10 | tail20 | final win | final deal-in | final avg_score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.170 | 1.955 | 49 | 2.066 | 2.205 | 2.164 | 0.1880 | 0.1844 | 29138.0 |
| 43 | 2.340 | 2.210 | 40 | 2.265 | 2.347 | 2.322 | 0.2182 | 0.2096 | 26898.0 |

2seed aggregate:

| metric | mean | seed42 | seed43 |
|---|---:|---:|---:|
| final | 2.255 | 2.170 | 2.340 |
| best | 2.083 | 1.955 | 2.210 |
| best10 | 2.165 | 2.066 | 2.265 |
| tail10 | 2.276 | 2.205 | 2.347 |
| tail20 | 2.243 | 2.164 | 2.322 |
| final win | 0.2031 | 0.1880 | 0.2182 |
| final deal-in | 0.1970 | 0.1844 | 0.2096 |
| final avg_score | 28018.0 | 29138.0 | 26898.0 |

## Learning Curve

10-cycle block mean:

| seed | 00-09 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.324 | 2.280 | 2.153 | 2.191 | 2.124 | 2.205 |
| 43 | 2.592 | 2.567 | 2.515 | 2.379 | 2.297 | 2.347 |

読み:

- seed42 は上振れとしては強いが、tail10 は `2.205` で終盤維持は弱い。
- seed43 は序盤から大きく悪化し、cycle 40 以降に少し持ち直しても最終的に弱い。
- seed43 の `best=2.210` は、ルール拡張前 baseline として採用するには明確に弱い。

## PPO Diagnostics

Final cycle の代表値。

| seed | clip_fraction | max_prob_mean | max_prob_p95 | entropy | ratio_max | log_ratio_p01 | log_ratio_p99 | approx_kl_mean | approx_kl_max | target_kl_stop | skipped/checked |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.0361 | 0.9815 | 1.0000 | 0.0494 | 9.81 | -0.785 | 0.149 | 0.0137 | 0.1215 | 2 | 2/113 |
| 43 | 0.0705 | 0.9400 | 1.0000 | 0.1452 | 23.07 | -0.490 | 0.314 | 0.0074 | 0.0931 | 1 | 1/350 |

読み:

- seed42 は entropy が非常に低く、max_prob_mean が `0.9815` まで上がっている。
- seed43 は clip_fraction と ratio_max が大きめで、policy 更新の振動が強い。
- `target_kl` は一部 minibatch を止めているが、`P5x` の seed 間不安定性を十分には抑えられていない。

## Comparison To exp_034

後続の `exp_034` では policy lr を `1x` に戻し、`value_semantic_lr=100x` と `target_kl` は維持した。

```text
exp_034 P1x_TKL 3seed:
final  mean 2.078
best   mean 1.998
best10 mean 2.048
tail10 mean 2.113
tail20 mean 2.122
```

特に seed43/44 が安定している。

```text
exp_034 P1x seed43:
final  2.180
best   2.005
best10 2.046
tail10 2.123
tail20 2.136

exp_034 P1x seed44:
final  2.095
best   2.030
best10 2.067
tail10 2.119
tail20 2.131
```

`exp_033 P5x seed43` と比べると差は大きい。

```text
exp_033 P5x seed43:
final  2.340
best   2.210
best10 2.265
tail10 2.347
tail20 2.322
```

## Interpretation

### 1. P5x は上振れを作るが安定しない

seed42 の結果だけを見ると `P5x + target_kl` は魅力的に見える。  
しかし seed43 で大きく崩れ、`best=2.210` までしか出なかった。

これは `policy_lr=5x` が性能上限を上げるというより、分散を増やし、当たり seed では良く見えるが外れ seed では弱くなる挙動と解釈するのが自然。

### 2. target_kl は万能ではない

`target_kl` は悪い踏み込みを一部止めるが、policy lr が高すぎる場合の seed 不安定性までは完全には解消しない。

`P5x` を使うなら追加の entropy / stronger KL / lower clip などを試す余地はあるが、ルール拡張前にそこを詰める価値は低い。

### 3. Final baseline は exp_034 に移す

`exp_034` の `P1x_TKL` は 3seed で安定して強い。  
したがって、Stage2a のルール拡張前 baseline は `exp_034` を採用する。

## Decision

`exp_033` は P5x 不採用の根拠として閉じる。

採用しない理由:

- seed42 は上振れとして良いが、seed43 が弱い。
- `best10 / tail10 / tail20` が不安定。
- `target_kl` を入れても seed 間ばらつきが残る。
- 後続の `exp_034` がより安定して強い。

次:

```text
exp_034 を Stage2a final baseline として採用し、ルール拡張へ進む。
```

