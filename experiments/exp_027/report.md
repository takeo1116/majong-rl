# Experiment Report: exp_027

作成日: 2026-03-10  
対象: `experiments/exp_027/runbook.md`  
目的: `exp_026` の延長として、value 表現をさらに強くしたときに `shanten_diag` / `turn_diag` の改善傾向が続くか、また通常評価が追随するかを確認する

## 1. 実験概要

比較参照:
- `experiments/exp_025/report.md`
- `experiments/exp_026/report.md`

新規実行条件:
- A: `model.hidden_dims=[768,384]`
- B: `model.hidden_dims=[1024,512]`
- 共通:
  - `feature_encoder.shanten_hint.enabled=true`
  - `training.imitation_loss_mode=tie_aware_best_set`
  - reward shaping 標準
    - `reward.shaping.shanten_delta.enabled=true`
    - `reward.shaping.shanten_delta.scale=0.01`
    - `reward.shaping.shanten_delta.mode=both`
    - `reward.shaping.shanten_delta.schedule.type=linear_decay`
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.1`
  - `model.value_features.current_shanten.enabled=true`

共通条件:
- seeds: `42,43,44,45,46`
- phases: `imitation,selfplay,learner,eval`
- evaluation: `rotation`, `num_matches=30`
- selfplay: `num_matches=200`

## 2. 実行結果

- A batch: `runs/20260310_stage1_full_flat_mlp_imitation_then_ppo_batch_6274ad41`
- B batch: `runs/20260310_stage1_full_flat_mlp_imitation_then_ppo_batch_cbdf3c52`
- 両条件とも `success_count = 5/5`
- `summary.json.success=true`
- `shanten_diag` / `turn_diag` を全 run で確認
- `summary.model_features.value_features.current_shanten.enabled=true` を確認

## 3. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| exp_025 (`[256,128]`, no current_shanten) | 3.3833 ± 0.0880 | -13269.2 ± 1585.1 | 0.04683 ± 0.01282 | 0.58175 ± 0.01120 |
| exp_026 (`[512,256]`, current_shanten) | 3.4300 ± 0.0509 | -14080.5 ± 1222.1 | 0.03957 ± 0.01300 | 0.58311 ± 0.01715 |
| A (`[768,384]`, current_shanten) | 3.5367 ± 0.0933 | -15391.8 ± 1916.3 | 0.02787 ± 0.01030 | 0.58765 ± 0.01203 |
| B (`[1024,512]`, current_shanten) | 3.5850 ± 0.0706 | -15661.2 ± 1087.9 | 0.02019 ± 0.00558 | 0.58309 ± 0.01835 |

`eval_before -> eval` の delta:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| exp_025 | -0.0150 ± 0.0388 | -197.5 ± 596.9 | -0.00303 ± 0.00708 | +0.00690 ± 0.00578 |
| exp_026 | +0.0217 ± 0.0548 | -591.8 ± 682.7 | -0.00586 ± 0.01216 | +0.00694 ± 0.01291 |
| A | +0.1067 ± 0.1140 | -1361.8 ± 1611.7 | -0.01403 ± 0.01298 | +0.00168 ± 0.02029 |
| B | +0.1017 ± 0.1268 | -1230.2 ± 1608.3 | -0.01785 ± 0.01724 | -0.00368 ± 0.00856 |

所見:

1. **通常評価はモデル拡大とともに一貫して悪化した。**
2. `exp_026` で見えた「診断改善と通常評価悪化の乖離」は、A/B でさらに拡大した。
3. `avg_rank` / `avg_score` / `win_rate` では `exp_025` が依然として最良。

## 4. imitation 指標

mean ± std（seed=5）

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate | imitation value_loss |
|---|---:|---:|---:|
| exp_025 | 0.1797 ± 0.0086 | 0.5876 ± 0.0067 | 9.04e6 ± 4.78e5 |
| exp_026 | 0.1812 ± 0.0055 | 0.5837 ± 0.0070 | 8.98e6 ± 4.81e5 |
| A | 0.1849 ± 0.0062 | 0.5854 ± 0.0084 | 8.91e6 ± 4.71e5 |
| B | 0.1879 ± 0.0123 | 0.5816 ± 0.0111 | 8.82e6 ± 4.75e5 |

所見:

1. teacher top1 はサイズとともに微増している。
2. best-set-hit は横ばいで、通常評価悪化を説明するほどの imitation 崩れは見えない。
3. imitation value_loss もやや改善しており、warm start 側の学習はむしろ素直。

## 5. 主診断: shanten_diag

mean ± std（seed=5）

| 条件 | improve adv mean | worsen adv mean | improve value_error mean | worsen value_error mean |
|---|---:|---:|---:|---:|
| exp_025 | -0.0877 ± 0.0052 | +0.0608 ± 0.0044 | +300.15 ± 21.81 | +173.43 ± 12.91 |
| exp_026 | -0.0771 ± 0.0035 | +0.0490 ± 0.0069 | +194.55 ± 23.36 | +107.18 ± 13.45 |
| A | -0.0740 ± 0.0068 | +0.0405 ± 0.0061 | +165.46 ± 22.30 | +89.27 ± 13.54 |
| B | -0.0805 ± 0.0049 | +0.0316 ± 0.0088 | +150.00 ± 22.58 | +77.30 ± 13.38 |

所見:

1. **value misfit はサイズとともに継続的に改善している。**
   - `improve.value_error.mean`: `300.15 -> 194.55 -> 165.46 -> 150.00`
   - `worsen.value_error.mean`: `173.43 -> 107.18 -> 89.27 -> 77.30`
2. `worsen.advantage.mean` は継続的に 0 に近づいている。
3. `improve.advantage.mean` も A では改善したが、B ではやや戻った。
4. **符号逆転自体はなお未解消。**
   - `improve` は負
   - `worsen` は正

## 6. 主診断: turn_diag

mean ± std（seed=5）

| 条件 | early value_error mean | mid value_error mean | late value_error mean | late advantage mean |
|---|---:|---:|---:|---:|
| exp_025 | +102.30 ± 1.43 | +138.90 ± 2.31 | +247.45 ± 17.82 | -0.0259 ± 0.0020 |
| exp_026 | +73.19 ± 5.85 | +95.15 ± 9.12 | +153.23 ± 21.34 | -0.0172 ± 0.0025 |
| A | +70.51 ± 5.65 | +89.51 ± 6.15 | +123.97 ± 21.50 | -0.0113 ± 0.0035 |
| B | +69.79 ± 10.16 | +87.02 ± 10.99 | +101.85 ± 17.91 | -0.0062 ± 0.0031 |

所見:

1. **turn_diag でも改善傾向は継続。**
2. 特に `late.value_error.mean` はサイズ拡大とともに大きく低下している。
3. `late.advantage.mean` も 0 に近づいているが、まだ負のまま。
4. つまり、終盤 misfit は表現強化でかなり緩和できるが、通常評価を反転させるには足りない。

## 7. learner 補助指標

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | value_error_mean |
|---|---:|---:|---:|
| exp_025 | 0.5974 ± 0.0349 | 0.7163 ± 0.1417 | 225.24 ± 14.61 |
| exp_026 | 0.6502 ± 0.0354 | 2.6638 ± 2.6950 | 141.22 ± 18.51 |
| A | 0.8131 ± 0.0338 | 1.3367 ± 0.1386 | 116.34 ± 18.67 |
| B | 0.9110 ± 0.0148 | 1.9090 ± 0.2792 | 97.77 ± 16.15 |

所見:

1. global `value_error_mean` は一貫して改善している。
2. 一方で `clip_fraction` はサイズ拡大とともに大きく悪化している。
3. `ratio_std` は `exp_025` より常に悪く、policy 更新の不安定化が通常評価悪化の有力候補である。

## 8. 解釈

今回の結果から言えることは次の通り。

1. **表現力不足仮説はさらに支持された。**
   - `shanten_diag` / `turn_diag` / global `value_error` は一貫して改善している。
2. **しかし通常評価は一貫して悪化した。**
   - したがって「value 診断値を改善すれば性能も上がる」という単純な関係は成り立っていない。
3. **新たな主疑惑は policy 更新の不安定化。**
   - `clip_fraction` の急増
   - `ratio_std` の悪化
   が通常評価悪化と整合する。
4. `current_shanten=true` を含む強化版 value 表現は、診断上は有効だが、そのままでは採用できない。

## 9. 結論

1. **A/B とも不採用。**
   `exp_025` の通常評価を更新できない。
2. **value 表現強化で診断値は改善する。**
   ただし、改善は PPO の通常評価にそのまま繋がらない。
3. 次段では、
   - value 診断改善
   - policy 更新安定性
   のトレードオフを切る必要がある。

## 10. 次アクション

1. 次に見るべきは、表現強化そのものより **policy 更新安定性との関係** である。  
2. 具体的には、現採用候補 `exp_025` を基準に、
   - model 拡大をやめるべきか
   - あるいは model 拡大条件で PPO 更新強度を弱めるべきか
   を比較する。  
3. 少なくとも、これ以上 hidden_dims を大きくするだけの探索は優先しない。
