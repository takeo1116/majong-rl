# Experiment Report: exp_028

作成日: 2026-03-11  
対象: `experiments/exp_028/runbook.md`  
目的: `exp_027 A` の value 診断改善を保ったまま、PPO 更新強度を弱めて通常評価悪化を抑えられるかを確認する

## 1. 実験概要

比較参照:
- `experiments/exp_025/report.md`
- `experiments/exp_027/report.md`

新規実行条件:
- A: weak-lr
  - `training.lr=5e-5`
  - `training.epochs=4`
- B: weak-epochs
  - `training.lr=1e-4`
  - `training.epochs=2`

共通:
- `feature_encoder.shanten_hint.enabled=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- reward shaping 標準
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.1`
- `model.hidden_dims=[768,384]`
- `model.value_features.current_shanten.enabled=true`

共通条件:
- seeds: `42,43,44,45,46`
- phases: `imitation,selfplay,learner,eval`
- evaluation: `rotation`, `num_matches=30`
- selfplay: `num_matches=200`

## 2. 実行結果

- A batch: （ローカル run）
- B batch: （ローカル run）
- 両条件とも `success_count = 5/5`
- `summary.json.success=true`
- `shanten_diag` / `turn_diag` を全 run で確認

注記:
- A は最初の driver 検証で `config.yaml` の `lr` 表記差 (`5.0e-05`) を文字列比較して落ちた
- 実験本体は完了していたため、driver 側検証だけ修正した
- B はその後 CLI で個別実行し、`exp_028/run_map.json` に追記した

## 3. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| exp_025 | 3.3833 ± 0.0880 | -13269.2 ± 1585.1 | 0.04683 ± 0.01282 | 0.58175 ± 0.01120 |
| exp_027 A | 3.5367 ± 0.0933 | -15391.8 ± 1916.3 | 0.02787 ± 0.01030 | 0.58765 ± 0.01203 |
| A weak-lr | 3.4817 ± 0.0863 | -14297.0 ± 1347.9 | 0.04499 ± 0.01010 | 0.58369 ± 0.01042 |
| B weak-epochs | 3.5750 ± 0.0503 | -16073.2 ± 622.9 | 0.02072 ± 0.00902 | 0.58746 ± 0.01662 |

`eval_before -> eval` の delta:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| exp_025 | -0.0150 ± 0.0347 | -197.5 ± 533.8 | -0.00303 ± 0.00633 | +0.00690 ± 0.00517 |
| exp_027 A | +0.1067 ± 0.1020 | -1361.8 ± 1441.6 | -0.01403 ± 0.01161 | +0.00168 ± 0.01814 |
| A weak-lr | +0.1100 ± 0.0910 | -1065.8 ± 1629.6 | -0.00661 ± 0.01278 | +0.00878 ± 0.01431 |
| B weak-epochs | +0.1450 ± 0.1333 | -2043.2 ± 1593.1 | -0.02119 ± 0.01341 | +0.00149 ± 0.02106 |

所見:

1. **weak-lr は `exp_027 A` より通常評価をかなり戻した。**
   - `avg_rank`: `3.5367 -> 3.4817`
   - `Δavg_rank`: `+0.1067 -> +0.1100` と悪化幅は近いが、after 指標は明確に改善
   - `avg_score` / `win_rate` も `exp_027 A` より改善
2. ただし **`exp_025` にはまだ届かない。**
3. **weak-epochs は通常評価をさらに悪化させた。**
   - after 指標、delta 指標とも最悪

## 4. imitation 指標

mean ± std（seed=5）

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate | imitation value_loss |
|---|---:|---:|---:|
| exp_025 | 0.1797 ± 0.0086 | 0.5876 ± 0.0067 | 9.04e6 ± 4.78e5 |
| exp_027 A | 0.1849 ± 0.0062 | 0.5854 ± 0.0084 | 8.91e6 ± 4.71e5 |
| A weak-lr | 0.1843 ± 0.0092 | 0.5878 ± 0.0062 | 9.03e6 ± 4.69e5 |
| B weak-epochs | 0.1843 ± 0.0092 | 0.5878 ± 0.0062 | 9.03e6 ± 4.69e5 |

所見:

1. imitation 指標は A/B で同一で、差は PPO learner 側だけにある。
2. `exp_027 A` と比べても大差なく、今回の差は imitation では説明できない。

## 5. 主診断: policy 更新安定性

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | value_error_mean |
|---|---:|---:|---:|
| exp_025 | 0.5974 ± 0.0349 | 0.7163 ± 0.1417 | 225.24 ± 14.61 |
| exp_027 A | 0.8131 ± 0.0338 | 1.3367 ± 0.1386 | 116.34 ± 18.67 |
| A weak-lr | 0.6078 ± 0.0459 | 0.8338 ± 0.2519 | 229.98 ± 27.26 |
| B weak-epochs | 0.8566 ± 0.0242 | 1.6061 ± 0.1786 | 116.34 ± 18.67 |

所見:

1. **weak-lr は `clip_fraction` と `ratio_std` を `exp_025` 近傍まで戻した。**
   - `clip_fraction`: `0.8131 -> 0.6078`
   - `ratio_std`: `1.3367 -> 0.8338`
2. その代わり、**global `value_error_mean` は `exp_027 A` の改善をほぼ失った。**
   - `116.34 -> 229.98`
3. **weak-epochs は `value_error_mean` は維持したが、policy 更新安定性は改善しなかった。**
   - `clip_fraction` はむしろ悪化
   - `ratio_std` も `exp_027 A` より悪い

## 6. 主診断: shanten_diag

mean ± std（seed=5）

| 条件 | improve adv mean | worsen adv mean | improve value_error mean | worsen value_error mean |
|---|---:|---:|---:|---:|
| exp_025 | -0.0877 ± 0.0052 | +0.0608 ± 0.0039 | +300.15 ± 19.50 | +173.43 ± 11.55 |
| exp_027 A | -0.0740 ± 0.0068 | +0.0405 ± 0.0061 | +165.46 ± 22.30 | +89.27 ± 13.54 |
| A weak-lr | -0.0890 ± 0.0047 | +0.0592 ± 0.0040 | +308.05 ± 21.05 | +178.05 ± 11.95 |
| B weak-epochs | -0.0785 ± 0.0050 | +0.0406 ± 0.0056 | +165.46 ± 22.30 | +89.27 ± 13.54 |

所見:

1. **weak-lr は `exp_027 A` で得た `shanten_diag` 改善をほぼ失い、`exp_025` 近傍へ戻った。**
2. **weak-epochs は `exp_027 A` とほぼ同水準の `shanten_diag` を維持した。**
3. ただし、どちらも符号逆転自体は解消していない。

## 7. 主診断: turn_diag

mean ± std（seed=5）

| 条件 | early value_error mean | mid value_error mean | late value_error mean | late advantage mean |
|---|---:|---:|---:|---:|
| exp_025 | +102.30 ± 1.28 | +138.90 ± 2.06 | +247.45 ± 15.94 | -0.0259 ± 0.0018 |
| exp_027 A | +70.51 ± 5.65 | +89.51 ± 6.15 | +123.97 ± 21.50 | -0.0113 ± 0.0035 |
| A weak-lr | +100.56 ± 1.88 | +137.70 ± 3.36 | +252.76 ± 17.49 | -0.0259 ± 0.0019 |
| B weak-epochs | +69.37 ± 5.28 | +86.53 ± 6.30 | +123.97 ± 21.50 | -0.0099 ± 0.0029 |

所見:

1. **weak-lr は `turn_diag` 改善もほぼ失い、late misfit は `exp_025` 近傍に戻った。**
2. **weak-epochs は `turn_diag` 改善を維持した。**
3. つまり今回の 2 条件は、かなり綺麗に
   - `weak-lr = 更新安定性は改善するが value 診断改善を失う`
   - `weak-epochs = value 診断改善は維持するが通常評価がさらに悪化`
   に分かれた。

## 8. 解釈

今回の最重要点は、**診断改善と通常評価改善の両立が、単純な PPO 弱化ではできなかった** こと。

1. **weak-lr**
   - update 指標は明確に改善
   - 通常評価も `exp_027 A` よりは戻る
   - ただし value 診断改善をほぼ失う
   - つまり「大きいモデルで得た value fit 改善」を保てていない
2. **weak-epochs**
   - value 診断改善は保つ
   - しかし通常評価はさらに悪化
   - つまり「同じデータを少なく学習する」だけでは policy 壊れを止められない
3. したがって、**単純な lr / epochs 弱化だけではトレードオフを解けない。**

## 9. 結論

1. **採用なし。**
2. **weak-lr は診断対照として有用。**
   - `exp_027 A` で悪化した `clip_fraction / ratio_std` を、ほぼ `exp_025` 近傍まで戻せることが分かった。
3. **weak-epochs は不採用。**
   - 通常評価がさらに悪化し、改善方向が見えない。
4. 次段では、単純な PPO 強度弱化ではなく、**policy-value 干渉または target 定義** を本丸として見るべき。

## 10. 次アクション

1. 次に疑うべきは、`value_error` 改善と `clip_fraction / ratio_std` 悪化が同時に起きる理由、すなわち **policy-value 干渉**。  
2. 単純な `lr` / `epochs` 探索はここで一旦打ち止めとし、  
   - target 定義
   - trunk 共有の影響
   - value 更新が policy 更新へ与える影響  
   を優先して切り分ける。  
3. `exp_025` は依然として実用上の最良参照点であり、今後の比較基準は引き続きこれを用いる。
