# Experiment Report: exp_030

作成日: 2026-03-11  
対象: `experiments/exp_030/runbook.md`  
目的: baseline 条件 1 本だけを再実行し、`shanten_diag` に追加した `reward / point_delta_reward / shanten_delta_reward / delta_t` を使って、advantage 逆転が reward 段階から存在するのかを確認する

## 1. 実験概要

新規実行 1 条件:
- A: baseline only
  - `model.hidden_dims=[256,128]`
  - `model.policy_tower.enabled=false`
  - `model.value_tower.enabled=false`
  - `model.value_features.current_shanten.enabled=true`

共通固定:
- `feature_encoder.shanten_hint.enabled=true`
- reward shaping 標準
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
- `training.imitation_loss_mode=tie_aware_best_set`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.1`
- `training.epochs=4`
- `training.lr=1e-4`
- `training.value_loss_coef=0.25`
- seeds: `42,43,44,45,46`

batch:
- A: `runs/20260311_stage1_full_flat_mlp_imitation_then_ppo_batch_a40f4cbb`

`success_count = 5/5`。

## 2. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| `exp_025` 参照 | 3.3833 ± 0.0880 | -13269.2 ± 1585.1 | 0.04683 ± 0.01282 | 0.58175 ± 0.01120 |
| `exp_029 A` 参照 | 3.4450 ± 0.0594 | -13740.7 ± 1377.7 | 0.04487 ± 0.00500 | 0.58170 ± 0.01400 |
| A baseline only | 3.4450 ± 0.0594 | -13740.7 ± 1377.7 | 0.04487 ± 0.00500 | 0.58170 ± 0.01400 |

`eval_before -> eval` の delta:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| A baseline only | +0.0583 ± 0.0743 | -826.5 ± 1239.4 | -0.00758 ± 0.00302 | +0.00624 ± 0.00659 |

所見:
- 通常評価は `exp_029 A` と同等。今回の主目的は性能比較ではなく、新しい `shanten_diag` 内訳の確認。

## 3. imitation 指標

mean ± std（seed=5）

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate | imitation value_loss |
|---|---:|---:|---:|
| A baseline only | 0.17983 ± 0.00838 | 0.58837 ± 0.00762 | 9.051e6 ± 4.546e5 |

## 4. 主診断: 更新安定性

mean ± std（seed=5）

| 条件 | clip_fraction | ratio_std | old_value_mean | new_value_mean | value_error_mean |
|---|---:|---:|---:|---:|---:|
| A baseline only | 0.58458 ± 0.01287 | 0.66952 ± 0.07465 | -26.98 ± 4.85 | -223.61 ± 19.41 | +223.30 ± 15.34 |

所見:
- 更新安定性指標は `exp_029 A` と同等。
- 今回の主眼は `value_error` そのものより、`reward` と `delta_t` の群別符号を見ること。

## 5. 主診断: shanten_diag（reward / delta_t 分解）

mean ± std（seed=5）

| 群 | count | reward mean | point_delta_reward mean | shanten_delta_reward mean | delta_t mean | return mean | old_value mean | value_error mean | advantage mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| improve | 23769.8 ± 246.9 | -39.6600 ± 3.2875 | -39.6653 ± 3.2875 | +0.005301 ± 0.000021 | -39.4669 ± 3.2705 | -324.16 ± 24.26 | -26.47 ± 4.23 | +297.69 ± 20.05 | -0.08776 ± 0.00438 |
| same | 78413.0 ± 788.5 | -7.2880 ± 0.8118 | -7.2880 ± 0.8118 | 0.000000 ± 0.000000 | -6.9671 ± 0.7825 | -242.00 ± 17.28 | -27.64 ± 4.44 | +214.36 ± 12.84 | +0.01053 ± 0.00089 |
| worsen | 18861.6 ± 242.6 | -8.2688 ± 2.2900 | -8.2624 ± 2.2900 | -0.006384 ± 0.000027 | -7.9217 ± 2.2763 | -197.84 ± 14.61 | -25.27 ± 4.05 | +172.57 ± 10.76 | +0.05993 ± 0.00449 |

メタ:
- `status = partial`
- `available_samples = 121044.4 ± 799.8`
- `unavailable_samples = 800.0 ± 0.0`

所見:
- **逆転は reward 段階ですでに存在する。**
  - `improve.reward.mean` は `worsen.reward.mean` より大きく悪い。
  - しかもそのほぼ全ては `point_delta_reward` 側で起きている。
- **shaping reward の符号自体は正しい。**
  - improve 群は正、worsen 群は負。
  - ただし絶対値は非常に小さく、`point_delta_reward` 群差を覆せない。
- **`delta_t` も reward とほぼ同じ向きで逆転している。**
  - つまり逆転は GAE の後ろ向き累積より前、1-step TD 誤差段階ですでに形成されている。
- value はその逆転をさらに増幅している。
  - improve 群の `old_value` は worsen 群と大差ない浅い負値で、より大きい `value_error` を生んでいる。

## 6. 主診断: turn_diag

mean ± std（seed=5）

| bucket | count | return mean | old_value mean | new_value mean | value_update_delta mean | value_error mean | advantage mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| early | 10658.4 ± 53.5 | -107.01 ± 3.16 | -12.75 ± 0.37 | -140.05 ± 2.31 | -127.30 ± 2.20 | +94.26 ± 2.95 | +0.1406 ± 0.0050 |
| mid | 10658.4 ± 53.5 | -144.04 ± 6.15 | -14.03 ± 0.55 | -163.29 ± 4.47 | -149.27 ± 4.29 | +130.02 ± 5.80 | +0.0924 ± 0.0018 |
| late | 99727.6 ± 728.8 | -253.71 ± 17.43 | -29.96 ± 4.84 | -231.81 ± 20.73 | -201.85 ± 16.45 | +223.75 ± 16.30 | -0.0261 ± 0.0014 |

所見:
- `late` で `return` の絶対値も `value_error` も最も大きい。
- ただし `early` / `mid` でも `old_value` はかなり浅く、全 turn で過大評価が存在する。
- `late` だけが壊れているというより、**全 turn で value misfit があり、late で強く露出している** という読みが自然。

## 7. 解釈

今回の 1 条件診断で、逆転現象について次が分かった。

1. **逆転は value だけの問題ではない。**
   - improve 群の `reward` / `point_delta_reward` 自体が worsen 群より悪い。
   - したがって、局所ラベル（シャンテン改善）と系列 return のズレがすでにある。

2. **しかし value は逆転を明確に増幅している。**
   - improve 群の `old_value` は worsen 群とほぼ同じ浅い負値で、群差を十分反映できていない。
   - その結果、improve 群の `value_error` が最も大きくなり、`advantage` の負方向が強まる。

3. **逆転は `delta_t` 段階でほぼ完成している。**
   - `delta_t` は GAE と同じ 1-step TD 誤差であり、ここで improve がより負、worsen がより高い。
   - つまり「GAE の後ろ向き累積だけが悪い」のではない。

4. **shaping reward の設計自体は方向として正しいが、強さが足りない。**
   - improve には正、worsen には負が入っている。
   - ただし `point_delta_reward` 群差に比べて絶対値が小さい。

## 8. 結論

- advantage 逆転は、**reward 段階ですでに存在し、value がそれを増幅している**。
- よって「value 側だけ直せば解決」とは言えない。
- 次段で確認したい本命は、
  1. `point_delta_reward` 群差がなぜ improve/worsen で逆転するのか
  2. その群差を `policy_tower only` がどこまで縮めるのか
  3. value 側改善がその上にどれだけ上乗せされるのか

## 9. 次アクション

1. `exp_029 C policy_tower only` を、新しい `reward / delta_t` 診断付きで取り直す。
2. `exp_030` baseline と比較し、
   - `point_delta_reward`
   - `shanten_delta_reward`
   - `delta_t`
   のどこで逆転がどれだけ縮むかを確認する。
3. その結果を見て、
   - 逆転主因が reward 群差のままか
   - value 増幅が支配的か
   を切り分ける。
