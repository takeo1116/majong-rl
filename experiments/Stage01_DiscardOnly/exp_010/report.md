# Experiment Report: exp_010

作成日: 2026-03-07  
対象: `experiments/exp_010/runbook.md`

## 1. 実験概要

目的: `training.value_loss_coef=0.25 (A)` と `0.5 (B)` を、seed ごとに同一の参照元 run（imitation/selfplay/eval_before）から分岐して比較する。  
比較軸: `eval_before -> eval` の delta を主、after 指標を副として評価。

- seeds: `42,43,44,45,46`
- 参照元 run: 各 seed で 1 本（`value_loss_coef=0.5`, full run）
- 比較 run: 各 seed で A/B 各 1 本（`--reuse-phases imitation,selfplay,eval_before`）
- 評価: `evaluation.mode=rotation`, `evaluation.num_matches=50`, `evaluation.num_workers=10`

## 2. run 対応

run 対応表は `experiments/exp_010/run_map.json` を正とする。

| seed | ref run | A run (`v=0.25`) | B run (`v=0.5`) |
|---:|---|---|---|
| 42 | `20260307_stage1_full_flat_mlp_imitation_then_ppo_ad6ea9d4` | `20260307_stage1_full_flat_mlp_imitation_then_ppo_c30b9058` | `20260307_stage1_full_flat_mlp_imitation_then_ppo_2491d84b` |
| 43 | `20260307_stage1_full_flat_mlp_imitation_then_ppo_8eebd29b` | `20260307_stage1_full_flat_mlp_imitation_then_ppo_12011789` | `20260307_stage1_full_flat_mlp_imitation_then_ppo_deb14f3b` |
| 44 | `20260307_stage1_full_flat_mlp_imitation_then_ppo_039b24c0` | `20260307_stage1_full_flat_mlp_imitation_then_ppo_7b67359e` | `20260307_stage1_full_flat_mlp_imitation_then_ppo_954d8f5c` |
| 45 | `20260307_stage1_full_flat_mlp_imitation_then_ppo_b33dfd79` | `20260307_stage1_full_flat_mlp_imitation_then_ppo_a7d401fd` | `20260307_stage1_full_flat_mlp_imitation_then_ppo_9653c112` |
| 46 | `20260307_stage1_full_flat_mlp_imitation_then_ppo_7e131051` | `20260307_stage1_full_flat_mlp_imitation_then_ppo_90051d6f` | `20260307_stage1_full_flat_mlp_imitation_then_ppo_3f4fb7ca` |

## 3. 再利用成立確認

A/B 全 10 run で以下を確認:

- `summary.json.success == true`
- `summary.json.reuse_info.reused_phases` に `imitation,selfplay,eval_before`
- `eval/eval_diff.json` が存在
- `eval_diff` の 4 指標 delta がすべて非 null

今回の主目的（rotation + reuse で eval_diff が出ること）は達成。

## 4. 主評価（delta: eval_before -> eval）

mean ± std（seed=5）

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| A (`value_loss_coef=0.25`) | +0.0430 ± 0.0453 | -689.6 ± 725.1 | -0.01466 ± 0.00422 | +0.00563 ± 0.00864 |
| B (`value_loss_coef=0.5`)  | +0.0520 ± 0.0435 | -925.1 ± 434.0 | -0.01438 ± 0.00527 | +0.00215 ± 0.00777 |

補足（B - A）:

- `Δavg_rank`: +0.0090（A優位）
- `Δavg_score`: -235.5（A優位）
- `Δwin_rate`: +0.00028（B優位, ごく小）
- `Δdeal_in_rate`: -0.00348（B優位）

## 5. after 指標（eval 後）

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (`0.25`) | 3.4200 ± 0.0677 | -13774.8 ± 987.9 | 0.04283 ± 0.00602 | 0.57414 ± 0.01355 |
| B (`0.5`)  | 3.4290 ± 0.0592 | -14010.3 ± 795.7 | 0.04311 ± 0.00703 | 0.57066 ± 0.01370 |

## 6. 時間（参考）

A/B は再利用 run なので、imitation/selfplay は再実行されず learner+eval が支配的。

1 run あたり平均:

| 条件 | total sec | learner sec | eval sec |
|---|---:|---:|---:|
| A (`0.25`) | 287.49 | 16.93 | 270.55 |
| B (`0.5`)  | 284.99 | 18.37 | 266.62 |

## 7. 結論

1. `0.25` vs `0.5` は今回も僅差で、方向は完全には一貫しない。  
2. ただし runbook の主評価優先順（`Δavg_rank` → `Δavg_score` → `Δdeal_in_rate`）に従うと、今回は **A (`0.25`) を僅差採用**。  
3. 一方で `deal_in_rate` と `win_rate` は B がわずかに良く、過信は禁物。

## 8. 次アクション

1. baseline を `value_loss_coef=0.25` に仮置きし、次 runbook で再確認。  
2. 同時に `deal_in_rate` 悪化を抑える補助ノブ（`entropy_coef` / `clip_epsilon` など）を小規模 sweep で確認。  
3. 今後の learner 比較は本 runbook と同様に reuse 前提（seed ごとの ref 分岐）で継続する。
