# Experiment Report: exp_014

作成日: 2026-03-07  
対象: `experiments/exp_014/runbook.md`

## 1. 実験概要

目的: `shanten_hint` on/off を imitation-only 相当経路で比較し、imitation 後の到達点（after 指標）を評価する。

比較条件:
- A: `feature_encoder.shanten_hint={"enabled":false}`
- B: `feature_encoder.shanten_hint={"enabled":true}`

実行方式:
- seeds: `42,43,44,45,46`
- phases: `imitation,selfplay,eval`
- `selfplay.num_matches=0`（imitation 経路成立のため）
- eval: `rotation`, `num_matches=50`, `num_workers=10`

備考:
- この経路では `eval_before` を生成しないため、`eval_diff` は主評価に使わない。
- 比較は `eval` の after 指標そのものを使用。

## 2. 実行結果

| 条件 | shanten_hint | batch_dir | success |
|---|---|---|---:|
| A | off | `runs/20260307_stage1_full_flat_mlp_imitation_then_ppo_batch_1ed6c43f` | 5/5 |
| B | on  | `runs/20260307_stage1_full_flat_mlp_imitation_then_ppo_batch_797651d0` | 5/5 |

両条件とも `aggregate.eval_mode=rotation` を確認。

## 3. 主評価（after 指標）

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (off) | 3.4190 ± 0.0591 | -13553.9 ± 951.7 | 0.04861 ± 0.00577 | 0.57683 ± 0.01800 |
| B (on)  | 3.4070 ± 0.0584 | -13512.5 ± 1104.2 | 0.04688 ± 0.00713 | 0.57719 ± 0.01807 |

差分（B - A）:
- `avg_rank`: **-0.0120**（改善）
- `avg_score`: **+41.4**（改善）
- `win_rate`: **-0.00174**（悪化）
- `deal_in_rate`: **+0.00037**（わずか悪化）

## 4. 追跡情報（encoder / 入力次元）

- A: `summary.encoder_features.shanten_hint=false`, `input_dim=455`
- B: `summary.encoder_features.shanten_hint=true`, `input_dim=489`

`notes.md` にも `shanten_hint=on/off, input_dim=...` が記録され、run 単位で識別可能。

## 5. 実行時間と imitation 学習の補助観測

1 run あたり平均（sec）

| 条件 | imitation | selfplay | eval | total |
|---|---:|---:|---:|---:|
| A (off) | 45.75 | 0.003 | 265.98 | 311.73 |
| B (on)  | 47.87 | 0.002 | 269.56 | 317.43 |

補助観測:
- imitation loss（mean）
  - A: 2.18095
  - B: 2.17708
- imitation `total_steps` は両条件で同等（mean 11730.6）

解釈:
- `shanten_hint=on` は入力次元増加に伴い、総時間が約 +5.7s/run 増加。
- selfplay は `num_matches=0` のため実質コストなし（正常挙動）。

## 6. 結論

1. imitation-only 相当の after 比較では、`shanten_hint=on` は `avg_rank`/`avg_score` で小幅改善。  
2. ただし `win_rate` は小幅悪化、`deal_in_rate` はほぼ同等（わずか悪化）。  
3. 総合すると、**強い採用根拠が確定するほどの差ではないが、次段（warm start + PPO）での on/off 比較に進む価値はある**。

## 7. 次アクション

1. 同一 learner 条件で `shanten_hint` on/off の **warm start + PPO 比較** を実施（5 seeds, rotation=50 維持）。
2. 主判定は引き続き `avg_rank` と `avg_score` を優先し、`win_rate`/`deal_in_rate` を副指標で監視する。
3. 実行時間増分（約 +2%）が継続して許容範囲かを次段でも確認する。
