# Stage3 Experiment Report (Runbook 3 + Additional D/E)

作成日時: 2026-03-05 19:35 JST  
Source Runbook: `experiments/exp_003/runbook.md`  
対象: Stage1 Full/Flat/MLP PPO 系比較（10 seed）

## 1. 実験一覧

- 実験A: PPO直学習（基準）
- 実験B: imitation→PPO（フル, 当初設定）
- 実験C: imitation→PPO（軽量, 当初設定）
- 実験D: imitation→PPO（フル, 公平化追加実験）
- 実験E: imitation→PPO（軽量, 公平化追加実験）

追加実験D/Eの意図:
- A と B/C で `selfplay.policy_ratio` と `selfplay.save_baseline_actions` が異なり、
  warm start 効果以外の要因が混ざる交絡があった。
- D/E は `policy_ratio=1.0` `save_baseline_actions=false` に揃えて、
  warm start の効果をより公平に比較するために実施。

## 2. 条件サマリ

| 実験 | batch_dir | imitation_matches | imitation_epochs | policy_ratio | save_baseline_actions | selfplay_matches | eval_matches | epochs |
|---|---|---:|---:|---:|---|---:|---:|---:|
| A | runs/20260305_stage1_full_flat_mlp_ppo_batch_b1764667 | - | - | 1.0 | false | 200 | 50 | 4 |
| B | runs/20260305_stage1_full_flat_mlp_imitation_then_ppo_batch_fc6e0e01 | 50 | 8 | 0.5 | true | 200 | 50 | 4 |
| C | runs/20260305_stage1_full_flat_mlp_imitation_then_ppo_batch_5bae3b83 | 25 | 4 | 0.5 | true | 200 | 50 | 4 |
| D | runs/20260305_stage1_full_flat_mlp_imitation_then_ppo_batch_292952f4 | 50 | 8 | 1.0 | false | 200 | 50 | 4 |
| E | runs/20260305_stage1_full_flat_mlp_imitation_then_ppo_batch_8990ada5 | 25 | 4 | 1.0 | false | 200 | 50 | 4 |

共通:
- seeds: 42-51 (10本)
- workers: selfplay=10, evaluation=10, imitation=10
- device: training=cuda, selfplay/eval=cpu
- 成功率: 全実験 10/10

## 3. 結果（after eval）

| 実験 | avg_rank (mean±std) | avg_score (mean±std) | win_rate (mean±std) | deal_in_rate (mean±std) |
|---|---:|---:|---:|---:|
| A | 3.700 ± 0.098 | -17508.6 ± 1427.6 | 0.001924 ± 0.002483 | 0.596958 ± 0.031258 |
| B | 3.580 ± 0.066 | -15589.4 ± 1338.9 | 0.016593 ± 0.010447 | 0.590760 ± 0.021869 |
| C | 3.618 ± 0.057 | -16126.2 ± 1101.5 | 0.010181 ± 0.004072 | 0.585317 ± 0.017619 |
| D | 3.602 ± 0.078 | -16232.0 ± 1263.5 | 0.015403 ± 0.009555 | 0.593226 ± 0.027895 |
| E | 3.604 ± 0.086 | -15593.8 ± 1318.5 | 0.019658 ± 0.006141 | 0.598091 ± 0.021092 |

## 4. 学習前後差分（eval_before → eval の平均）

| 実験 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| A | +0.008 | -338.0 | +0.000280 | +0.006778 |
| B | +0.046 | -688.6 | -0.017143 | -0.003248 |
| C | +0.162 | -2371.6 | -0.036688 | +0.013509 |
| D | +0.068 | -1331.2 | -0.018332 | -0.000783 |
| E | +0.148 | -1839.2 | -0.027210 | +0.026282 |

観察:
- warm start 系（B/C/D/E）は最終到達点は A より良い傾向。
- ただし run 内の PPO 段（eval_before→eval）は悪化寄りで、
  現状は「warm start が効いているが PPO 段の上積みが弱い」構図。

## 5. 時間比較（run.log 先頭〜末尾）

| 実験 | total sec (mean±std) | imitation sec | selfplay sec | eval_before sec | eval sec |
|---|---:|---:|---:|---:|---:|
| A | 157.13 ± 4.74 | - | 17.58 | 62.77 | 62.02 |
| B | 403.07 ± 15.46 | 77.53 | 182.68 | 62.94 | 63.06 |
| C | 374.74 ± 20.46 | 46.89 | 183.98 | 63.62 | 64.70 |
| D | 235.03 ± 7.05 | 75.93 | 16.88 | 63.04 | 60.51 |
| E | 204.55 ± 7.60 | 46.19 | 17.25 | 61.75 | 63.48 |

主要比較:
- B/A: 2.57x 時間
- C/A: 2.39x 時間
- D/A: 1.50x 時間
- E/A: 1.30x 時間
- E/D: 0.87x 時間（Dより約13%短い）

## 6. 追加実験D/Eを入れた結論

- 交絡を取り除いた D/E でも、A 比で最終指標改善傾向は維持。
- フル warm start（D）と軽量 warm start（E）の比較では、
  今回の10 seedでは E が時間面で有利、最終指標も同等以上の項目がある。
- ただし eval_before→eval が悪化寄りなのは D/E でも継続しており、
  PPO 段の学習設定改善は引き続き課題。

## 7. 懸念点（今回レポートに明記する事項）

1. 単席 eval のまま
- seat バイアスが残る。rotation eval での再確認が望ましい。

2. phase 時間は run.log 由来
- `summary.json` に phase duration がないため、後処理解析に依存。

3. CUDA 前提
- 今回は全 run で `training.device` は `resolved=cuda` だったが、
  環境依存のため実行前ヘルスチェックは推奨。

## 8. 次アクション提案

1. D/E 設定をベースに `rotation eval` を追加した比較（まず5 seed、その後10 seed）。
2. PPO 段の上積み改善を狙う小規模探索（例: lr/epochs/batch size を限定グリッド）。
3. `summary.json` への phase_duration 保存（運用性向上）。
