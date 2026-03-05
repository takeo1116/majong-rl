# Stage1 Experiment Report (Runbook 1)

作成日時: 2026-03-05 00:21 JST  
Source Runbook: `experiments/exp_001/runbook.md`  
対象: Runbook 1（実験1〜5）  
方針: ベンチマーク（optional）は未実施

## 1. 実験一覧

| 実験 | 目的 | run_dir | seed | 構成 |
|---|---|---|---:|---|
| 実験1 | PPO直学習（小）+ parallel eval の導線確認 | `runs/20260304_stage1_full_flat_mlp_ppo_cb8da312` | 42 | selfplay=1 worker, eval=4 workers |
| 実験2 | PPO直学習（中）で傾向確認 | `runs/20260304_stage1_full_flat_mlp_ppo_7a8359c7` | 42 | selfplay=1 worker, eval=8 workers |
| 実験3 | warm start（小）の導線確認 | `runs/20260304_stage1_full_flat_mlp_imitation_then_ppo_114957ed` | 42 | selfplay=1 worker, eval=4 workers |
| 実験4 | warm start（中）比較 | `runs/20260304_stage1_full_flat_mlp_imitation_then_ppo_3efdf7f5` | 42 | selfplay=1 worker, eval=8 workers |
| 実験5 | parallel self-play + parallel eval の効果確認 | `runs/20260305_stage1_full_flat_mlp_ppo_4064d1a7` | 42 | selfplay=8 workers, eval=8 workers |

全実験で device 指定は共通:  
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

## 2. 実行結果サマリ

| 実験 | success | phase | selfplay steps | eval before avg_rank | eval after avg_rank | avg_score(after) | win_rate(after) | deal_in_rate(after) | 概算時間 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 実験1 | true | selfplay→learner→eval | 3,150 | 3.40 | 3.60 | -18,020 | 0.0000 | 0.5455 | 40s |
| 実験2 | true | selfplay→learner→eval | 31,308 | 3.65 | 3.65 | -16,030 | 0.0000 | 0.5887 | 109s |
| 実験3 | true | imitation→selfplay→learner→eval | 2,326 | 2.80 | 3.40 | -11,160 | 0.0270 | 0.5135 | 558s |
| 実験4 | true | imitation→selfplay→learner→eval | 27,356 | 3.45 | 3.55 | -14,650 | 0.0141 | 0.5986 | 841s |
| 実験5 | true | selfplay→learner→eval | 31,264 | 3.65 | 3.65 | -15,690 | 0.0141 | 0.6056 | 78s |

補足:
- 概算時間は `run.log` の先頭/末尾時刻差（秒単位）で算出
- 全 run で `summary.json.success=true`、`phase_status` は全 phase `success`

## 3. 主要な観察

1. 導線安定性
- 実験1〜5すべて完走し、Runbook 1 本編の目的（安定実行・比較可能な出力確保）は達成。

2. warm start の現状
- 今回条件では、実験3/4ともに `avg_rank` は学習後に悪化（+0.60, +0.10）。
- imitation 導線自体は成立しているが、効果優位は未確認。

3. 並列 self-play の速度効果
- 中規模条件で selfplay 1 worker（実験2）に対し selfplay 8 workers（実験5）は大幅短縮。
- 全体時間は eval/その他フェーズの比率があるため、self-play 単体ほどは短縮しない。

4. 結果解釈
- `evaluation.num_matches=20` は依然として分散が大きく、性能差の断定には不足。
- 現時点では「実験基盤の安定化と速度改善確認」が主成果。

## 4. 並列出力・再現情報の確認

- 実験5（parallel self-play）:
  - `phase_stats.selfplay.num_workers = 8`
  - `phase_stats.selfplay.shard_count = 8`
  - `phase_stats.selfplay.seed_strategy.method = derive_worker_seed + derive_match_seed`
- 実験1〜4（single self-play）:
  - `num_workers = 1`
  - `seed_strategy = null`

## 5. 結論

- Runbook 1 の必須実験（1〜5）は完了。
- Stage1 実験は、指定構成（GPU learner / CPU selfplay+eval）で安定して回せる状態。
- warm start の有効性は現時点で未確認（追加試行が必要）。
- 速度面では parallel self-play の導入効果が明確。

## 6. 次アクション（ベンチマーク保留前提）

1. 本日の結果を baseline として固定（比較時は worker数とmatch数を固定）。
2. 次回は seed を変えた再試行で再現性レンジを確認（同設定で2〜3本）。
3. optional ベンチマーク（worker=1/2/4/8/10）は別セッションで実施。
