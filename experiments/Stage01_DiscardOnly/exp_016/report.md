# Experiment Report: exp_016

作成日: 2026-03-08  
対象: `experiments/exp_016/runbook.md`  
目的: `shanten_hint` on/off で imitation 教師再現度（top-1 / best-set）を比較し、特徴量の効き方を診断する。

## 1. 実験条件

- 共通 config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- phases: `["imitation","selfplay","eval"]`（imitation-only 相当）
- 共通主要値:
  - `imitation.num_workers=10`
  - `selfplay.imitation_matches=25`
  - `training.imitation_epochs=4`
  - `selfplay.num_matches=0`
  - `evaluation.mode=rotation`
  - `evaluation.num_matches=50`
  - `evaluation.num_workers=10`

比較条件:
- A (off): `feature_encoder.shanten_hint={"enabled":false}`
- B (on): `feature_encoder.shanten_hint={"enabled":true}`

batch_dir:
- A: （ローカル run）
- B: （ローカル run）

## 2. 実行結果サマリ

- A/B とも `success_count=5/5`
- A/B とも `aggregate.eval_mode=rotation`
- `teacher_best_set_status` は全 run で `available`
- `input_dim`:
  - A: 455
  - B: 489

## 3. 主評価（教師再現度）

mean ± std（seed=5）

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate |
|---|---:|---:|
| A (off) | 0.3067 ± 0.0049 | 0.5208 ± 0.0105 |
| B (on)  | 0.3134 ± 0.0073 | 0.5339 ± 0.0126 |

差分（B - A）:
- `teacher_top1_match_rate`: **+0.0066**（約 +2.16%）
- `teacher_best_set_hit_rate`: **+0.0131**（約 +2.51%）

補足:
- 全 run で `teacher_best_set_hit_rate >= teacher_top1_match_rate` を満たした。

## 4. 補助指標

mean ± std（seed=5）

| 条件 | imitation_loss | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|---:|
| A (off) | 2.1810 ± 0.0066 | 3.4190 ± 0.0591 | -13553.9 ± 951.7 | 0.04861 ± 0.00577 | 0.57683 ± 0.01800 |
| B (on)  | 2.1771 ± 0.0070 | 3.4070 ± 0.0584 | -13512.5 ± 1104.2 | 0.04688 ± 0.00713 | 0.57719 ± 0.01807 |

所見（補助）:
- `avg_rank` / `avg_score` は B がわずかに良い。
- `win_rate` は B がわずかに悪い。
- `deal_in_rate` はほぼ同等（B が微悪化）。

## 5. 診断結論

1. `shanten_hint=on` は、教師再現度指標（top-1 / best-set）を**小幅ながら一貫して改善**した。  
2. したがって「特徴量が全く使われていない」可能性は低い。  
3. 一方で改善幅は限定的で、最終対局指標の優位は明確でないため、`exp_015` で見えた warm start + PPO 側の悪化は、教師再現そのものより downstream（PPO interaction / shortcut 性 / 教師方策限界）を優先して疑うのが妥当。

## 6. 次アクション

1. teacher 再現が上がっても PPO 後で悪化する要因を切り分ける（PPO 側ノブ固定で `shanten_hint` 影響を再評価）。  
2. 必要なら小規模 sanity（極小データ過学習）で top-1 再現上限を確認し、モデル容量・学習設定不足との切り分けを行う。  
3. strict top-1 ではなく tie 構造を扱える学習目標の検討は次段候補とする。
