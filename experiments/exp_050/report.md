# Experiment Report: exp_050

作成日: 2026-03-15  
対象: [experiments/exp_050/runbook.md](/home/takeo1116/Git/majong-rl/experiments/exp_050/runbook.md)  
目的: `policy_anchor(kl, coef=0.5) + entropy=0.0` 条件を `20 seeds × 20 cycles` で再検証し、改善傾向を統計的に確認する

## 1. 実験概要

- 条件: 1条件（Aのみ）
- seeds: `42..61`（20 seeds）
- cycles: `20`
- eval: `rotation, num_matches=100`
- 主要設定:
- `training.policy_anchor.enabled=true`
- `training.policy_anchor.type=kl`
- `training.policy_anchor.coef=0.5`
- `training.entropy_coef=0.0`

## 2. 実行結果

- batch_dir: `runs/20260315_stage1_full_flat_mlp_imitation_then_ppo_batch_a271c3d0`
- success: `20/20`
- failure: `0`

## 3. 最終 after 指標

mean ± std（seed=20）

| 指標 | 値 |
|---|---:|
| avg_rank | `3.3736 ± 0.0398` |
| avg_score | `-12760.2 ± 471.8` |
| win_rate | `0.04977 ± 0.00402` |
| deal_in_rate | `0.57127 ± 0.00864` |

95% CI（aggregate）:
- avg_rank: `[3.3547, 3.3926]`
- avg_score: `[-12985.0, -12535.3]`

## 4. 各cycle内の PPO 差分（eval_before -> eval）

mean ± std（seed=20）

| 指標 | 値 |
|---|---:|
| Δavg_rank | `-0.0116 ± 0.0316` |
| Δavg_score | `+152.3 ± 373.0` |
| Δwin_rate | `+0.00124 ± 0.00365` |
| Δdeal_in_rate | `-0.00177 ± 0.00456` |

所見:
- 「各cycle内」の更新は平均で改善側（rank↓, score↑）。
- `exp_049` の5seed結果で見えていた傾向（PPO差分は改善）は、20seedでも再現。

## 5. cycle推移（after の時系列）

aggregate.cycles の seed平均。

| cycle | eval avg_rank | eval avg_score | eval_diff_avg_rank |
|---:|---:|---:|---:|
| 0 | 3.3672 | -12647.2 | +0.0198 |
| 1 | 3.3684 | -12662.2 | +0.0011 |
| 2 | 3.3678 | -12606.6 | -0.0006 |
| 3 | 3.3735 | -12740.5 | +0.0057 |
| 4 | 3.3729 | -12696.7 | -0.0006 |
| 5 | 3.3604 | -12605.9 | -0.0125 |
| 6 | 3.3745 | -12721.1 | +0.0141 |
| 7 | 3.3676 | -12686.9 | -0.0069 |
| 8 | 3.3723 | -12757.1 | +0.0046 |
| 9 | 3.3660 | -12662.8 | -0.0063 |
| 10 | 3.3724 | -12737.6 | +0.0064 |
| 11 | 3.3730 | -12742.3 | +0.0006 |
| 12 | 3.3727 | -12695.5 | -0.0002 |
| 13 | 3.3843 | -12809.5 | +0.0115 |
| 14 | 3.3789 | -12769.6 | -0.0054 |
| 15 | 3.3744 | -12808.8 | -0.0045 |
| 16 | 3.3830 | -12835.6 | +0.0086 |
| 17 | 3.3737 | -12842.1 | -0.0092 |
| 18 | 3.3853 | -12912.5 | +0.0115 |
| 19 | 3.3736 | -12760.2 | -0.0116 |

所見:
- 中盤〜後半で揺れが大きく、単調改善ではない。
- 最終cycle（19）は cycle18 からは回復するが、cycle0 よりは悪い。

## 6. imitation基準との比較（追加観察）

定義:
- imitation基準 = 各runの `cycle0.eval_before`。
- その基準に対して各cycleの `eval(after)` が上回るかを確認。

基準（seed平均）:
- imitation avg_rank: `3.3475`
- imitation avg_score: `-12499.0`

結果:
- 20cycleのうち、**平均で imitation を上回る cycle は 0**。
- 各cycleでの平均との差分（vs imitation）は常に:
- `Δrank > 0`（悪化）
- `Δscore < 0`（悪化）

代表値:
- 最も近い cycle: `cycle5`（vs imitation `Δrank=+0.0129`, `Δscore=-106.9`）
- 最も悪い cycle: `cycle18`（vs imitation `Δrank=+0.0377`, `Δscore=-413.5`）

## 7. 診断補足

- `anchor_kl_mean`（run別 learner_diag から集計）: `0.00801 ± 0.00069`
- `clip_fraction`（aggregate learner_diag）: `0.03673 ± 0.00781`
- `ratio_std`（aggregate learner_diag）: `0.06953 ± 0.00364`

所見:
- update自体は暴れておらず、anchor は効いている。
- それでも imitation 基準を継続的に超えないため、課題は「更新の安定化」より「データ分布ドリフトと目標整合」に残る。

## 8. 結論

1. `20 seeds × 20 cycles` の大規模検証で、`exp_049` の楽観を修正できた。  
2. この条件は「各cycle内の差分改善」は作れるが、「imitation基準を超える長期改善」は作れない。  
3. 次段では、anchor/entropy だけで押すより、方策データ分布の設計（オフポリシー混合や参照方策の扱い）を優先して見直すべき。
