# Experiment Report: exp_044

作成日: 2026-03-14  
対象: [experiments/exp_044/runbook.md](/home/takeo1116/Git/majong-rl/experiments/exp_044/runbook.md)  
目的: `exp_043 E` 基準で `turn_context` / `huber` / `advantage clip` の寄与を切り分ける

## 1. 実験概要

条件（20 seeds, 42..61）:
- A: baseline (`turn_context=off`, `value_loss=mse`, `adv_clip=None`)
- B: turn_context_on (`turn_context=on`, `value_loss=mse`, `adv_clip=None`)
- C: huber_on (`turn_context=off`, `value_loss=huber(delta=1.0)`, `adv_clip=None`)
- D: adv_clip_on (`turn_context=off`, `value_loss=mse`, `adv_clip=2.0`)
- E: all_on (`turn_context=on`, `value_loss=huber(delta=1.0)`, `adv_clip=2.0`)

共通:
- `lr=5e-5, epochs=1, gae_lambda=0.85, clip_epsilon=0.15`
- `imitation/self-play = 200/200`
- `hidden=[512,256] + dual towers`
- `discard_ukeire_hint=false`

## 2. 実行結果

| 条件 | batch_dir | success |
|---|---|---:|
| A | `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_b56116b6` | 20/20 |
| B | `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_887808df` | 20/20 |
| C | `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_63d664a3` | 20/20 |
| D | `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_c490c231` | 20/20 |
| E | `runs/20260314_stage1_full_flat_mlp_imitation_then_ppo_batch_9cb2ced5` | 20/20 |

全条件完走（driver: `completed=5, failed=0`）。

## 3. 主評価（after）

mean ± std（seed=20）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A | 3.3779 ± 0.0586 | -13035.0 | 0.0516 | 0.5746 |
| B | **3.3708 ± 0.0772** | -12770.6 | 0.0521 | 0.5724 |
| C | 3.3742 ± 0.0572 | -12986.3 | 0.0516 | 0.5751 |
| D | 3.3879 ± 0.0699 | -13250.0 | 0.0499 | 0.5755 |
| E | 3.3729 ± 0.0912 | **-12710.0** | **0.0522** | **0.5711** |

所見:
- `avg_rank` 最良は **B (turn_context_on)**。
- `avg_score / win_rate / deal_in_rate` は **E (all_on)** が最良。
- `D (adv_clip_on 単独)` は主要指標で悪化し、今回の設定では不採用寄り。

## 4. eval_before -> eval の悪化幅

`Δavg_rank = eval.avg_rank - eval_before.avg_rank`（小さいほど良い）

| 条件 | eval_before avg_rank | eval avg_rank | Δavg_rank |
|---|---:|---:|---:|
| A | 3.3579 | 3.3779 | +0.0200 |
| B | 3.3575 | 3.3708 | **+0.0133** |
| C | 3.3579 | 3.3742 | +0.0163 |
| D | 3.3579 | 3.3879 | +0.0300 |
| E | 3.3575 | 3.3729 | +0.0154 |

所見:
- PPO後悪化幅の最小は **B**、次点 **E/C**。
- `adv_clip` 単独（D）は悪化幅を拡大した。

## 5. Learner 診断

`batch_summary.aggregate.learner_diag` より

| 条件 | clip_fraction | ratio_std | value_error_mean |
|---|---:|---:|---:|
| A | 0.1236 | 0.1058 | -0.00244 |
| B | **0.1203** | **0.1039** | -0.00225 |
| C | 0.1239 | 0.1058 | -0.00244 |
| D | 0.1276 | 0.1081 | -0.00244 |
| E | 0.1241 | 0.1060 | -0.00225 |

所見:
- 安定性指標（`clip_fraction`, `ratio_std`）も **B が最良**。
- D は主評価だけでなく診断でも悪化傾向。

## 6. Advantage clip 補足（手動集計）

`advantage_clip_*` は現状 `batch_summary.aggregate.learner_diag` に集約されないため、`runs[*].learner_diag` を手動集計。

| 条件 | advantage_clip_fraction (mean±std) | advantage_abs_mean_before | advantage_abs_mean_after |
|---|---:|---:|---:|
| D | 0.0527 ± 0.0026 | 0.5629 | 0.4996 |
| E | 0.0526 ± 0.0030 | 0.5586 | 0.4949 |

所見:
- clip 自体は意図どおり効いている（`abs_mean` を低下）。
- ただしこのクリップ量（2.0）は、今回の主評価改善には直結しなかった。

## 7. Imitation 指標

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate |
|---|---:|---:|
| A | 0.2048 | 0.6722 |
| B | 0.2035 | 0.6710 |
| C | 0.2048 | 0.6722 |
| D | 0.2048 | 0.6722 |
| E | 0.2035 | 0.6710 |

差は小さく、今回の差分は主に PPO 側の挙動差と解釈できる。

## 8. 実行時間

`phase_timing.total.mean`（秒/seed）

| 条件 | total_sec |
|---|---:|
| A | 71.05 |
| B | 76.72 |
| C | 73.79 |
| D | 67.16 |
| E | 66.83 |

注記:
- `turn_context_on`（B/E）は、A比較で速くなっていない（Bは特に遅い）。
- ただし全体としては高速化後の実用レンジ内。

## 9. 総合結論

1. 今回の最有力は **B (turn_context_on)**。  
   - `avg_rank` 最良
   - `Δavg_rank` 最良
   - PPO安定性指標も最良
2. **E (all_on)** は `avg_rank` 次点だが、`avg_score / win_rate / deal_in_rate` は最良。  
   - 目的指標の重み付け次第で有力候補
3. **D (adv_clip_on 単独)** は不採用寄り。  
   - clip は機能したが、対戦性能への寄与が見えない

## 10. 次アクション案

1. `turn_context` は採用候補として次実験の基準条件に昇格する。  
2. `adv_clip` は単独採用せず、値を再探索する場合は `turn_context` 併用条件に限定する。  
3. 長期挙動観測（multi-cycle）実装後に、B/E で「壊れてから回復するか」を時系列で再評価する。
