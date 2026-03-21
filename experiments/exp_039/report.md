# Experiment Report: exp_039

作成日: 2026-03-13  
対象: `experiments/exp_039/runbook.md`  
目的: `exp_038 B` を基準に imitation データをさらに増やし、`eval_before` がまだ伸びるか、またその改善が PPO 後の最終性能にも波及するかを確認する

## 1. 実験概要

比較条件:
- A: baseline reference（`exp_038 B` を流用）
  - `selfplay.imitation_matches=50`
  - `training.imitation_epochs=8`
  - `selfplay.num_matches=200`
- B: heavier imitation only
  - `selfplay.imitation_matches=200`
  - `training.imitation_epochs=8`
  - `selfplay.num_matches=200`

共通固定（主要）:
- `model.hidden_dims=[512,256]`
- `model.policy_tower.enabled=true`
- `model.value_tower.enabled=true`
- `training.gae_lambda=0.90`
- `training.imitation_value_warmstart.coef=0.3`
- `training.batch_size=512`
- `training.epochs=2`
- `training.lr=0.0001`
- `training.exclude_post_riichi_discards.enabled=true`
- shaping: `mode=both`, `scale=0.01`
- seeds: `42,43,44,45,46`

batch:
- A reference: （ローカル run）
- B: （ローカル run）

B は `success_count = 5/5`。

## 2. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A ref (`exp_038 B`) | 3.3700 ± 0.0689 | -13191.2 ± 592.0 | 0.05266 ± 0.00875 | 0.57195 ± 0.01796 |
| B heavier imitation | 3.4317 ± 0.0802 | -13520.0 ± 1444.2 | 0.04899 ± 0.01227 | 0.58320 ± 0.00788 |

所見:
- B は after 指標で A を更新できなかった。
- `avg_rank`、`win_rate`、`deal_in_rate` が揃って悪化しており、今回の imitation 追加増量は最終成績にはマイナス。
- `avg_score` も悪化し、分散も大きくなっている。

## 3. `eval_before` と `eval_before -> eval`

`eval_before`（seed=5 mean ± std）:

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A ref | 3.3683 ± 0.0505 | -13105.5 ± 871.4 | 0.05472 ± 0.00851 | 0.57376 ± 0.01109 |
| B heavier imitation | 3.4083 ± 0.0543 | -13198.8 ± 1275.9 | 0.05201 ± 0.00981 | 0.57655 ± 0.00951 |

`eval_before -> eval` の delta:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| A ref | +0.0017 ± 0.0800 | -85.7 ± 1302.5 | -0.00206 ± 0.00966 | -0.00181 ± 0.01969 |
| B heavier imitation | +0.0233 ± 0.0715 | -321.2 ± 859.7 | -0.00301 ± 0.00617 | +0.00665 ± 0.00676 |

所見:
- 今回の最重要点は、**imitation を 50 -> 200 に増やしても `eval_before` は伸びなかった**こと。
- `avg_rank`、`avg_score`、`win_rate` の主要 3 指標はいずれも A より悪い。
- PPO 後悪化も A より大きく、`eval_before -> eval` も改善していない。
- したがって、今回の条件では「imitation をさらに増やせば PPO の壊れ方も小さくなる」という傾向は再現しなかった。

## 4. imitation 指標

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate | imitation value_loss |
|---|---:|---:|---:|
| A ref | 0.18241 ± 0.01190 | 0.65313 ± 0.00733 | 0.02380 ± 0.00182 |
| B heavier imitation | 0.22758 ± 0.00250 | 0.70182 ± 0.00293 | 0.02436 ± 0.00143 |

所見:
- teacher 追従は明確に改善している。
- 特に `teacher_top1_match_rate` と `teacher_best_set_hit_rate` の伸びは大きい。
- 一方で `imitation value_loss` はほぼ横ばいで、critic 初期値が大きく改善したとは言いにくい。
- **teacher にはより合うが、実対戦性能には結び付かない**という構図が今回の核心。

## 5. 主診断: 更新安定性と value

| 条件 | clip_fraction | ratio_std | value_error_mean | value_error_std | old_value_mean | new_value_mean |
|---|---:|---:|---:|---:|---:|---:|
| A ref | 0.13141 ± 0.02262 | 0.16094 ± 0.01820 | -0.00972 ± 0.00250 | 0.09071 ± 0.00350 | -0.18706 ± 0.02088 | -0.17703 ± 0.01839 |
| B heavier imitation | 0.13675 ± 0.00811 | 0.14887 ± 0.00393 | -0.01179 ± 0.00032 | 0.09807 ± 0.00538 | -0.21824 ± 0.00845 | -0.20591 ± 0.00835 |

所見:
- update 強度は A と大差なく、`clip_fraction` はむしろわずかに悪化。
- `ratio_std` は少し下がるが、after 改善には結び付いていない。
- critic は B の方が全体に悲観的で、`value_error_std` も悪化している。
- 今回の失敗は「policy imitation が悪化した」ではなく、**teacher への適合増加が value / PPO への橋渡しに失敗した**と読むのが自然。

## 6. 主診断: shanten_diag / turn_diag

advantage mean（seed=5 mean ± std）

### 6.1 shanten_diag

| 群 | A ref | B heavier imitation |
|---|---:|---:|
| improve | -0.03905 ± 0.00557 | -0.04068 ± 0.00688 |
| same | +0.03917 ± 0.00435 | +0.03627 ± 0.00406 |
| worsen | -0.06836 ± 0.01566 | -0.06718 ± 0.01038 |

### 6.2 turn_diag

| 群 | A ref | B heavier imitation |
|---|---:|---:|
| early | -0.74456 ± 0.05083 | -0.63949 ± 0.05801 |
| mid | -0.58537 ± 0.03857 | -0.52047 ± 0.02784 |
| late | +0.14478 ± 0.00947 | +0.12640 ± 0.00852 |

所見:
- `shanten_diag` はほぼ変わっていない。ここからは imitation 追加増量の主効果は読み取りにくい。
- `turn_diag` は B の方がむしろ自然化している。early/mid の負、late の正がいずれも少し縮んだ。
- それでも after は悪化しているので、今回も **群平均診断の改善だけでは採用判断できない**。

## 7. 立直後打牌除外と reward composition

### 7.1 立直後打牌除外
- A ref: `2969.8 ± 105.2`
- B heavier imitation: `3063.2 ± 119.1`

所見:
- imitation 増量に伴って除外件数もやや増えたが、差は限定的。
- 今回の差分の主因は exclusion ではなく imitation データ量そのものとみてよい。

### 7.2 reward composition（mean）
- A ref:
  - `point_delta = -0.000939`
  - `shanten_delta = +0.000033`
  - `total = -0.000906`
- B heavier imitation:
  - `point_delta = -0.001090`
  - `shanten_delta = +0.000030`
  - `total = -0.001060`

所見:
- sample-level reward 平均も B の方が悪い。
- 今回の重い imitation は、reward 面でも有利な自己対戦データを作れていない可能性がある。

## 8. seed 別の見え方

after の seed 別 `avg_rank`:
- A ref: `3.4167, 3.3167, 3.2917, 3.4583, 3.3667`
- B heavier imitation: `3.5417, 3.3667, 3.4417, 3.4667, 3.3417`

after の seed 別 `win_rate`:
- A ref: `0.05497, 0.04472, 0.06643, 0.05131, 0.04587`
- B heavier imitation: `0.05029, 0.04233, 0.03657, 0.04686, 0.06893`

所見:
- B は一部 seed でだけ良いのではなく、むしろ多くの seed で rank / win_rate が悪化している。
- したがって今回の悪化は「平均だけの偶然」ではなく、かなり系統的と見てよい。

## 9. 解釈

1. **現行特徴量 + モデルは、teacher imitation を増やせばまだ teacher に近づける**  
   これは `teacher_top1_match_rate` と `teacher_best_set_hit_rate` が明確に示している。

2. **しかし、その改善は PPO 前評価にも最終評価にも繋がらなかった**  
   今回は `eval_before` 自体が悪化しており、teacher 追従の増加が対戦強度の改善に変換されていない。

3. **`exp_038 B` にあった「imitation を増やすと PPO の壊れ方も小さくなる」傾向は、50 -> 200 では崩れた**  
   これは imitation に関しても sweet spot があることを示唆する。

4. **今の特徴量に対して、teacher 追従の追加改善はすでに過学習寄りに入っている可能性が高い**  
   つまり「teacher には合うが、麻雀の実力としては伸びない」領域に入ったと見るのが自然。

## 10. 結論

- **採用候補: A ref (`exp_038 B`) 維持**
- **不採用: B heavier imitation only**

今回の主結論:
- imitation を `50 -> 200` にさらに増やしても、`eval_before` は伸びず、after も悪化した。
- したがって、**現行特徴量 + モデルで imitation データ量をさらに積む方向は、少なくともこの設定では有効でない**。
- これは、特徴量改善へ進む根拠としてかなり強い。

## 11. 次アクション

1. 主系列 baseline は引き続き **`exp_038 B`** を維持する。
2. 次は PPO 調整や imitation 増量を続けるより、**特徴量改善** に進む優先度が高い。
3. もし現行特徴量系を続けるなら、追加 imitation は `200` まで増やすのではなく、`50` 近傍を上限とみなすべき。
