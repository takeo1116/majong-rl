# Experiment Report: exp_038

作成日: 2026-03-13  
対象: `experiments/exp_038/runbook.md`  
目的: `exp_037 D` を高表現力 baseline として、(1) imitation データ増量で `eval_before` が伸びるか、(2) post-fix 条件で shaping 設定を再評価すると after 指標が改善するかを同時に確認する

## 1. 実験概要

比較条件:
- A: baseline reference（`exp_037 D` を流用）
- B: imitation 増量
  - `selfplay.imitation_matches=50`
  - `training.imitation_epochs=8`
- C: shaping improve_only
  - `reward.shaping.shanten_delta.mode=improve_only`
- D: shaping scale 0.02
  - `reward.shaping.shanten_delta.scale=0.02`
- E: shaping improve_only + scale 0.02
  - `reward.shaping.shanten_delta.mode=improve_only`
  - `reward.shaping.shanten_delta.scale=0.02`

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
- seeds: `42,43,44,45,46`

batch:
- A reference: （ローカル run）
- B: （ローカル run）
- C: （ローカル run）
- D: （ローカル run）
- E: （ローカル run）

全新規条件 `success_count = 5/5`。

## 2. 通常評価

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A ref (`exp_037 D`) | 3.3917 ± 0.0601 | -13054.3 ± 1346.5 | 0.04619 ± 0.01231 | 0.57198 ± 0.00476 |
| B more imitation | 3.3700 ± 0.0689 | -13191.2 ± 592.0 | 0.05266 ± 0.00875 | 0.57195 ± 0.01796 |
| C improve_only | 3.3933 ± 0.0379 | -13314.2 ± 965.6 | 0.04558 ± 0.00799 | 0.57628 ± 0.00969 |
| D scale=0.02 | 3.4117 ± 0.0718 | -13555.2 ± 1604.3 | 0.04584 ± 0.01057 | 0.57685 ± 0.01820 |
| E improve_only + 0.02 | 3.3900 ± 0.0767 | -13387.5 ± 1099.1 | 0.04720 ± 0.01058 | 0.56925 ± 0.00618 |

所見:
- **B が最良**。
  - `avg_rank` は A を上回った
  - `win_rate` も改善
  - `deal_in_rate` はほぼ同等
  - `avg_score` はわずかに悪化だが分散は大きく縮小
- shaping 系（C/D/E）は総合では A を更新できなかった。
- D (`scale=0.02` 単独) は明確に悪い。

## 3. `eval_before` と `eval_before -> eval`

`eval_before`（seed=5 mean ± std）:

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A ref | 3.3667 ± 0.0705 | -12775.8 ± 1104.2 | 0.05158 ± 0.00367 | 0.57198 ± 0.00433 |
| B more imitation | 3.3683 ± 0.0452 | -13105.5 ± 779.4 | 0.05472 ± 0.00761 | 0.57376 ± 0.00992 |
| C improve_only | 3.3800 ± 0.0344 | -12852.8 ± 894.8 | 0.05103 ± 0.00389 | 0.57177 ± 0.00394 |
| D scale=0.02 | 3.3733 ± 0.0887 | -12815.5 ± 1159.8 | 0.05232 ± 0.00477 | 0.57179 ± 0.00384 |
| E improve_only + 0.02 | 3.3750 ± 0.0740 | -12935.7 ± 1123.0 | 0.05061 ± 0.00586 | 0.56989 ± 0.00462 |

`eval_before -> eval` の delta:

| 条件 | Δavg_rank | Δavg_score | Δwin_rate | Δdeal_in_rate |
|---|---:|---:|---:|---:|
| A ref | +0.0250 ± 0.0803 | -278.5 ± 1014.9 | -0.00539 ± 0.00981 | +0.00000 ± 0.00530 |
| B more imitation | +0.0017 ± 0.0716 | -85.7 ± 1165.0 | -0.00206 ± 0.00864 | -0.00181 ± 0.01761 |
| C improve_only | +0.0133 ± 0.0314 | -461.3 ± 435.3 | -0.00546 ± 0.00380 | +0.00451 ± 0.00666 |
| D scale=0.02 | +0.0383 ± 0.0670 | -739.7 ± 1054.9 | -0.00647 ± 0.00730 | +0.00506 ± 0.01343 |
| E improve_only + 0.02 | +0.0150 ± 0.0484 | -451.8 ± 397.0 | -0.00341 ± 0.00600 | -0.00064 ± 0.00376 |

所見:
- **B の主目的は達成**。`eval_before -> eval` の悪化幅は A より小さい。
- `eval_before` の `win_rate` は B が最良で、imitation 増量により imitation 段階の改善余地がまだあることが示唆された。
- shaping 系は `eval_before` を押し上げていない。PPO に入る前の方策品質改善には効いていない。

## 4. imitation 指標

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate | imitation value_loss |
|---|---:|---:|---:|
| A ref | 0.17578 ± 0.00536 | 0.61692 ± 0.00936 | 0.02900 ± 0.00341 |
| B more imitation | 0.18241 ± 0.01190 | 0.65313 ± 0.00733 | 0.02380 ± 0.00182 |
| C improve_only | 0.17571 ± 0.00564 | 0.61726 ± 0.00948 | 0.02694 ± 0.00320 |
| D scale=0.02 | 0.17556 ± 0.00508 | 0.61687 ± 0.00941 | 0.03144 ± 0.00339 |
| E improve_only + 0.02 | 0.17557 ± 0.00530 | 0.61702 ± 0.00896 | 0.02686 ± 0.00305 |

所見:
- B は imitation 指標が明確に改善している。
- 特に `teacher_best_set_hit_rate` の上昇が大きい。
- これは「現行特徴量 + モデルでも、imitation を増やせばまだ伸びる」ことのかなり強い証拠。
- shaping 条件変更は imitation 指標にはほぼ影響しない。これは当然だが、今回の目的上重要。

## 5. 主診断: 更新安定性

| 条件 | clip_fraction | ratio_std | value_error_mean |
|---|---:|---:|---:|
| A ref | 0.09096 ± 0.01418 | 0.12621 ± 0.00669 | -0.00609 ± 0.00183 |
| B more imitation | 0.13141 ± 0.02262 | 0.16094 ± 0.01820 | -0.00972 ± 0.00250 |
| C improve_only | 0.10645 ± 0.01092 | 0.13523 ± 0.00586 | -0.01393 ± 0.00131 |
| D scale=0.02 | 0.06571 ± 0.01636 | 0.11238 ± 0.00902 | -0.00695 ± 0.00168 |
| E improve_only + 0.02 | 0.08779 ± 0.01512 | 0.12491 ± 0.00823 | -0.02239 ± 0.00087 |

所見:
- B は `clip_fraction` / `ratio_std` が悪化している。
- それでも after 指標は改善しているので、今回は「少し強く動く update がむしろ有利」だったと読むべき。
- D は update 指標だけ見ると最も穏やかだが、通常評価は悪い。`exp_036` と同じく、綺麗な update がそのまま性能改善を意味しない例。

## 6. 主診断: shanten_diag

advantage mean（seed=5 mean ± std）

| 群 | A ref | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| improve | -0.02769 ± 0.00749 | -0.03906 ± 0.00498 | -0.03645 ± 0.00717 | +0.04096 ± 0.01561 | +0.02140 ± 0.01114 |
| same | +0.04902 ± 0.00416 | +0.03917 ± 0.00389 | +0.02944 ± 0.00529 | +0.05016 ± 0.00441 | +0.01330 ± 0.00711 |
| worsen | -0.13135 ± 0.00786 | -0.06836 ± 0.01401 | -0.04568 ± 0.01111 | -0.22112 ± 0.01051 | -0.05576 ± 0.01092 |

所見:
- shaping 条件を変えると `shanten_diag` は大きく動く。
- ただしその改善/悪化は after 指標と素直に対応していない。
- D/E では `improve` が正に転じるが、通常評価はむしろ良くない。ここでも「shanten 群平均の綺麗さだけでは採用判断できない」ことが再確認された。

## 7. 立直後打牌除外の実績

`post_riichi_exclusion.excluded_post_riichi_discards`:
- A ref: `2502.2 ± 161.2`
- B: `2969.8 ± 94.1`
- C: `2519.0 ± 168.2`
- D: `2497.6 ± 163.1`
- E: `2532.4 ± 149.3`

所見:
- B は imitation 試合数増加により除外件数も増えている。
- C/D/E は基準とほぼ同水準で、差分の主因は shaping 条件とみてよい。

## 8. 解釈

1. **imitation 増量は有望**  
   B は `eval_before` と after 指標の両方で前向き。  
   `teacher_best_set_hit_rate` も明確に上がっており、現行特徴量 + モデルでも imitation だけでまだ伸びる余地がある。  
   これは「PPO で伸びる余地がそもそもあるか」という問いに対して、かなり前向きな答えになっている。

2. **shaping は post-fix 条件で再評価価値はあったが、今回は採用に届かない**  
   C/D/E はそれぞれ診断上の変化はあるが、A を総合で更新できていない。  
   特に `scale=0.02` 単独は不採用。

3. **update の綺麗さより、まず imitation 段階の押し上げが効いている**  
   B は update 指標だけ見れば悪化しているが、性能は良い。  
   今回の主効果は PPO 調整ではなく、pre-PPO の方策品質向上にある。

4. **shanten 群平均の改善だけで採用判断してはいけない**  
   これは今回も繰り返し確認された。  
   D/E の `improve` 正転は見た目には良いが、通常評価は伸びなかった。

## 9. 結論

- **採用候補: B more imitation**
- **不採用: C improve_only, D scale=0.02, E improve_only + scale=0.02**

今回の主結論は 2 つ。

1. `exp_037 D` を基準にすると、**imitation 増量はまだ有効**。  
2. shaping は post-fix 条件で再評価しても、今回の 3 条件では baseline を更新できなかった。

したがって、次の主系列は
- `exp_037 D`
- `exp_038 B`
のどちらを新 baseline とするかを検討する段階に入った。

実務上は、`eval_before` と `win_rate` の改善が明確な **`exp_038 B` を次の有力 baseline 候補** とみなしてよい。

## 10. 次アクション

1. 次は `exp_038 B` を基準に、さらに imitation / self-play データ量を増やすかを検討する。
2. shaping 系は一旦停止してよい。少なくとも
   - `mode=improve_only`
   - `scale=0.02`
は主系列候補ではない。
3. もし shaping を再訪するなら、別方向の定義変更や schedule 変更が必要で、少なくとも今回の近傍条件を再試行する優先度は低い。
