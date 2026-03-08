# Experiment Report: exp_017

作成日: 2026-03-08  
対象: `experiments/exp_017/runbook.md`

## 1. 実行サマリ

- Part1 A1（hint=off, strict）: `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_batch_0a08b8b3`
- Part1 B1（hint=on, strict）: `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_batch_010d325e`
- Part2 A2（hint=on, strict, tiny）: `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_2d594059`
- Part2 B2（hint=on, tie-aware, tiny）: `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_41dba41e`
- Part3 A3: **Part1-B1 を再利用（再実行なし）**
- Part3 B3（hint=on, tie-aware）: `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_batch_ee33120b`
- Part4 A4（hint=on, strict, PPO）: `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_batch_6252f3d0`
- Part4 B4（hint=on, tie-aware, PPO）: `runs/20260308_stage1_full_flat_mlp_imitation_then_ppo_batch_2aa13236`

注記:
- `exp_017_driver.py` は Part4-A4 後の検証で停止したため、Part4-B4 は手動実行。
- ただし A4/B4 とも batch は `5/5` 成功。

---

## 2. Part1: shanten_hint on/off（strict）

比較: A1 (off) vs B1 (on), seeds=5

### 2.1 教師再現指標（batch aggregate mean ± std）

| 条件 | teacher_top1_match_rate | teacher_best_set_hit_rate |
|---|---:|---:|
| A1 off/strict | 0.306742 | 0.520838 |
| B1 on/strict  | 0.313375 | 0.533919 |

所見:
- `shanten_hint=on` で teacher 指標は両方改善。

### 2.2 after 指標（batch aggregate mean ± std）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A1 off/strict | 3.419 ± 0.059 | -13553.9 ± 951.7 | 0.048611 ± 0.005774 | 0.576828 ± 0.017997 |
| B1 on/strict  | 3.407 ± 0.058 | -13512.5 ± 1104.2 | 0.046875 ± 0.007135 | 0.577193 ± 0.018066 |

所見:
- `avg_rank/avg_score` は on が微改善。
- `win_rate/deal_in_rate` は微悪化。

---

## 3. Part2: 極小過学習（seed=42）

### 3.1 結果

| 条件 | teacher_top1 | teacher_best_set | imitation loss | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| A2 on/strict | 0.5378 | 0.7121 | 1.8000 | 3.5500 | -14738.8 | 0.03242 | 0.57509 |
| B2 on/tie-aware | 0.3585 | 0.8490 | 0.8697 | 3.4125 | -13447.5 | 0.05076 | 0.56853 |

所見:
- tie-aware は `top1` を下げるが `best_set_hit` を大きく上げる。
- tiny 条件では after 指標も tie-aware が良い。
- 「strict の単一教師ラベル」と「best-set 構造」の目的差が明確に出ている。

---

## 4. Part3: imitation-only strict vs tie-aware（hint=on）

比較: A3 = Part1-B1再利用（on/strict） vs B3（on/tie-aware）, seeds=5

### 4.1 教師再現指標

| 条件 | teacher_top1 | teacher_best_set |
|---|---:|---:|
| A3 on/strict (B1再利用) | 0.313375 | 0.533919 |
| B3 on/tie-aware | 0.210928 | 0.657403 |

### 4.2 after 指標

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A3 on/strict | 3.407 ± 0.058 | -13512.5 ± 1104.2 | 0.046875 ± 0.007135 | 0.577193 ± 0.018066 |
| B3 on/tie-aware | 3.341 ± 0.075 | -12479.5 ± 744.1 | 0.052120 ± 0.004166 | 0.569952 ± 0.015789 |

所見:
- imitation-only では tie-aware が after 指標で明確に優位。
- ただし teacher 指標の内訳は「top1低下・best_set向上」で、目的関数差に整合。

---

## 5. Part4: warm start + PPO strict vs tie-aware（hint=on）

比較: A4 strict vs B4 tie-aware, seeds=5

### 5.1 eval_before -> eval の delta（run別平均 ± std）

| 条件 | Δavg_rank | Δavg_score | Δdeal_in_rate | Δwin_rate |
|---|---:|---:|---:|---:|
| A4 strict | +0.0550 ± 0.0874 | -869.7 ± 1274.1 | +0.001202 ± 0.007032 | -0.009229 ± 0.010016 |
| B4 tie-aware | +0.0550 ± 0.0454 | -713.9 ± 566.5 | +0.001603 ± 0.008522 | -0.012692 ± 0.009246 |

所見:
- 両条件とも平均では悪化方向（rank↑, score↓, win_rate↓）。
- tie-aware は `Δavg_score` の悪化幅が小さいが、`Δwin_rate` は strict より悪い。
- 主評価順で見ると「明確な勝者なし（ほぼ拮抗）」。

### 5.2 after 指標（batch aggregate mean ± std）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A4 strict | 3.443 ± 0.101 | -14025.1 ± 1464.5 | 0.042082 ± 0.006571 | 0.571125 ± 0.013986 |
| B4 tie-aware | 3.414 ± 0.070 | -13323.7 ± 988.5 | 0.041843 ± 0.011680 | 0.571935 ± 0.016378 |

所見:
- after は `avg_rank/avg_score` で tie-aware がやや良い。
- `win_rate/deal_in_rate` は strict が僅差で良い。
- ここでも決定的優位は出ていない。

---

## 6. 時間感（平均）

### 6.1 imitation-only（Part1/3）
- A1 total: 655.1s（imitation 117.3s / eval 537.9s）
- B1 total: 657.9s（imitation 118.0s / eval 539.9s）
- B3 total: 667.7s（imitation 118.8s / eval 548.8s）

### 6.2 PPO込み（Part4）
- A4 total: 1258.8s（imitation 119.2s / selfplay 41.2s / eval_before 540.5s / learner 15.0s / eval 542.9s）
- B4 total: 1285.1s（imitation 117.4s / selfplay 43.3s / eval_before 551.5s / learner 16.6s / eval 556.3s）

所見:
- 時間支配は `eval_before + eval`（rotation 50）で、learner差は小さい。

---

## 7. 総合結論

1. `shanten_hint` は教師再現には効いている（Part1で再確認）。  
2. imitation-only では tie-aware が有望（Part2/Part3で優位）。  
3. ただし warm start + PPO では優位が崩れ、strict/tie-aware はほぼ拮抗（Part4）。  
4. 現時点の本命仮説は「目的関数改善だけでは downstream 悪化を解消しきれない」。  

---

## 8. 次アクション案

1. Part4のみ seeds を 10 に拡張し、strict vs tie-aware の差を再判定。  
2. evalコストを抑えるため、PPO比較時は `evaluation.num_matches` を 30 へ下げたスクリーニング版を先行。  
3. tie-aware のまま PPO 側の更新強度（`training.epochs` または `training.lr`）を弱める小規模追試を追加。  
