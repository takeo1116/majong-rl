# Experiment Report: exp_048

作成日: 2026-03-15  
対象: [experiments/exp_048/runbook.md](/home/takeo1116/Git/majong-rl/experiments/exp_048/runbook.md)  
目的: `entropy_coef` を `0.001 / 0.0` に下げたときの傾向を、`5 seed × 3 cycle` で確認する

## 1. 実験概要

条件（5 seeds, 42..46）:
- A: `entropy_0001` (`entropy_coef=0.001`)
- B: `entropy_0000` (`entropy_coef=0.0`)

共通:
- `policy_anchor: kl, coef=0.5, reference=imitation_fixed`
- `multi_cycle.num_cycles=3`
- `selfplay_matches_per_cycle=200`

## 2. 実行結果

| 条件 | batch_dir | success |
|---|---|---:|
| A | `runs/20260315_stage1_full_flat_mlp_imitation_then_ppo_batch_fd6b34e8` | 5/5 |
| B | `runs/20260315_stage1_full_flat_mlp_imitation_then_ppo_batch_63b45c1e` | 5/5 |

driver完了: `completed=2, failed=0`

## 3. 主評価（after）

mean ± std（seed=5）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (`entropy=0.001`) | 3.3950 ± 0.0628 | -13489.0 ± 1246.0 | 0.04967 ± 0.00384 | 0.57559 ± 0.01187 |
| B (`entropy=0.0`) | **3.3833 ± 0.0441** | **-13140.3 ± 1092.6** | **0.04980 ± 0.00526** | **0.57364 ± 0.00877** |

所見:
- この5 seedでは、`entropy=0.0` が4指標すべてで優位。
- ただし seed数が小さいため、効果量は「傾向」として扱う。

## 4. eval_before -> eval の悪化幅

`delta = eval.after - eval.before`（avg_rank は小さいほど良い）

| 条件 | Δavg_rank mean ± std | Δavg_score mean ± std |
|---|---:|---:|
| A (`entropy=0.001`) | -0.00167 ± 0.04515 | -42.17 ± 298.38 |
| B (`entropy=0.0`) | **-0.00833 ± 0.04859** | **+268.67 ± 639.99** |

所見:
- 両条件とも rank悪化幅は0付近〜負側（悪化抑制）。
- `entropy=0.0` は score差分が正側で、最終cycleでの改善寄り。

## 5. cycle別推移（aggregate.cycles）

### A: entropy=0.001
- cycle0: rank `3.4000`, score `-13404.0`, Δrank `+0.0583`
- cycle1: rank `3.3967`, score `-13446.8`, Δrank `-0.0033`
- cycle2: rank `3.3950`, score `-13489.0`, Δrank `-0.0017`

### B: entropy=0.0
- cycle0: rank `3.3867`, score `-13336.3`, Δrank `+0.0467`
- cycle1: rank `3.3917`, score `-13409.0`, Δrank `+0.0050`
- cycle2: rank `3.3833`, score `-13140.3`, Δrank `-0.0083`

所見:
- どちらも cycle0→1 で一度悪化し、その後 cycle2 で回復。
- `entropy=0.0` の回復幅が大きい（特に score）。

## 6. policy_anchor / learner診断

| 条件 | anchor_kl_mean | clip_fraction | ratio_std |
|---|---:|---:|---:|
| A (`entropy=0.001`) | 0.00564 ± 0.00083 | 0.03490 ± 0.01241 | 0.06911 ± 0.00591 |
| B (`entropy=0.0`) | **0.00543 ± 0.00095** | 0.03532 ± 0.00361 | 0.06919 ± 0.00135 |

所見:
- anchor強度はほぼ同等（`anchor_kl_mean` 差は小さい）。
- 更新安定性指標（clip/ratio）もほぼ同水準。
- 差分は主に entropy の違い由来と解釈できる。

## 7. 参照比較（exp_047 B: entropy=0.01, coef=0.5, 20seed）

参照値:
- after `avg_rank=3.3858`, `avg_score=-12655.3`, `win_rate=0.05107`, `deal_in_rate=0.56882`
- `Δavg_rank=+0.0150`, `Δavg_score=-168.75`

注意:
- exp_047 B は 20 seed、exp_048 は 5 seedで統計的信頼度が異なる。
- ただし方向性として、entropy低下（0.001/0.0）は `Δavg_rank` を大きく改善（正→ほぼ0/負）している。

## 8. 結論

1. 小規模（5 seed）では、`entropy_coef=0.0` が最良傾向。  
2. 少なくとも「探索を減らすと壊れにくくなる」仮説は支持された。  
3. 次はこの結果を 20 seed で再検証する価値が高い。

## 9. 次アクション

1. `entropy=0.0` を第一候補として 20 seed 再検証（同じ3cycle条件）。  
2. 併せて `entropy=0.001` を再現確認用に残す。  
3. 20 seed で傾向が再現すれば、次段は `argmax主体 + 低確率sampling` 実装に進む。
