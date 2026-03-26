# Experiment Report: exp_047

作成日: 2026-03-15  
対象: `experiments/exp_047/runbook.md`  
目的: `policy_anchor (KL)` の係数 `0.3 / 0.5` を `3 cycle` で比較する

## 1. 実験概要

条件（20 seeds, 42..61）:
- A: `anchor_kl_coef_03_cycle3` (`coef=0.3`)
- B: `anchor_kl_coef_05_cycle3` (`coef=0.5`)

共通:
- `training.multi_cycle.num_cycles=3`
- `selfplay_matches_per_cycle=200`
- `turn_context=true`
- `entropy_coef=0.01`
- `policy_anchor.type=kl`, `reference=imitation_fixed`

## 2. 実行結果

| 条件 | batch_dir | success |
|---|---|---:|
| A (coef=0.3) | （ローカル run） | 20/20 |
| B (coef=0.5) | （ローカル run） | 20/20 |

driver完了: `completed=2, failed=0`

## 3. 主評価（after）

mean ± std（seed=20）

| 条件 | avg_rank | avg_score | win_rate | deal_in_rate |
|---|---:|---:|---:|---:|
| A (coef=0.3) | **3.3804 ± 0.0711** | -12973.8 ± 1003.9 | 0.04940 ± 0.00665 | 0.57082 ± 0.01399 |
| B (coef=0.5) | 3.3858 ± 0.0730 | **-12655.2 ± 846.8** | **0.05107 ± 0.00818** | **0.56882 ± 0.01353** |

所見:
- `avg_rank` は A(0.3) がわずかに良い。
- `avg_score / win_rate / deal_in_rate` は B(0.5) が良い。
- 係数増加で「rankは微悪化、score系は改善」というトレードオフ。

## 4. eval_before -> eval の悪化幅

`delta = eval.after - eval.before`（avg_rank は小さいほど良い）

| 条件 | Δavg_rank mean ± std | Δavg_score mean ± std |
|---|---:|---:|
| A (coef=0.3) | **-0.00042 ± 0.03722** | -199.63 ± 682.57 |
| B (coef=0.5) | +0.01500 ± 0.03976 | **-168.75 ± 495.49** |

所見:
- A(0.3) は rank悪化幅をほぼ0に抑制（最も良い）。
- B(0.5) は score悪化幅は小さいが、rank悪化幅はAより大きい。

## 5. cycle別推移（aggregate.cycles）

### A: coef=0.3
- cycle0: rank `3.3588`, score `-12547.8`, Δrank `+0.0013`
- cycle1: rank `3.3808`, score `-12774.1`, Δrank `+0.0221`
- cycle2: rank `3.3804`, score `-12973.8`, Δrank `-0.0004`

### B: coef=0.5
- cycle0: rank `3.3583`, score `-12570.8`, Δrank `+0.0008`
- cycle1: rank `3.3708`, score `-12486.5`, Δrank `+0.0125`
- cycle2: rank `3.3858`, score `-12655.2`, Δrank `+0.0150`

所見:
- どちらも cycle0 が最良で、cycleが進むと悪化。
- Aは cycle2 で `Δrank` が負側に戻るが、after rankの回復は弱い。
- Bは cycle進行で rankがじわ悪化。

## 6. policy_anchor / learner診断

| 条件 | anchor_kl_mean | clip_fraction | ratio_std |
|---|---:|---:|---:|
| A (coef=0.3) | 0.01109 ± 0.00122 | 0.04534 ± 0.00612 | 0.07449 ± 0.00280 |
| B (coef=0.5) | **0.00599 ± 0.00070** | **0.03379 ± 0.00751** | **0.06879 ± 0.00375** |

所見:
- 係数を上げると `anchor_kl_mean` は低下（参照方策への拘束が強い）。
- 同時に `clip_fraction` / `ratio_std` も低下し、更新はより保守的。

## 7. exp_046（coef=0.1, 1cycle）との比較

参考値（exp_046）:
- after `avg_rank=3.3596`, `avg_score=-12637.8`, `win_rate=0.05315`, `deal_in_rate=0.57174`
- `Δavg_rank=+0.00208`

注意:
- exp_046 は `1 cycle`、exp_047 は `3 cycle` で条件が完全一致しない。
- ただし傾向として、`coef=0.3/0.5` + 3cycle は rank面で `coef=0.1` を上回らなかった。

## 8. 結論

1. `coef` を 0.3 / 0.5 に上げても、`3 cycle` 条件では rank改善は見えない。  
2. `0.3` は rank悪化幅抑制（Δrank）で優位、`0.5` は score系指標で優位。  
3. 高係数化だけでは「cycle進行での性能悪化」を止め切れていない。

## 9. 次アクション

1. 予定どおり `entropy_coef` 低下実験（exp_048: 0.001 / 0.0, 5seed×3cycle）を実施。  
2. `coef=0.3` を採るなら rank重視、`coef=0.5` を採るなら score重視で目的指標を明確化。  
3. その後、必要なら `argmax主体 + 低確率sampling`（例 95/5）実装を検討。
