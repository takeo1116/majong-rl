# Experiment Report: exp_052

作成日: 2026-03-16  
対象: [experiments/exp_052/runbook.md](/home/takeo1116/Git/majong-rl/experiments/exp_052/runbook.md)  
目的: `rule_mix(policy_ratio=0.75) + two-stage learner + policy_anchor(kl, coef=0.5) + entropy=0.0` を `20 seeds x 20 cycles` で再検証し、長期学習で改善が持続するかを確認する

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
- `training.multi_cycle.num_cycles=20`
- `training.multi_cycle.selfplay_matches_per_cycle=200`
- `training.multi_cycle.rule_mix.enabled=true`
- `training.multi_cycle.rule_mix.policy_ratio=0.75`
- `training.multi_cycle.rule_mix_learner.enabled=true`
- `training.multi_cycle.rule_mix_learner.ppo_mode=separated`

## 2. 実行結果

- batch_dir: `runs/20260315_stage1_full_flat_mlp_imitation_then_ppo_batch_cf5a257f`
- success: `20/20`
- failure: `0`

phase timing（seed平均）:
- imitation: `23.8 ± 1.3 sec`
- selfplay: `22.5 ± 1.4 sec`
- learner: `1409.6 ± 44.5 sec`
- total: `1456.0 ± 45.7 sec`

## 3. imitation 基準と最終 after

imitation 基準は各 run の `cycle0.eval_before`、最終 after は `cycle19.eval`。

| 指標 | imitation 基準 | 最終 after |
|---|---:|---:|
| avg_rank | `3.3475 ± 0.0328` | `3.3689 ± 0.0401` |
| avg_score | `-12499.0 ± 386.2` | `-12740.9 ± 493.4` |

95% CI:
- imitation avg_rank: `[3.3331, 3.3619]`
- imitation avg_score: `[-12668.2, -12329.7]`
- final avg_rank: `[3.3513, 3.3865]`
- final avg_score: `[-12957.2, -12524.7]`

所見:
- 最終着地は imitation 基準より悪い。
- final cycle の内部差分もほぼ横ばいで、`Δavg_rank=+0.0008 ± 0.0399`, `Δavg_score=-45.3 ± 384.2`。
- 最終 cycle で imitation 基準を上回った seed 数は `avg_rank: 6/20`, `avg_score: 7/20`, 両方同時: `6/20`。

## 4. cycle 推移

seed平均の after 指標と、同cycle内 `eval_before -> eval` 差分。

| cycle | after avg_rank | after avg_score | cycle内 Δavg_rank | cycle内 Δavg_score |
|---:|---:|---:|---:|---:|
| 0 | `3.3628` | `-12621.1` | `+0.0153` | `-122.1` |
| 1 | `3.3555` | `-12478.3` | `-0.0073` | `+142.8` |
| 5 | `3.3612` | `-12562.9` | `+0.0021` | `+0.2` |
| 10 | `3.3690` | `-12736.5` | `+0.0020` | `+82.9` |
| 19 | `3.3689` | `-12740.9` | `+0.0008` | `-45.3` |

補足:
- 平均最良 cycle は `cycle 1`。
- best mean after:
- avg_rank: `3.3555`（cycle 1）
- avg_score: `-12478.3`（cycle 1）
- worst mean after:
- avg_rank: `3.3805`（cycle 12）
- avg_score: `-12773.0`（cycle 16）

所見:
- cycle 1 で一度かなり良い位置まで行く。
- ただしその後は改善が蓄積せず、後半で再び沈む。
- 20 cycle 全体平均の内部差分も `Δavg_rank=+0.0011`, `Δavg_score=-12.1` で、PPO が平均的に押し上げているとは言いにくい。

## 5. imitation 基準との比較

各cycleの after を、各 seed の imitation 基準（`cycle0.eval_before`）と比較した。

| cycle | rank改善 seed数 | score改善 seed数 | rank/score 両方改善 |
|---:|---:|---:|---:|
| 0 | `6/20` | `8/20` | `5/20` |
| 1 | `7/20` | `10/20` | `6/20` |
| 5 | `5/20` | `10/20` | `4/20` |
| 10 | `7/20` | `5/20` | `4/20` |
| 19 | `6/20` | `7/20` | `6/20` |

重要点:
- `20 cycle` の中で、seed平均 `avg_rank` が imitation 基準を上回った cycle は `0`。
- seed平均 `avg_score` が imitation 基準を上回った cycle は `cycle 1` のみ。
- つまり「一時的に score は押し上がるが、imitation を安定して超える long-run 改善にはつながらない」という形。

## 6. `exp_050` / `exp_051` との比較

参照:
- [experiments/exp_050/report.md](/home/takeo1116/Git/majong-rl/experiments/exp_050/report.md)
- `exp_051` は 5 seeds x 10 cycles の pilot（report 未作成）

| 条件 | seeds x cycles | imitation 基準 avg_rank | best after avg_rank | best after avg_score | final avg_rank | final avg_score |
|---|---:|---:|---:|---:|---:|---:|
| exp_050 anchor only | `20 x 20` | `3.3475` | `3.3604` | `-12605.9` | `3.3736` | `-12760.2` |
| exp_051 rule_mix pilot | `5 x 10` | `3.3410` | `3.3280` | `-12287.8` | `3.3530` | `-12650.3` |
| exp_052 rule_mix full | `20 x 20` | `3.3475` | `3.3555` | `-12478.3` | `3.3689` | `-12740.9` |

所見:
- `exp_050` と比べると、`exp_052` は best/final ともに少し良い。
- つまり rule_mix + two-stage learner は、anchor only よりは悪くない。
- ただし `exp_052` でも imitation 基準超えは安定せず、問題の本質はまだ解けていない。
- `exp_051` の 5 seed pilot は比較的楽観的だったが、20 seed に広げるとその強さはかなり薄まった。

## 7. 診断補足

final cycle（aggregate / seed平均）:
- actor_type_counts: `policy 89071`, `baseline 28929`
- `clip_fraction = 0.0635 ± 0.0051`
- `ratio_std = 0.0807 ± 0.0023`
- `advantage_abs_mean_before_clip = 0.4344 ± 0.0061`
- `anchor_kl_mean = 0.02548 ± 0.00193`

所見:
- 学習は数値的には安定している。
- policy/baseline の混合比も大きく崩れていない。
- それでも長期改善にならないため、失敗要因は「発散」より「学習信号の質と方向」にある可能性が高い。

## 8. 結論

1. `rule_mix(policy_ratio=0.75) + two-stage learner` は、`anchor only` よりは少し良い。  
2. ただし `20 seeds x 20 cycles` では、imitation 基準を安定して超える長期改善は確認できなかった。  
3. 最良点は早期（cycle 1）にあり、その後は改善が蓄積せず再び悪化する。  
4. 次段は「更新をさらに穏やかにする」より、rule データの扱い自体を変える方向が本命。今回の結果は、その判断材料として十分に強い。
