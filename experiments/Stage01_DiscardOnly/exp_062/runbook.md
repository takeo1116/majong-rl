# Experiment Runbook: exp_062

作成日: 2026-03-18  
目的: bugfix 後の新モデルを前提に、`rule-only PPO + policy_anchor` を主軸として、`policy_anchor.coef` を振って peak の維持性能を比較する。

## 1. 背景

- `exp_061` では、4 条件比較の結果
  - A `rule_only_no_anchor`
  - B `rule_only_anchor`
  - C `actor_no_anchor`
  - D `actor_anchor`
  のうち、**B `rule_only_anchor`** が最良だった。
- 同じ `rule-only` でも
  - anchor なし A は前半で大きく伸びるが後半で崩れる
  - anchor あり B は 30 cycle 後も高い性能を維持する
  ことから、現時点では **policy drift が主因候補** と考えられる。
- 一方で、B も
  - peak `avg_score ≈ 2803.6`
  - final `avg_score ≈ 1943.0`
  であり、まだ peak を完全には保持できていない。
- また、`D actor_anchor` も B にかなり近く、anchor 下での `rule-only` / `actor` の優劣はまだ断定しづらい。
- そのため今回は、より解釈しやすく、かつ `exp_061` で最良だった **`rule-only + anchor`** を固定し、anchor 強度だけを切る。

したがって次は、

**`policy_anchor.coef=0.5` が強すぎるのか、弱すぎるのか、ちょうどよいのか**

を切り分ける。

## 2. 実験の問い

1. `policy_anchor.coef=0.5` より弱い `0.25` は、peak をより高く保てるか
2. `policy_anchor.coef=0.5` より強い `0.75` は、崩れをさらに抑えられるか
3. 係数変更で、
   - final `avg_score`
   - best→final の戻り幅
   - `best_set_hit` の保持
   がどう変わるか

## 3. 条件

- 条件数: 3
- seeds: `42,43,44`
- learner 形態: **すべて `rule-only PPO + anchor`**
- 振るもの: `training.policy_anchor.coef`

条件一覧:

| 条件 | anchor coef |
|---|---:|
| A `anchor025` | `0.25` |
| B `anchor050` | `0.50` |
| C `anchor075` | `0.75` |

## 4. 共通固定条件

- `experiment.phases=["imitation","selfplay","learner","eval"]`
- 新モデル:
  - `model.policy_direct_hints.enabled=true`
  - `model.policy_direct_hints.sources=["shanten_hint","discard_ukeire_hint"]`
  - `model.policy_direct_hints.local_hidden_dim=16`
  - `model.policy_direct_hints.tile_embedding_dim=4`
  - `model.policy_direct_hints.context_gate.enabled=true`
- feature:
  - `feature_encoder.shanten_hint.enabled=true`
  - `feature_encoder.discard_ukeire_hint.enabled=true`
  - `feature_encoder.current_shanten.enabled=true`
  - `feature_encoder.shape_hint.enabled=true`
  - `feature_encoder.turn_context.enabled=true`
- imitation:
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.3`
  - `training.multi_chunk_imitation.enabled=true`
  - `training.multi_chunk_imitation.num_chunks=3`
  - `training.multi_chunk_imitation.imitation_matches_per_chunk=1000`
  - total imitation matches = `3000`
  - `training.imitation_epochs=8`
  - `imitation.num_workers=10`
- PPO:
  - `training.rule_mix.enabled=true`
  - `training.rule_mix.policy_ratio=0.0`
  - `training.rule_mix.save_baseline_actions=true`
  - `training.rule_mix_learner.enabled=true`
  - `training.rule_mix_learner.ppo_mode=mixed`
  - `training.rule_mix_learner.baseline_sample_weight=1.0`
  - `training.policy_anchor.enabled=true`
  - `training.policy_anchor.type=kl`
  - `training.policy_anchor.reference=imitation_fixed`
- optimization:
  - `training.lr=5e-5`
  - `training.epochs=1`
  - `training.batch_size=512`
  - `training.value_loss.type=mse`
  - `training.value_loss_coef=0.25`
  - `training.advantage_stabilization.clip=null`
  - `training.entropy_coef=0.0`
  - `training.clip_epsilon=0.15`
- reward / target:
  - `training.gamma=0.50`
  - `training.gae_lambda=0.0`
  - `reward.point_delta_scale=0.0001`
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.003`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
- selfplay / cycle:
  - `selfplay.imitation_matches=1000`
  - `selfplay.num_matches=200`
  - `selfplay.num_workers=10`
  - `selfplay.policy_ratio=1.0`
  - `selfplay.save_baseline_actions=false`
  - `training.multi_cycle.enabled=true`
  - `training.multi_cycle.num_cycles=30`
  - `training.multi_cycle.selfplay_matches_per_cycle=200`
  - `training.multi_cycle.eval_each_cycle=true`
- eval:
  - `evaluation.mode=rotation`
  - `evaluation.rotation_seats=[0,1,2,3]`
  - `evaluation.num_matches=100`
  - `evaluation.num_workers=10`
- device:
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`

## 5. 主評価指標

1. final 指標
   - `avg_rank`
   - `avg_score`
   - `win_rate`
   - `deal_in_rate`
2. peak 保持
   - `best avg_score cycle`
   - `best avg_score`
   - `final avg_score`
   - `best -> final` 戻り幅
3. teacher 診断
   - `teacher_agreement.action_match_rate_before/after`
   - `teacher_agreement.best_set_hit_rate_before/after`
4. learner 診断
   - `clip_fraction`
   - `ratio_std`
   - `value_error_mean`
   - `turn_diag.late.value_error`
5. mixed PPO 診断
   - `num_policy_samples == 0`
   - `num_baseline_samples > 0`

## 6. 見たい読み方

### ケース 1: `0.25` が最良

解釈:
- `0.5` は縛りすぎ
- drift は抑えたいが、改善も止めていた

### ケース 2: `0.5` が最良

解釈:
- 現在の anchor 強度は概ね妥当
- 次は anchor 以外の要因を見るべき

### ケース 3: `0.75` が最良

解釈:
- まだ drift を抑え切れていない
- より強い制約が必要

### ケース 4: 3 条件とも大差なし

解釈:
- anchor 強度より、sample weighting / surrogate 側が本命
- 次は baseline sample の良し悪しを切る方向へ進む

## 7. 成功条件

- 条件数 `3/3` 完走
- `failed == 0`
- 各条件で:
  - `success_count == 3`
  - `failure_count == 0`
- 各 run で:
  - imitation / selfplay / learner / eval が `success`
  - `summary.phase_stats.cycles` 長さ `30`
  - `summary.phase_stats.imitation.multi_chunk_imitation.enabled == true`
  - `num_chunks == 3`
  - `sum(chunks[*].num_matches) == 3000`
  - `phase_stats.learner.ppo_diag.policy_anchor.coef` が条件どおり
  - `mixed_ppo.num_policy_samples == 0`
  - `mixed_ppo.num_baseline_samples > 0`

## 8. 想定所要時間

- 1条件あたり `70〜90分` 程度
- 3条件合計で `4〜5時間` 程度

## 9. 実行後にやること

1. `coef=0.25 / 0.50 / 0.75` を比較した `report.md` を作る
2. peak 保持の観点で最適係数を暫定決定する
3. その後、必要なら
   - `batch_size`
   - baseline sample weighting
   - `advantage > 0` のみ利用
   の方向へ進む
