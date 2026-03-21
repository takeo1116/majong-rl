# Experiment Runbook: exp_061

作成日: 2026-03-18  
目的: bugfix 後の新モデルを前提に、`imitation 1000 x 3 chunks` を warmstart として PPO を積み、`rule-only / actor` と `policy_anchor on/off` の 2x2 で、PPO がうまくいかない原因の切り分けに必要な基礎データを取得する。

## 1. 背景

- CQ-0208 修正後、新モデルの imitation-only は `1000 x 10 chunks` で
  - `teacher_best_set_hit_rate = 1.0`
  - `teacher_top1_match_rate = 0.7007`
  - `avg_score = +383.25`
  まで到達し、imitation 基盤は大きく改善した。
- 一方で、新モデル + `rule-only PPO` の sanity check では
  - imitation `1000 x 1`
  - PPO `200 x 10 cycles`
  の条件で、前半は改善するが後半で崩れることが確認された。
- また `strict_top1` imitation は `tie_aware_best_set` より悪く、PPO 不安定の主因を「imitation objective の best_set -> exact action 切り替え」だけで説明するのは難しくなった。

したがって次は、追加実装なしで切れる要因として

1. `policy_anchor` の有無  
2. `rule-only` と `actor` データ分布の違い  

をまず確認する。

## 2. 実験の問い

1. `policy_anchor` を入れると PPO の崩れは抑えられるか  
2. `rule-only PPO` ではなく `actor PPO` にすると改善するか  
3. `actor + anchor` の組み合わせは、単独より効くか  
4. 主因は
   - policy drift なのか
   - state distribution mismatch なのか
   - それでも説明しきれない surrogate / weighting 問題なのか
   をどこまで狭められるか

## 3. 条件

- 条件数: 4
- seeds: `42,43,44`（3 seeds）
- imitation:
  - `training.multi_chunk_imitation.enabled=true`
  - `training.multi_chunk_imitation.num_chunks=3`
  - `training.multi_chunk_imitation.imitation_matches_per_chunk=1000`
  - total imitation matches = `3000`
- PPO:
  - `training.multi_cycle.enabled=true`
  - `training.multi_cycle.num_cycles=30`
  - `training.multi_cycle.selfplay_matches_per_cycle=200`
- eval:
  - `rotation`
  - `num_matches=100`

条件一覧:

| 条件 | selfplay / learner | anchor |
|---|---|---|
| A `rule_only_no_anchor` | `rule-only PPO` | off |
| B `rule_only_anchor` | `rule-only PPO` | on |
| C `actor_no_anchor` | `actor PPO` | off |
| D `actor_anchor` | `actor PPO` | on |

意味:
- `rule-only PPO`
  - `training.rule_mix.enabled=true`
  - `training.rule_mix.policy_ratio=0.0`
  - `training.rule_mix_learner.enabled=true`
  - `training.rule_mix_learner.ppo_mode=mixed`
- `actor PPO`
  - `training.rule_mix.enabled=false`
  - learner は通常 PPO

## 4. 共通固定条件

- `experiment.phases=["imitation","selfplay","learner","eval"]`
- `feature_encoder.shanten_hint.enabled=true`
- `feature_encoder.discard_ukeire_hint.enabled=true`
- `feature_encoder.current_shanten.enabled=true`
- `feature_encoder.shape_hint.enabled=true`
- `feature_encoder.turn_context.enabled=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.3`
- `training.exclude_post_riichi_discards.enabled=true`
- `training.value_loss.type=mse`
- `training.advantage_stabilization.clip=null`
- `training.entropy_coef=0.0`
- `reward.point_delta_scale=0.0001`
- `reward.shaping.shanten_delta.enabled=true`
- `reward.shaping.shanten_delta.scale=0.003`
- `reward.shaping.shanten_delta.mode=both`
- `reward.shaping.shanten_delta.schedule.type=linear_decay`
- `imitation.num_workers=10`
- `training.imitation_epochs=8`
- `selfplay.imitation_matches=1000`
- `selfplay.num_matches=200`
- `selfplay.num_workers=10`
- `selfplay.policy_ratio=1.0`
- `selfplay.save_baseline_actions=false`
- `evaluation.mode=rotation`
- `evaluation.rotation_seats=[0,1,2,3]`
- `evaluation.num_matches=100`
- `evaluation.num_workers=10`
- `model.hidden_dims=[512,256]`
- `model.policy_tower.enabled=true`
- `model.policy_tower.hidden_dim=128`
- `model.value_tower.enabled=true`
- `model.value_tower.hidden_dim=128`
- `model.value_features.current_shanten.enabled=true`
- `model.policy_direct_hints.enabled=true`
- `model.policy_direct_hints.sources=["shanten_hint","discard_ukeire_hint"]`
- `model.policy_direct_hints.local_hidden_dim=16`
- `model.policy_direct_hints.tile_embedding_dim=4`
- `model.policy_direct_hints.context_gate.enabled=true`
- `training.lr=5e-5`
- `training.epochs=1`
- `training.value_loss_coef=0.25`
- `training.batch_size=512`
- `training.gamma=0.50`
- `training.gae_lambda=0.0`
- `training.clip_epsilon=0.15`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`
- `training.multi_cycle.eval_each_cycle=true`

## 5. 条件ごとの差分

### A: rule_only_no_anchor

- `training.rule_mix.enabled=true`
- `training.rule_mix.policy_ratio=0.0`
- `training.rule_mix.save_baseline_actions=true`
- `training.rule_mix_learner.enabled=true`
- `training.rule_mix_learner.ppo_mode=mixed`
- `training.rule_mix_learner.baseline_sample_weight=1.0`
- `training.policy_anchor.enabled=false`

### B: rule_only_anchor

- A に加えて
  - `training.policy_anchor.enabled=true`
  - `training.policy_anchor.type=kl`
  - `training.policy_anchor.coef=0.5`
  - `training.policy_anchor.reference=imitation_fixed`

### C: actor_no_anchor

- `training.rule_mix.enabled=false`
- `training.rule_mix_learner.enabled=false`
- `training.policy_anchor.enabled=false`

### D: actor_anchor

- C に加えて
  - `training.policy_anchor.enabled=true`
  - `training.policy_anchor.type=kl`
  - `training.policy_anchor.coef=0.5`
  - `training.policy_anchor.reference=imitation_fixed`

## 6. 主評価指標

1. imitation 基準（`cycle0.eval_before`）に対する after の改善量
   - `Δrank_vs_init`
   - `Δscore_vs_init`
2. peak と最終着地
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
5. rule_mix 条件では
   - `mixed_ppo.num_policy_samples`
   - `mixed_ppo.num_baseline_samples`

補足:
- `teacher_agreement` と `mixed_ppo.*` は主に A/B（rule-only）で見る。
- C/D（actor PPO）は通常 learner パスになるため、`learner_stages` は空でもよく、`learner_diag` の `policy_anchor` / `ratio_std` / `clip_fraction` / `value_error_mean` を主に見る。

## 7. 見たい読み方

### ケース 1: B だけ良い

解釈:
- 主因は policy drift
- imitation 基準から離れすぎるのを anchor が止めている

### ケース 2: C だけ良い

解釈:
- 主因は rule-only の state distribution mismatch
- actor 自身の分布を見せると改善する

### ケース 3: D が最良

解釈:
- actor 分布は必要
- ただし anchor なしでは drift が強い

### ケース 4: A/B/C/D 全部だめ

解釈:
- surrogate / weighting 側が本命
- 次は `advantage > 0` の rule sample だけ使うなど、rule 行動の中の良し悪しを切る方向へ進む

## 8. 成功条件

- 条件数 `4/4` 完走
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

## 9. 想定所要時間

- 1条件あたりおおむね `70〜100分`
- 4条件合計で `5〜7時間` 程度

補足:
- `actor PPO` は `rule-only` より learner 挙動が荒れる可能性があり、条件差で所要時間が多少ぶれる。
- 夜間実行前提としては現実的な範囲。

## 10. 実行後にやること

1. A/B/C/D を比較した `report.md` を作る  
2. drift 主因か distribution 主因かを暫定判定する  
3. それでも説明できない場合は、翌日に
   - `advantage > 0` の rule action だけ使う
   - あるいは `advantage-weighted imitation`
   の方向を検討する  
