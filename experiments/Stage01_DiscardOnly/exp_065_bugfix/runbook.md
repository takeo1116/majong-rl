# Experiment Runbook: exp_065

作成日: 2026-03-21  
目的: bugfix 後の新モデル + `rule-only PPO + policy_anchor(coef=0.5) + clip_epsilon=0.15 + gamma=0.75 + gae_lambda=0.3` を固定し、`value_loss_coef` を振って、残っている peak 後の失速が critic / advantage quality にどの程度起因しているかを切り分ける。

## 1. 背景

- `exp_061` で `policy_anchor` の効果が明確に確認され、PPO の主問題は単純な no-anchor drift ではないところまで整理できた。
- `exp_062` で `policy_anchor.coef=0.5` が最良バランス、`exp_063` で `clip_epsilon=0.15` を暫定固定値として採用した。
- `exp_064` では `gamma x gae` を再確認し、現時点の最良条件は:
  - `gamma=0.75`
  - `gae_lambda=0.3`
  となった。
- ただし最良条件でも、PPO は
  - 序盤〜中盤で peak を作る
  - その後 final までに少し戻す
  という傾向が残っている。
- また `ppo_diag` の `shanten_diag` では、引き続き
  - `same` が正寄り
  - `improve` が強く負寄り
  という違和感が残っている。

したがって次の自然な問いは、

**「残っている失速は critic / advantage quality の問題か」**

である。

今回はこの問いに対して、**`value_loss_coef` だけを切る**。

## 2. 実験の問い

1. `value_loss_coef=0.25` は current best horizon 条件でも妥当か
2. critic の重みを弱める (`0.10`) と peak 後の失速は改善するか
3. critic の重みを強める (`0.50`) と advantage の質は改善するか、それとも悪化するか
4. `value_loss_coef` の違いは
   - final `avg_score`
   - best -> final drawdown
   - `teacher_best_set_hit_after`
   - `value_error_mean`
   - `turn_diag.late.value_error.mean`
   - `shanten_diag.improve/same/worsen.advantage.mean`
   にどう表れるか

## 3. 条件

- 条件数: 3
- seeds: `42,43,44`
- learner 形態: **すべて `rule-only PPO + anchor(0.5)`**
- 振るもの:
  - `training.value_loss_coef`
- 固定する current best horizon:
  - `training.gamma=0.75`
  - `training.gae_lambda=0.3`

条件一覧:

| 条件 | value_loss_coef | 備考 |
|---|---:|---|
| A `v010` | `0.10` | 新規実行 |
| B `v025` | `0.25` | `exp_064` の `g075_gae030` を共通基準として再利用 |
| C `v050` | `0.50` | 新規実行 |

補足:
- `B v025` は `exp_064` の best 条件 `g075_gae030` と完全に同一である。
- driver では `exp_064` の採用 batch をそのまま再利用しつつ、各 run の `config.yaml` で
  - `gamma=0.75`
  - `gae_lambda=0.3`
  - `value_loss_coef=0.25`
  を再確認する。
- したがって今回は、**新規実行は `v010` と `v050` の 2 条件**でよい。

## 4. 共通固定条件

- base config:
  - `configs/stage1_full_flat_mlp_rule_only_anchor_ppo_baseline.yaml`
- ただし全条件で以下を override して current best horizon に揃える:
  - `training.gamma=0.75`
  - `training.gae_lambda=0.3`
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
  - `training.policy_anchor.coef=0.5`
- optimization:
  - `training.lr=5e-5`
  - `training.epochs=1`
  - `training.batch_size=512`
  - `training.value_loss.type=mse`
  - `training.advantage_stabilization.clip=null`
  - `training.entropy_coef=0.0`
  - `training.clip_epsilon=0.15`
- reward:
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
4. critic / target 診断
   - `value_error_mean`
   - `turn_diag.late.value_error.mean`
   - `shanten_diag.improve/same/worsen.advantage.mean`
   - `shanten_diag.delta_t.mean`
5. PPO update 診断
   - `clip_fraction`
   - `ratio_std`
   - `advantage_positive_ratio`
   - `advantage_abs_mean_before_clip`

## 6. 見たい読み方

### ケース 1: `v010` が改善

解釈:
- critic が actor を押しすぎていた
- 現在の失速は value fitting の強さが主因に近い
- 本命は critic / advantage quality 側

### ケース 2: `v050` が改善

解釈:
- critic が弱すぎて advantage が粗かった
- 現在の問題は critic の過剰干渉ではなく、価値推定不足寄り

### ケース 3: `v025` が最良で差が小さい

解釈:
- `value_loss_coef` は二次要因
- 次は `policy_ratio` など distribution 側へ進む方が自然

### ケース 4: どの条件でも `improve < worsen` が強く残る

解釈:
- critic の重みよりも、reward / target / weighting 側の順位づけが本丸
- 次は `value_loss_coef` ではなく advantage の作り方を疑うべき

## 7. 成功条件

- 条件数 `3/3` 完了
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
  - `config.yaml -> training.gamma == 0.75`
  - `config.yaml -> training.gae_lambda == 0.3`
  - `config.yaml -> training.value_loss_coef` が条件どおり
  - `phase_stats.learner.ppo_diag.policy_anchor.coef == 0.5`
  - `mixed_ppo.num_policy_samples == 0`
  - `mixed_ppo.num_baseline_samples > 0`

## 8. 想定所要時間

- `B v025` は `exp_064` best 条件を再利用可能
- 新規実行は `2` 条件
- 1 条件あたり `70〜90分` 程度
- 新規追加ぶん合計で `2.5〜3.5時間` 程度

## 9. 実行後にやること

1. `value_loss_coef` 比較の `report.md` を作る
2. current best PPO baseline における critic 重みの暫定固定値を決める
3. その後、
   - `value_loss_coef` が効いたなら critic / advantage quality 側をさらに詰める
   - 差が小さければ `policy_ratio` sweep に進む
