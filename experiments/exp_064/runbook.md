# Experiment Runbook: exp_064

作成日: 2026-03-20  
目的: bugfix 後の新モデル + `rule-only PPO + policy_anchor(coef=0.5) + clip_epsilon=0.15` を固定し、`gamma` と `gae_lambda` を再sweepして、pre-bugfix 由来の horizon 仮説が post-bugfix baseline でも成立するかを再確認する。

## 1. 背景

- `CQ-0208` bugfix 後、imitation baseline と PPO baseline の解釈は大きく更新された。
- 現時点の暫定 PPO baseline は:
  - 新モデル
  - `rule-only PPO`
  - `policy_anchor.coef=0.5`
  - `clip_epsilon=0.15`
  - `gamma=0.50`
  - `gae_lambda=0.0`
  である。
- 一方で、`gamma` / `gae_lambda` の系統的な sweep は主に pre-bugfix 期の簡約条件で行われた。
- pre-bugfix では
  - `gae_lambda` は低いほど良い
  - `gamma=0.50` 付近が有望
  という知見があったが、bugfix 後 baseline にそのまま当てはまるかは未再確認である。
- いまの本題は `rule-only PPO` がなぜ序盤 peak 後に下がるかの切り分けなので、まずは **current baseline のまま horizon だけを切る** のが自然である。

したがって今回は、

**`rule-only + anchor(0.5) + clip(0.15)` を固定し、`gamma` と `gae_lambda` の影響だけを post-bugfix 条件で再確認する。**

## 2. 実験の問い

1. bugfix 後 baseline でも `gae_lambda=0.0` は依然として最良か
2. bugfix 後 baseline でも `gamma=0.50` は `0.75` より良いか
3. `gamma / gae` の違いは
   - final `avg_score`
   - best -> final drawdown
   - `teacher_best_set_hit_after`
   - `value_error_mean`
   - `late.value_error`
   にどう表れるか

## 3. 条件

- 条件数: 6
- seeds: `42,43,44`
- learner 形態: **すべて `rule-only PPO + anchor(0.5)`**
- 振るもの:
  - `training.gamma`
  - `training.gae_lambda`

条件一覧:

| 条件 | gamma | gae_lambda |
|---|---:|---:|
| A `g050_gae000` | `0.50` | `0.0` |
| B `g050_gae030` | `0.50` | `0.3` |
| C `g050_gae060` | `0.50` | `0.6` |
| D `g075_gae000` | `0.75` | `0.0` |
| E `g075_gae030` | `0.75` | `0.3` |
| F `g075_gae060` | `0.75` | `0.6` |

補足:
- `A g050_gae000` は current baseline と同一条件であり、既存 batch を共通基準として再利用できる。
- 今回はまず `0.95` を入れない。pre-bugfix では `0.95` が `0.75 / 0.50` より一段悪かったため、まずは有望帯の再確認を優先する。

## 4. 共通固定条件

- config:
  - `configs/stage1_full_flat_mlp_rule_only_anchor_ppo_baseline.yaml`
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
  - `training.value_loss_coef=0.25`
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

## 6. 見たい読み方

### ケース 1: `gae=0.0` が依然として最良

解釈:
- pre-bugfix の「短い horizon が有効」という知見は post-bugfix でも維持
- いまの主因は引き続き long-horizon credit assignment ではなく、その先の reward / weighting 側

### ケース 2: `gae=0.3` か `0.6` が改善

解釈:
- bugfix 後の stronger imitation / anchor 条件では、以前より少し長い horizon が効く
- 現在の `gae=0.0` は短すぎる可能性

### ケース 3: `gamma=0.75` が `0.50` を更新

解釈:
- pre-bugfix 由来で持ち込んでいた `gamma=0.50` は、current baseline では短すぎる
- bootstrap を少し長く戻した方がよい

### ケース 4: `gamma=0.50` が一貫して優位

解釈:
- current baseline でも `0.50` は妥当
- `gamma` は当面固定し、次は `value_loss_coef` / `policy_ratio` / weighting 側へ進む

### ケース 5: どれでも大差なし

解釈:
- horizon は二次要因
- 次段では `advantage` の作り方や baseline sample の扱い方を優先する

## 7. 成功条件

- 条件数 `6/6` 完走
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
  - `config.yaml -> training.gamma` が条件どおり
  - `config.yaml -> training.gae_lambda` が条件どおり
  - `phase_stats.learner.ppo_diag.policy_anchor.coef == 0.5`
  - `mixed_ppo.num_policy_samples == 0`
  - `mixed_ppo.num_baseline_samples > 0`

## 8. 想定所要時間

- `A g050_gae000` は current baseline と同一条件なので再利用可能
- 新規実行は `5` 条件
- 1 条件あたり `70〜90分` 程度
- 新規追加ぶん合計で `6〜8時間` 程度

## 9. 実行後にやること

1. `gamma x gae` の 2x3 比較 `report.md` を作る
2. post-bugfix baseline における `gamma` / `gae` の暫定固定値を決める
3. その後、
   - horizon がまだ本命なら微調整
   - そうでなければ `value_loss_coef` / `policy_ratio` / sample weighting に進む
