# Experiment Runbook: exp_063

作成日: 2026-03-19  
目的: bugfix 後の新モデルを前提に、`rule-only PPO + policy_anchor(coef=0.5)` を固定し、`clip_epsilon` を振って peak 後の劣化が更新幅由来かどうかを切り分ける。

## 1. 背景

- `exp_061` では
  - A `rule_only_no_anchor`
  - B `rule_only_anchor`
  - C `actor_no_anchor`
  - D `actor_anchor`
  を比較し、**anchor の有無が最も大きい差分** であることが見えた。
- `exp_062` では `rule-only + anchor` を固定して `policy_anchor.coef` を振り、
  - `0.25`: peak は高いが戻りが大きい
  - `0.50`: final score が最良
  - `0.75`: teacher らしさ保持は強いが score はやや伸び切らない
  という結果になった。
- したがって現時点の主系列は **`rule-only + anchor(coef=0.5)`** とみなしてよい。
- 一方で `exp_062` でも依然として
  - 序盤から中盤で peak
  - 終盤で final が peak を下回る
  という挙動が 3 seeds で再現している。
- この形から、次に切るべき仮説は
  **「1 update あたりの PPO step がまだ強すぎるのではないか」**
  である。

そこで今回は、

**`clip_epsilon` を下げる/上げることで、peak の作り方と peak の保持がどう変わるか**

を確認する。

## 2. 実験の問い

1. `clip_epsilon=0.15` より小さい `0.10` は、best→final の戻り幅を減らせるか
2. `clip_epsilon=0.15` より大きい `0.20` は、peak を高くできるか、それとも drift を悪化させるか
3. `clip_fraction` / `ratio_std` の変化と final `avg_score` の関係はどうなるか

## 3. 条件

- 条件数: 3
- seeds: `42,43,44`
- learner 形態: **すべて `rule-only PPO + anchor(coef=0.5)`**
- 振るもの: `training.clip_epsilon`

条件一覧:

| 条件 | clip epsilon |
|---|---:|
| A `clip010` | `0.10` |
| B `clip015` | `0.15` |
| C `clip020` | `0.20` |

`B clip015` は、`exp_062` の最良設定 `anchor050` と同一条件であり、共通基準点として使う。

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
  - `training.policy_anchor.coef=0.5`
- optimization:
  - `training.lr=5e-5`
  - `training.epochs=1`
  - `training.batch_size=512`
  - `training.value_loss.type=mse`
  - `training.value_loss_coef=0.25`
  - `training.advantage_stabilization.clip=null`
  - `training.entropy_coef=0.0`
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
3. PPO step 診断
   - `clip_fraction`
   - `ratio_std`
   - `ratio_p90`
   - `ratio_p99`
4. teacher 診断
   - `teacher_agreement.action_match_rate_before/after`
   - `teacher_agreement.best_set_hit_rate_before/after`
5. mixed PPO 診断
   - `num_policy_samples == 0`
   - `num_baseline_samples > 0`

## 6. 見たい読み方

### ケース 1: `0.10` が最良

解釈:
- 現在の `0.15` は update がまだ強すぎる
- drift / over-update が peak 喪失の主因候補
- 次はさらに update 強度を詰める
  - 例: `clip_epsilon`
  - あるいは将来的に `lr`, `batch_size`（CQ-0209 実装後）

### ケース 2: `0.15` が最良

解釈:
- 現在の clip 強度は概ね妥当
- peak 後の劣化の主因は clip ではない
- 次は critic / sample weighting 側を見るべき

### ケース 3: `0.20` が最良

解釈:
- 現在は clip がやや強すぎて改善を止めている
- step を少し緩めた方が total return では得
- この場合、保持不足より改善不足の方が大きい

### ケース 4: 3 条件とも大差なし

解釈:
- `clip_epsilon` は二次要因
- 次は
  - `value_loss_coef`
  - `policy_ratio`
  - あるいは sample weighting / advantage quality
  の方が本命

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
  - `phase_stats.learner.ppo_diag.policy_anchor.coef == 0.5`
  - `config.yaml -> training.clip_epsilon` が条件どおり
  - `mixed_ppo.num_policy_samples == 0`
  - `mixed_ppo.num_baseline_samples > 0`

## 8. 想定所要時間

- `B clip015` は既存の `exp_062 anchor050` を共通基準として再利用可能
- 新規に回す必要があるのは
  - `A clip010`
  - `C clip020`
  の 2 条件
- 1 条件あたり `70〜90分` 程度
- 新規追加ぶん合計で `2.5〜3時間` 程度
- 3 条件フル再実行なら `4〜5時間` 程度

## 9. 実行後にやること

1. `clip_epsilon=0.10 / 0.15 / 0.20` を比較した `report.md` を作る
2. peak 維持の観点で最適 `clip_epsilon` を暫定決定する
3. その後、必要なら
   - `value_loss_coef`
   - `policy_ratio`
   - あるいは sample weighting / advantage quality
   の方向へ進む
