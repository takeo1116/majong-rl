# exp_057 runbook

最終更新: 2026-03-17  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `imitation=0 + policy_ratio=0.0 + mixed_ppo + gae=0.0 + shanten_delta.scale=0.003` の簡約問題で、`gamma=0.95 / 0.75 / 0.50` を `30 cycles` まで伸ばしたときに、本当に戻りが残るのか、peak がどこに来るのかを確認する

---

## 0. 実験の位置づけ

- 背景:
  - `exp_055` では `gamma=0.95, shanten_delta.scale=0.003` が、それ以前の簡約条件より良かった。
  - `exp_056` ではさらに `gamma=0.75 / 0.50 / 0.25` を試し、`0.75` と `0.50` が `0.95` より明確に良かった。
  - ただし、ここまでの比較はすべて `10 cycles` であり、「戻りが消えた」のか「peak が後ろにずれただけ」なのかはまだ判別できていない。
- 仮説:
  - `gamma=0.75` または `0.50` では、従来より plateau / 戻りがかなり減っている可能性がある。
  - ただし長く回すと再び悪化する可能性もあり、その場合はまだ「崩壊を遅らせた」段階に留まる。
- 方針:
  - `scale=0.003` を固定し、`gamma=0.95 / 0.75 / 0.50` の 3 条件だけを `30 cycles` で比較する。
  - これは最適値探索ではなく、**長期ダイナミクスの解釈実験** と位置づける。

## 1. 条件

- 条件数: 3
- seeds: `42,43,44`（3 seeds）
- cycles: `30`
- eval: `rotation, num_matches=100`

条件一覧:

- A: `gamma095_shanten0003_cycle30`
  - `gamma=0.95`
  - `shanten_delta.scale=0.003`
- B: `gamma075_shanten0003_cycle30`
  - `gamma=0.75`
  - `shanten_delta.scale=0.003`
- C: `gamma050_shanten0003_cycle30`
  - `gamma=0.50`
  - `shanten_delta.scale=0.003`

## 2. 共通固定（override）

- `experiment.phases=["selfplay","learner","eval"]`
- `feature_encoder.shanten_hint.enabled=true`
- `feature_encoder.discard_ukeire_hint.enabled=false`
- `feature_encoder.current_shanten.enabled=true`
- `feature_encoder.shape_hint.enabled=true`
- `feature_encoder.turn_context.enabled=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- `training.imitation_value_warmstart.enabled=true`
- `training.imitation_value_warmstart.coef=0.3`
- `training.exclude_post_riichi_discards.enabled=true`
- `training.value_loss.type=mse`
- `training.advantage_stabilization.clip=null`
- `training.policy_anchor.enabled=false`
- `training.entropy_coef=0.0`
- `reward.point_delta_scale=0.0001`
- `reward.shaping.shanten_delta.enabled=true`
- `reward.shaping.shanten_delta.scale=0.003`
- `reward.shaping.shanten_delta.mode=both`
- `reward.shaping.shanten_delta.schedule.type=linear_decay`
- `selfplay.num_matches=200`
- `selfplay.num_workers=10`
- `selfplay.policy_ratio=1.0`  # fallback
- `selfplay.save_baseline_actions=false`  # fallback
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
- `training.lr=5e-5`
- `training.epochs=1`
- `training.value_loss_coef=0.25`
- `training.batch_size=512`
- `training.gae_lambda=0.0`
- `training.clip_epsilon=0.15`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`
- `training.multi_cycle.enabled=true`
- `training.multi_cycle.num_cycles=30`
- `training.multi_cycle.selfplay_matches_per_cycle=200`
- `training.multi_cycle.eval_each_cycle=true`
- `training.rule_mix.enabled=true`
- `training.rule_mix.policy_ratio=0.0`
- `training.rule_mix.save_baseline_actions=true`
- `training.rule_mix_learner.enabled=true`
- `training.rule_mix_learner.ppo_mode=mixed`
- `training.rule_mix_learner.baseline_sample_weight=1.0`

条件ごとの追加 override:

- A:
  - `training.gamma=0.95`
- B:
  - `training.gamma=0.75`
- C:
  - `training.gamma=0.50`

## 3. 主評価

1. 初期基準（`cycle0.eval_before`）に対する after の改善量
   - `Δrank_vs_init`
   - `Δscore_vs_init`
2. peak の位置と大きさ
   - `best avg_rank cycle`
   - `best avg_score cycle`
   - `best rank gain`
   - `best score gain`
3. 戻りの有無
   - `best -> final` の落ち幅
   - `cycle 10 / 20 / 29` の after 推移
4. learner 診断
   - `value_error_mean`
   - `advantage_abs_mean_before_clip`
   - `clip_fraction`
   - `ratio_std`
   - `turn_diag.late.value_error`
5. teacher agreement 診断
   - `teacher_agreement.action_match_rate_before/after`
   - `teacher_agreement.best_set_hit_rate_before/after`
   - 長期で teacher に近づき続けるか

## 4. 成功判定

- 各条件の `batch_summary.json` で:
  - `success_count == 3`, `failure_count == 0`
- driver 全体で:
  - `completed == 3`, `failed == 0`
- 各 run で:
  - `summary.phase_stats.cycles` 長さ `30`
  - `cycles[*].learner_stages` が `mixed_ppo` のみ
  - `cycles[*].learner_diag.mixed_ppo.mixed_ppo_enabled == true`
  - `cycles[*].learner_diag.mixed_ppo.num_policy_samples == 0`
  - `cycles[*].learner_diag.mixed_ppo.num_baseline_samples > 0`
  - `cycles[*].learner_diag.teacher_agreement.enabled == true`

## 5. 判定基準（解釈実験）

- 本当に安定化している:
  - `gamma=0.75` または `0.50` で `cycle 20-29` でも大きな戻りが出ない
  - `best -> final` が小さいまま維持される
  - teacher agreement も維持または改善する
- peak が後ろにずれただけ:
  - `10 cycles` では安定に見えても、`15-25 cycles` で再び悪化する
  - `best cycle` が単に後ろへ移る
- `0.50` は短すぎる:
  - 前半は強いが、後半で `0.75` より先に失速する
- `0.95` でも意外に持つ:
  - `30 cycles` で差が縮まるなら、`gamma` 以外の因子も同程度に重要

## 6. 見たい結論

この runbook で決めたいのは次の一点。

> `gamma=0.50〜0.75` の改善は、本当に plateau / 戻りを消しているのか、それとも peak を後ろにずらしているだけなのか。

## 7. 想定所要時間

- 1条件 x 3 seeds x 30 cycles: おおむね `100〜130分`
- 3条件合計: `300〜390分`
- 余裕込み: `5〜6.5時間`

補足:
- 夜間実行前提なら `6時間前後` を目安にしてよい。
- learner 時間が伸びやすいので、`gamma=0.50` 条件は少し重く出る可能性がある。

## 8. 実行方針

- まず `30 cycles` の長期挙動を確認する
- ここで `0.75` か `0.50` が最後まで維持するなら、その条件を本命候補とする
- ここで両方とも戻るなら、「gamma は効いたがまだ本質解ではない」と整理し、次は別の target / value 設計へ進む
