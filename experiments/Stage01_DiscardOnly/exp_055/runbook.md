# exp_055 runbook

最終更新: 2026-03-16  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: `imitation=0 + policy_ratio=0.0 + mixed_ppo + gae=0.0` の簡約問題で、残っている「peak 後の戻り」が `gamma` 起因か `shanten shaping` 起因かを切り分ける

---

## 0. 実験の位置づけ

- 背景:
  - `exp_053` と `exp_054` により、現行の簡約問題では `lr / clip / batch` よりも advantage / target horizon 側の影響が強いことが分かった。
  - とくに `gae=0.0` は `0.3 / 0.6 / 0.7 / 0.85` より良く、長い horizon の credit assignment が主ノイズ源である可能性が強い。
  - ただし `gae=0.0` でも `cycle 3-5` で peak を打った後に少し戻る挙動は残っている。
- 仮説:
  - 戻りの原因はまだ残っている bootstrap horizon (`gamma`) か、短期報酬を歪めている `shanten shaping` のどちらか、あるいは両方である。
  - `gae=0.0` を固定して `gamma` と `shanten shaping` を同時に振れば、残差の主因をかなり直接に切り分けられる。
- 方針:
  - 条件数を `2 x 3 = 6` に抑え、`gamma` 2水準と `shanten shaping scale` 3水準の最小 full-factorial を組む。
  - これにより interaction を見つつ、全体を `3〜4時間` に収める。

## 1. 条件

- 条件数: 6
- seeds: `42,43,44`（3 seeds）
- cycles: `10`
- eval: `rotation, num_matches=100`

条件一覧:

- A: `gamma099_shanten0010`
  - `gamma=0.99`, `shanten_delta.scale=0.01`
- B: `gamma095_shanten0010`
  - `gamma=0.95`, `shanten_delta.scale=0.01`
- C: `gamma099_shanten0003`
  - `gamma=0.99`, `shanten_delta.scale=0.003`
- D: `gamma095_shanten0003`
  - `gamma=0.95`, `shanten_delta.scale=0.003`
- E: `gamma099_shanten0000`
  - `gamma=0.99`, `shanten_delta.scale=0.0`
- F: `gamma095_shanten0000`
  - `gamma=0.95`, `shanten_delta.scale=0.0`

補足:

- `gamma=0.95` は、`gae=0.0` の上でさらに future bootstrap を短くする中庸条件として採用する。
- `shanten_delta.scale=0.0` は `enabled=true` のまま scale のみを 0 にし、設定差分を最小化する。
- `gamma=0.90` は今回あえて入れない。`0.95` で差が強く出た場合に次段で追加する。

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
- `training.multi_cycle.num_cycles=10`
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
  - `training.gamma=0.99`
  - `reward.shaping.shanten_delta.scale=0.01`
- B:
  - `training.gamma=0.95`
  - `reward.shaping.shanten_delta.scale=0.01`
- C:
  - `training.gamma=0.99`
  - `reward.shaping.shanten_delta.scale=0.003`
- D:
  - `training.gamma=0.95`
  - `reward.shaping.shanten_delta.scale=0.003`
- E:
  - `training.gamma=0.99`
  - `reward.shaping.shanten_delta.scale=0.0`
- F:
  - `training.gamma=0.95`
  - `reward.shaping.shanten_delta.scale=0.0`

## 3. 主評価

1. 初期基準（`cycle0.eval_before`）に対する after の改善量
   - `Δrank_vs_init`
   - `Δscore_vs_init`
2. peak の位置と大きさ
   - `best avg_rank cycle`
   - `best avg_score cycle`
   - `best rank gain`
   - `best score gain`
3. 崩れ方
   - `best -> final` の落ち幅
4. learner 診断
   - `value_error_mean`
   - `advantage_abs_mean_before_clip`
   - `clip_fraction`
   - `ratio_std`
   - `turn_diag.late.value_error`
5. teacher agreement 診断
   - `teacher_agreement.action_match_rate_before/after`
   - `teacher_agreement.best_set_hit_rate_before/after`（利用可能なら）
   - `before -> after` で teacher に近づくか

## 4. 成功判定

- 各条件の `batch_summary.json` で:
  - `success_count == 3`, `failure_count == 0`
- driver 全体で:
  - `completed == 6`, `failed == 0`
- 各 run で:
  - `summary.phase_stats.cycles` 長さ `10`
  - `cycles[*].learner_stages` が `mixed_ppo` のみ
  - `cycles[*].learner_diag.mixed_ppo.mixed_ppo_enabled == true`
  - `cycles[*].learner_diag.mixed_ppo.num_policy_samples == 0`
  - `cycles[*].learner_diag.mixed_ppo.num_baseline_samples > 0`
  - `cycles[*].learner_diag.teacher_agreement.enabled == true`

## 5. 判定基準（診断実験）

- `gamma` が主因寄り:
  - `gamma=0.95` が `0.99` より一貫して `final_vs_init`、`best->final`、`late.value_error` を改善する
  - shaping の強さに関係なく傾向が再現する
- `shanten shaping` が主因寄り:
  - `scale=0.0` または `0.003` が `0.01` より一貫して改善する
  - とくに `teacher_agreement` が改善するのに `avg_score` も伸びるなら、短期 shaping の歪みが強い
- interaction が大きい:
  - `gamma=0.95 x shaping弱` の組み合わせだけが大きく改善し、単独変更では差が小さい
- なお悪い:
  - `teacher_agreement` は上がるのに `avg_score/avg_rank` が頭打ちのまま
  - これは reward/target のさらに別の歪み、または model/value 設計の問題を示唆する

## 6. 見たい結論

この runbook で決めたいのは次の一点。

> `gae=0.0` まで短期化しても残る plateau / 戻りは、`gamma` を詰めるべき問題か、それとも `shanten shaping` を弱めるべき問題か。

## 7. 想定所要時間

- 1条件 x 3 seeds: おおむね `30〜40分`
- 6条件合計: `180〜240分`
- 余裕込み: `3〜4時間`

## 8. 実行方針

- 今回は `gamma` と `shanten shaping` をまとめて見る
- ここで主因が片方に寄れば、次段ではその軸だけを細かく詰める
- ここで interaction が強ければ、次段は `2 x 3` をさらに狭い範囲で再度切る
- ここでどちらも効かなければ、次は reward/target ではなく value 依存や teacher 距離解析の比重を上げる
