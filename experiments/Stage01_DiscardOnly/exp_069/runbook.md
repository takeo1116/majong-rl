# Experiment Runbook: exp_069

作成日: 2026-03-23  
目的: corrected semantics 上の current best 候補 `anchor=0.75, policy_ratio=0.10` から出発し、`training.rule_mix.policy_ratio` を **より大きい帯 (`0.30 / 0.50 / 0.70`)** に振って、actor mix を強めた方が plateau / final が改善するかを 3 seeds で確認する。

## 1. 背景

- CQ-0210 / CQ-0211 修正後、`improve / same / worsen` の向きはかなり自然化した。
- `exp_067` では単発で `policy_ratio=0.05`, `0.10` を見て、`0.10` の方が明らかに良かった。
- `exp_068` では組み合わせ条件の中で
  - `anchor=0.75`
  - `policy_ratio=0.10`
  が最良だった。
- 特に `exp_068` B (`anchor075_ratio010`) を 3 seeds で見ると、`final-init` はノイジーだが、**cycle 20-29 平均では imitation 直後平均を上回った**。
- ここまでで、post-fix では `policy_ratio=0.10` が有望と分かった一方、`0.10` は actor mix としてはまだかなり保守的である。
- したがって次は、`policy_ratio` を **0.30 / 0.50 / 0.70** まで大きく振り、
  - さらに高い actor mix が有利か
  - あるところから急に崩れるか
  を切り分ける。

## 2. 実験の問い

1. `policy_ratio=0.10` よりも `0.30 / 0.50 / 0.70` の方が final / plateau 平均で良いか
2. actor mix を増やすと
   - distribution mismatch はさらに改善するか
   - それとも baseline anchor の利点を失って悪化するか
3. `policy_ratio` の最適帯が
   - 低い (`0.10` 前後)
   - 中間 (`0.30` 前後)
   - 高い (`0.50` 以上)
   のどこにあるか

## 3. 基準条件

基準は `exp_068` の B `anchor075_ratio010`。

- 新モデル (`policy_direct_hints + context_gate`)
- `training.gamma=0.75`
- `training.gae_lambda=0.3`
- `training.clip_epsilon=0.15`
- `training.policy_anchor.coef=0.75`
- `training.value_loss_coef=0.25`
- `training.rule_mix.policy_ratio=0.10`
- `reward.shaping.shanten_delta.scale=0.003`
- imitation `1000 x 3 chunks`
- PPO `200 x 30 cycles`
- seeds `42, 43, 44`

参照条件:
- `REF anchor075_ratio010`
  - 既存の `exp_068` B 3-seed 結果を参照点として使う
  - 新規再実行は前提にしない
  - 主な集計値:
    - `final_score mean = 2348.58`
    - `drawdown mean = 882.33`
    - `cycle 20-29 mean = 2494.93`
    - `best_set_after mean = 0.9130`

## 4. 条件一覧

- 新規条件数: 3
- seeds: `42, 43, 44`
- 方針: `policy_ratio` だけを大きく振る

| 条件 | 変更内容 | 意図 |
|---|---|---|
| `REF anchor075_ratio010` | `policy_ratio=0.10` | `exp_068` の current best 候補 |
| A `anchor075_ratio030` | `policy_ratio=0.30` | 中程度 actor mix |
| B `anchor075_ratio050` | `policy_ratio=0.50` | 半数近く actor mix |
| C `anchor075_ratio070` | `policy_ratio=0.70` | actor 主体に近い高 mix |

## 5. 共通固定条件

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
- reward:
  - `reward.point_delta_scale=0.0001`
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.scale=0.003`
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

## 6. 主評価指標

1. performance
   - `top-level imitation initial avg_score`
   - `best avg_score`
   - `final avg_score`
   - `avg_rank`
   - `win_rate`
   - `deal_in_rate`
2. plateau / 保持
   - `final - imitation_initial`
   - `best -> final` drawdown
   - `cycle 20-29 avg_score mean`
   - best cycle
3. teacher / update
   - `teacher_agreement.best_set_hit_rate_after`
   - `clip_fraction`
   - `ratio_std`
4. corrected signal の自然さ
   - `shanten_diag.improve/same/worsen.reward.mean`
   - `shanten_diag.improve/same/worsen.advantage.mean`
5. mixed 条件確認
   - `mixed_ppo.num_policy_samples`
   - `mixed_ppo.num_baseline_samples`
   - `actor_type_counts`

## 7. 読み方

### A `ratio030` が最良
- `0.10` はまだ保守的
- actor mix を増やすと distribution mismatch がさらに改善する
- 次は `0.20 / 0.30 / 0.40` の近傍を精査する

### B `ratio050` が最良
- かなり高い actor mix でも有利
- baseline 依存は思ったより弱く、policy 分布を積極的に見せた方が良い
- 次は `0.50 / 0.70 / 0.90` 方向も検討できる

### C `ratio070` が最良
- current setup では、ほぼ actor 主体の mixed PPO が本命
- `rule-only` に近い regime からかなり離れた方が良い
- その場合は `baseline_sample_weight` や `selfplay.policy_ratio` まで含めて regime を見直す価値がある

### `0.30 / 0.50 / 0.70` がすべて `0.10` に負ける
- 最適帯は `0.10` 近傍
- 次は `0.10 / 0.15 / 0.20` の細かい sweep に戻る

### 高 ratio 条件で signal が崩れる
- 例えば `worsen_adv > 0` や `best_set_after` 急落が出る
- その場合は actor mix を増やしすぎて baseline rail を失っている
- `policy_ratio` より `baseline_sample_weight` 調整の方が本命かもしれない

## 8. 実行上の注意

- `training.rule_mix.policy_ratio` が実際に PPO cycle 中の混合比を決めるため、今回見るべきノブはこれだけでよい
- `selfplay.policy_ratio=1.0` は current setup では `training.rule_mix.policy_ratio` に上書きされるため、今回の sweep では意味を持たない
- report には `runs/` 配下のローカル成果物パスを書かず、必要な数値を転記して残す
- `REF` は `exp_068` report の 3-seed 集計を参照する

## 9. 成功条件

- 新規 3 条件 × 3 seeds が完了する
- 各 run で:
  - imitation / selfplay / learner / eval が `success`
  - `summary.phase_stats.cycles` 長さ `30`
  - 変更対象の `config.yaml` が条件どおり
  - `num_policy_samples > 0`
- `cycle 20-29 mean` か `final_score mean` のどちらかで、`policy_ratio=0.10` を上回る条件があるかを判定できる

## 10. 想定所要時間

- 新規実行は `3` 条件 × `3` seeds = `9 runs`
- 1 run あたり `70〜90分` 程度
- 合計 `11〜14時間` 程度

## 11. 実行後にやること

1. `report.md` を作成
2. `policy_ratio` の有望帯を
   - `0.10` 近傍
   - `0.30` 近傍
   - `0.50+`
   のどれかに絞る
3. 必要なら次に
   - `baseline_sample_weight`
   - `policy_ratio` 細粒度 sweep
   を切る
