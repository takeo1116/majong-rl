# Experiment Runbook: exp_068

作成日: 2026-03-22  
目的: `exp_067` で効いた単発ノブを組み合わせ、corrected semantics 上で **final が imitation 直後を超えられるか** を 1 seed pilot で確認する。

## 1. 背景

- CQ-0210 / CQ-0211 修正後、`improve / same / worsen` の向きはかなり自然化した。
- ただし `exp_067` では、全条件で
  - `best > init`
  - `final < init`
  となり、問題は **改善不能ではなく保持失敗** だと分かった。
- `exp_067` の単発探索で最も効いたのは:
  - `anchor=0.75`
  - `clip=0.10`
  - 次点で `policy_ratio=0.10`
- したがって次は、これらの **当たりノブを組み合わせて** final 保持がさらに改善するかを見る。

## 2. 実験の問い

1. `anchor=0.75` と `clip=0.10` を同時に入れると、final が imitation 直後を超えるか
2. `policy_ratio=0.10` を足すと、distribution mismatch の改善が上乗せされるか
3. 保持改善の本命が
   - drift 制御 (`anchor`, `clip`)
   - distribution mismatch (`policy_ratio`)
   のどちらか

## 3. 基準条件

基準は corrected baseline `g075_gae030`。

- 新モデル (`policy_direct_hints + context_gate`)
- `rule-only PPO`
- `training.gamma=0.75`
- `training.gae_lambda=0.3`
- `training.clip_epsilon=0.15`
- `training.policy_anchor.coef=0.5`
- `training.value_loss_coef=0.25`
- `training.rule_mix.policy_ratio=0.0`
- `reward.shaping.shanten_delta.scale=0.003`
- imitation `1000 x 3 chunks`
- PPO `200 x 30 cycles`
- seed `42`

参照条件:
- `REF g075_gae030`
  - corrected baseline そのものを **今回あらためて 1 seed で再実行** する
  - `runs/` を整理済みのため、既存 batch 再利用は前提にしない

## 4. 条件一覧

- 条件数: 4
- seeds: `42`
- 方針: **exp_067 で効いたノブの組み合わせだけを見る**

| 条件 | 変更内容 | 意図 |
|---|---|---|
| `REF g075_gae030` | 変更なし | corrected baseline の再実行参照点 |
| A `anchor075_clip010` | `anchor=0.75`, `clip=0.10` | 保持寄り 2 ノブの本命組み合わせ |
| B `anchor075_ratio010` | `anchor=0.75`, `policy_ratio=0.10` | drift 制御 + actor mix |
| C `clip010_ratio010` | `clip=0.10`, `policy_ratio=0.10` | step 制御 + actor mix |
| D `anchor075_clip010_ratio010` | `anchor=0.75`, `clip=0.10`, `policy_ratio=0.10` | 当たりノブ全部載せ |

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
   - `cycle0.eval_before.avg_score`
   - `best avg_score`
   - `final avg_score`
   - `avg_rank`
   - `win_rate`
   - `deal_in_rate`
2. 保持
   - `final - imitation_initial`
   - `best -> final` drawdown
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

## 7. 読み方

### A `anchor075_clip010` が良い
- 主因はやはり drift / overshoot
- まずは保持寄り設定を本命にして 3 seeds 確認へ進む

### B / C より A が良い
- `policy_ratio` よりも `anchor + clip` の方が本丸

### B / D が良い
- actor mix もかなり効いている
- distribution mismatch も strong contributor

### D が一番良い
- drift 制御と actor mix の両方が必要
- corrected baseline の次本命になる

### どれも `final < init`
- 保持寄りノブの組み合わせでも足りない
- その場合は
  1. cycle 数短縮
  2. best checkpoint 採用
  3. target / reward の再設計
  を検討する

## 8. 実行上の注意

- baseline config 自体は `gamma=0.50`, `gae=0.0` を内包しているため、今回の条件はすべて override 指定で実行する
- `REF` も新規実行する
- report には `runs/` 配下のローカル成果物パスを書かず、必要な数値を転記して残す

## 9. 成功条件

- 4 条件の単発実行が完了する
- 各 run で:
  - imitation / selfplay / learner / eval が `success`
  - `summary.phase_stats.cycles` 長さ `30`
  - 変更対象の `config.yaml` が条件どおり
  - mixed 条件では `num_policy_samples > 0`
- `final >= init` を達成する条件があるか、少なくとも `final - init` が大きく改善する条件を 1 つ以上見つける

## 10. 想定所要時間

- 新規実行は `5` 条件
- 1 条件あたり `70〜90分` 程度
- 合計 `6〜8時間` 程度

## 11. 実行後にやること

1. `report.md` を作成
2. 最良組み合わせを 1〜2 条件に絞る
3. その条件だけ 3 seeds で再確認する
