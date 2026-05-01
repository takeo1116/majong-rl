# Experiment Runbook: exp_020

作成日: 2026-04-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_015/report.md`
- `experiments/Stage02_CallUnlock/exp_019/report.md`
- `docs/CHANGE_QUEUE.md`

## 1. 背景

`exp_015` 以降、実用 baseline は

- `A2_semaux_light_vhalf_tenpaifix_prnorm`

として扱ってきた。

ただし 20 日ぶりの再点検で、Stage2a の RL 更新まわりに、理論というより実装・運用 semantics の不備が複数見つかった。

今回までに修正済みの主な CQ は次の通り。

- `CQ-0274`: Stage2a selfplay の pending reward を上書きではなく全 pending sample に累積
- `CQ-0275`: PPO return / advantage を discard / call branch の元順に正しく scatter
- `CQ-0276`: `reward_config` を Stage2a selfplay / eval / parallel 経路に伝播
- `CQ-0277`: terminal player-round weight を discard / call cross-branch で計算
- `CQ-0278`: Stage2a selfplay の torch RNG を match seed で固定し、`selfplay.temperature` を実際に反映
- `CQ-0279`: Stage2a shard semantics を v3 に上げ、旧 v2 shard を learner 側で fail-fast

したがって、過去の `exp_015` / `exp_019` 結果は比較対象としては有用だが、現行コードの RL 更新性能を評価するには、fresh shard で取り直す必要がある。

## 2. 今回の問い

`exp_020` の主目的は、次の一点である。

- 修正後の Stage2a RL 更新が、A2 baseline に対して 3 seed で安定して imitation から改善するか

今回は yakuflags 系の追加特徴量は扱わない。まずは、現在の practical baseline を clean に取り直す。

## 3. 実験条件

全 run 共通:

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- condition: `A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl`
- `training.multi_cycle.num_cycles = 30`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `training.policy_anchor.coef = 0.75`
- `training.value_loss_coef = 0.125`
- `model.value_hidden_dims = [128, 64]`
- `model.semantic_aux.enabled = true`
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.enabled = true`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `feature_encoder.tile_presence_flags = false`
- `model.semantic_aux.tile_presence_flags_semantic_only = false`
- `selfplay.temperature = 1.0`
- expected shard semantics: `sample_semantics_version = 3`

## 4. Seeds

3 seed を連続実行する。

| label | seed |
|---|---:|
| `A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed42` | 42 |
| `A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed43` | 43 |
| `A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed44` | 44 |

## 5. 観測ポイント

### 5.1 policy performance

各 seed で見るもの:

- imitation eval
- final eval
- best cycle
- tail-5 / tail-10 average
- `avg_rank`
- `win_rate`
- `deal_in_rate`

最重要は、imitation から final/tail が改善しているかどうか。

### 5.2 PPO stability

各 cycle で見るもの:

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- `retain`
- policy loss / value loss の暴れ

修正後に見るべきポイントは、性能だけではなく「更新の意味が安定しているか」。特に `CQ-0275` によって advantage/return alignment が直っているため、以前より PPO 指標と policy 結果の対応が素直になることを期待する。

### 5.3 reward / shard semantics

確認事項:

- 新規 run の shard が v3 で生成されること
- learner が旧 shard を混ぜずに通ること
- reward backfill 修正後、cycle ごとの value learning が以前より不自然に崩れないこと

## 6. 成功判定

成功とみなす条件:

1. 3 seed 平均で final または tail 平均が imitation より改善する
2. 少なくとも 2/3 seed で RL 後の `avg_rank` が imitation を上回る
3. `clip_fraction` / `anchor_kl_discard` が過大化せず、PPO 更新が壊れていない
4. 旧実験より少なくとも「RL で悪化しやすい」という印象が弱まる

失敗とみなす条件:

- 3 seed で一貫して imitation より悪化
- early cycle だけ良く、tail が大きく崩れる
- PPO diagnostics が暴れる
- v3 shard / reward semantics の fail-fast に引っかかる

## 7. 実行コマンド

全 seed:

```bash
./.venv/bin/python scripts/local/stage2/exp_020_driver.py
```

1 seed だけ実行:

```bash
EXP020_ONLY=A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed42 \
  ./.venv/bin/python scripts/local/stage2/exp_020_driver.py
```

## 8. 期待アウトプット

- `experiments/Stage02_CallUnlock/exp_020/run_map.json`
- `experiments/Stage02_CallUnlock/exp_020/driver_logs/*.log`
- `runs/20260430_stage2a_exp020_*`
- 3 seed 比較用の `summary.json`
- 後続の `report.md`
