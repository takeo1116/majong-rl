# Experiment Runbook: exp_026

作成日: 2026-05-06  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_026` の目的は、`exp_025` で次期基準候補になった `policy_ratio=1.00` の pure learned-policy selfplay 構成に、`reward.point_delta_scale=0.0001` を入れた条件を 3seed で検証することである。

`exp_025` の後に実施した seed42 probe では、`point_delta_scale=0.0001` により value loss と semantic auxiliary が大きく改善し、policy performance も大きく伸びた。

seed42 probe:

```text
runs/20260503_stage2a_rewardscale_probe_P100_seed42_dd0b0c5d
```

この seed42 run は `exp_026` の seed42 として流用し、driver では seed43/44 の 2 本だけを新規実行する。

## 2. 背景

`exp_025` では `policy_ratio=0.50/0.75/1.00` を比較し、`policy_ratio=1.00` が 3seed 平均で最も良かった。

| condition | final | best | best10 | tail10 | tail20 | win | deal-in |
|---|---:|---:|---:|---:|---:|---:|---:|
| P050 exp024 | 2.1417 | 2.0267 | 2.0835 | 2.1690 | 2.1729 | 0.2216 | 0.1908 |
| P075 exp025 | 2.1600 | 1.9767 | 2.0670 | 2.1617 | 2.1498 | 0.2172 | 0.1863 |
| P100 exp025 | 2.1150 | 2.0083 | 2.0582 | 2.1545 | 2.1409 | 0.2219 | 0.1880 |

したがって、以後の Stage02a 基準は `P100` を優先する。

一方、ClaudeCode review で Stage2a config に `reward:` ブロックがなく、C++ default の `point_delta_scale=1.0` が使われている可能性が指摘された。Stage1 は `point_delta_scale=0.0001` を使っているため、Stage2a は reward / return / value target が想定より約 10000 倍大きい状態だった可能性がある。

seed42 probe では、この仮説を強く支持する結果が出た。

| condition | final | best | best10 | tail10 | tail20 | win | deal-in | value_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P100 raw seed42 | 2.115 | 2.055 | 2.092 | 2.155 | 2.171 | 0.2171 | 0.1974 | 1.19e6 |
| P100 scaled seed42 | 1.970 | 1.970 | 2.052 | 2.124 | 2.137 | 0.2368 | 0.1675 | 0.014 |

Semantic eval でも大きな改善が見えた。

| metric | raw | scaled |
|---|---:|---:|
| terminal accuracy | 0.6017 | 0.6351 |
| yaku micro F1 | 0.4945 | 0.6777 |
| yaku macro F1 | 0.1065 | 0.1924 |
| yaku exact match | 0.1640 | 0.3533 |
| deal_in ROC AUC | 0.5539 | 0.6510 |
| deal_in PR AUC | 0.1887 | 0.2861 |

特に Tanyao / Yakuhai / win_called terminal の復活が大きい。

## 3. 実験条件

Base: `exp_025 P100`

変更点:

```yaml
reward:
  type: "point_delta"
  point_delta_scale: 0.0001
```

固定条件:

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `training.rule_mix.policy_ratio = 1.00`
- `training.rule_mix_learner.ppo_mode = "separated"`
- `training.rule_mix_learner.baseline_imitation_epochs = 0`
- `training.rule_mix_learner.policy_ppo_epochs = 1`
- `training.rule_mix_learner.allow_mixed_offpolicy_baseline = false`
- `training.policy_anchor.enabled = false`
- `training.policy_anchor.coef = 0.0`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.15`
- `training.entropy_coef = 0.0`
- `training.value_loss_coef = 0.125`
- `training.multi_cycle.num_cycles = 60`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `feature_encoder.tile_presence_flags = true`
- `model.value_hidden_dims = [256,128]`
- `model.semantic_aux.enabled = true`
- `model.semantic_aux.policy_projection_dim = 16`
- `model.semantic_aux.tile_presence_flags_semantic_only = false`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `selfplay.temperature = 1.0`

## 4. Run 計画

| label | seed | status |
|---|---:|---|
| `P100_scaled_seed42` | 42 | 既存 probe を流用 |
| `P100_scaled_seed43` | 43 | 新規実行 |
| `P100_scaled_seed44` | 44 | 新規実行 |

seed42 run:

```text
runs/20260503_stage2a_rewardscale_probe_P100_seed42_dd0b0c5d
```

## 5. 実行方法

```bash
./.venv/bin/python scripts/local/stage2/exp_026_driver.py
```

単発実行:

```bash
EXP026_ONLY=P100_scaled_seed43 \
  ./.venv/bin/python scripts/local/stage2/exp_026_driver.py
```

validate-only:

```bash
EXP026_VALIDATE_ONLY=1 \
  ./.venv/bin/python scripts/local/stage2/exp_026_driver.py
```

失敗時に即停止:

```bash
EXP026_STOP_ON_ERROR=1 \
  ./.venv/bin/python scripts/local/stage2/exp_026_driver.py
```

## 6. 主評価

比較対象:

- `exp_025 P100 raw` 3seed
- `exp_026 P100 scaled` 3seed

主指標:

1. `final avg_rank`
2. `tail10 avg_rank`
3. `tail20 avg_rank`
4. `best10 avg_rank`
5. `deal_in_rate`
6. `win_rate`

採用基準:

- 3seed 平均で `P100 raw` より final / tail10 / tail20 のいずれかが明確に改善
- value_loss が全 seed で正常スケールになる
- PPO diagnostics が大きく悪化しない

## 7. Diagnostics

必ず見るもの:

- `value_loss`
- `terminal_loss`
- `yaku_loss`
- `entropy`
- `clip_fraction`
- `log_ratio_p01 / p99`
- `ratio_max`
- `max_prob_mean`

追加で seed42 と同様に semantic eval を final checkpoint で実施する。

見るもの:

- terminal accuracy
- `win_called` recall / mean_p / top3
- `draw_tenpai` recall / mean_p
- deal_in risk ROC AUC / PR AUC
- yaku micro/macro F1
- yaku exact match
- Tanyao / Yakuhai / Pinfu confidence

## 8. 次アクション

### 良い場合

`reward.point_delta_scale=0.0001` を Stage2a default に昇格する CQ を切る。

想定修正:

- `configs/stage2a_core_minimal_mixed_s1_baseline.yaml` に `reward:` ブロック追加
- summary / diagnostics で reward scale が追跡できることを確認
- config validation / smoke test を追加

その後、次の候補を検討する。

- `gae_lambda`
- `policy_ppo_epochs=2`
- target_kl early stop
- Stage02b ルール拡張

### 悪い場合

seed42 の改善が偶然だったと判断し、`exp_025 P100 raw` を暫定基準に戻す。ただし value_loss の異常スケール問題は残るため、value_loss_coef / semantic loss coef の調整は別途検討する。
