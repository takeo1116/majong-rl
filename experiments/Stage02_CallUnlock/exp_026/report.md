# Experiment Report: exp_026

作成日: 2026-05-07  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_026` の目的は、`exp_025` で最有力になった `policy_ratio=1.00` の pure learned-policy selfplay 構成に、`reward.point_delta_scale=0.0001` を追加した条件を 3seed で検証することである。

`exp_025` までの Stage2a 標準 config には `reward:` ブロックがなく、C++ default の `point_delta_scale=1.0` が使われていた可能性が高い。Stage1 では `point_delta_scale=0.0001` を使っているため、Stage2a だけ reward / return / value target が raw 点数スケールになっていた可能性がある。

seed42 probe では value loss / semantic / policy performance が大きく改善したため、`exp_026` で seed43/44 を追加して 3seed 化した。

## 2. 条件

Base: `exp_025 P100`

変更点:

```yaml
reward:
  type: "point_delta"
  point_delta_scale: 0.0001
```

固定条件:

- `training.rule_mix.policy_ratio = 1.00`
- `training.rule_mix_learner.ppo_mode = "separated"`
- `training.rule_mix_learner.baseline_imitation_epochs = 0`
- `training.rule_mix_learner.policy_ppo_epochs = 1`
- `training.rule_mix_learner.allow_mixed_offpolicy_baseline = false`
- `training.policy_anchor.enabled = false`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.15`
- `training.entropy_coef = 0.0`
- `training.value_loss_coef = 0.125`
- `training.multi_cycle.num_cycles = 60`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `feature_encoder.tile_presence_flags = true`
- `model.value_hidden_dims = [256,128]`
- `model.semantic_aux.enabled = true`
- `model.semantic_aux.tile_presence_flags_semantic_only = false`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`

## 3. Run 一覧

| label | seed | run_dir |
|---|---:|---|
| P100 scaled | 42 | `runs/20260503_stage2a_rewardscale_probe_P100_seed42_dd0b0c5d` |
| P100 scaled | 43 | `runs/20260506_stage2a_exp026_P100_scaled_seed43_3a09a61d` |
| P100 scaled | 44 | `runs/20260507_stage2a_exp026_P100_scaled_seed44_878278a4` |

比較対象:

- `exp_025 P100 raw` 3seed

## 4. 主結果

### 4.1 3seed 平均

| condition | n | final | best | best5 | best10 | tail5 | tail10 | tail20 | win | deal-in | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P100 raw exp025 | 3 | 2.1150 | 2.0083 | 2.0337 | 2.0582 | 2.1710 | 2.1545 | 2.1409 | 0.2219 | 0.1880 | 29478.3 |
| P100 scaled exp026 | 3 | 2.0817 | 1.9600 | 2.0110 | 2.0422 | 2.1043 | 2.1203 | 2.1292 | 0.2348 | 0.1835 | 30066.3 |

改善幅:

| metric | diff |
|---|---:|
| final avg_rank | -0.0333 |
| best avg_rank | -0.0483 |
| best10 avg_rank | -0.0160 |
| tail10 avg_rank | -0.0342 |
| tail20 avg_rank | -0.0117 |
| win_rate | +0.0129 |
| deal_in_rate | -0.0045 |
| avg_score | +588.0 |

### 4.2 seed ばらつき

| condition | final_sd | best10_sd | tail10_sd | tail20_sd | win_sd | deal_sd |
|---|---:|---:|---:|---:|---:|---:|
| P100 raw exp025 | 0.0200 | 0.0291 | 0.0170 | 0.0264 | 0.0046 | 0.0084 |
| P100 scaled exp026 | 0.0967 | 0.0229 | 0.0055 | 0.0076 | 0.0318 | 0.0138 |

scaled は final のばらつきが大きいが、tail10 / tail20 のばらつきは raw より小さい。終盤平均は安定して改善している。

### 4.3 seed 別詳細

| condition | seed | final | best | best_cycle | best10 | tail10 | tail20 | win | deal-in | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 42 | 2.115 | 2.055 | 53 | 2.092 | 2.155 | 2.171 | 0.2171 | 0.1974 | 29423.5 |
| raw | 43 | 2.095 | 1.965 | 35 | 2.038 | 2.137 | 2.131 | 0.2223 | 0.1852 | 29986.5 |
| raw | 44 | 2.135 | 2.005 | 43 | 2.045 | 2.171 | 2.120 | 0.2263 | 0.1813 | 29025.0 |
| scaled | 42 | 1.970 | 1.970 | 59 | 2.052 | 2.124 | 2.137 | 0.2368 | 0.1675 | 30757.5 |
| scaled | 43 | 2.135 | 1.955 | 57 | 2.058 | 2.123 | 2.129 | 0.2656 | 0.1918 | 29722.0 |
| scaled | 44 | 2.140 | 1.955 | 35 | 2.016 | 2.114 | 2.122 | 0.2020 | 0.1911 | 29719.5 |

## 5. Learner Diagnostics

### 5.1 final cycle 平均

| condition | value_loss | terminal_loss | yaku_loss | entropy | clip | log_ratio_p01 | ratio_max | max_prob | adv_pos | policy steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P100 raw exp025 | 1.16211e6 | 13.7275 | 0.2404 | 0.2693 | 0.0813 | -0.4294 | 10.1643 | 0.8907 | 0.4838 | 123105 |
| P100 scaled exp026 | 0.0147108 | 12.0533 | 0.1744 | 0.2145 | 0.0544 | -0.3168 | 9.0563 | 0.9134 | 0.5884 | 117828 |

### 5.2 tail10 平均

| condition | value_loss | terminal_loss | yaku_loss | entropy | clip | log_ratio_p01 | ratio_max | max_prob |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P100 raw exp025 | 1.17315e6 | 13.8451 | 0.2413 | 0.2704 | 0.0874 | -0.4765 | 10.4236 | 0.8901 |
| P100 scaled exp026 | 0.0147995 | 12.1038 | 0.1722 | 0.2179 | 0.0544 | -0.3174 | 7.1974 | 0.9124 |

## 6. 解釈

### 6.1 reward scale 仮説は 3seed で支持された

`point_delta_scale=0.0001` により、value loss は全 seed で正常スケールになった。

```text
raw:    ~1.16e6
scaled: ~0.0147
```

これは Stage2a が raw 点数スケールで value target を学習していた、という仮説を強く支持する。

### 6.2 policy performance も改善した

scaled は final / best / best10 / tail10 / tail20 のすべてで raw を上回った。seed42 の改善幅が特に大きいが、seed43/44 でも best / tail10 / tail20 は改善している。

特に tail10 は:

```text
raw:    2.1545
scaled: 2.1203
```

であり、終盤平均としても改善がある。

### 6.3 PPO 更新は安定化したが、方策はやや硬くなった

scaled では以下が改善した。

- `clip_fraction`: `0.0813 → 0.0544`
- `log_ratio_p01`: `-0.4294 → -0.3168`
- `tail10 ratio_max`: `10.4236 → 7.1974`

一方で、以下は方策が硬くなったことを示す。

- `entropy`: `0.2693 → 0.2145`
- `max_prob`: `0.8907 → 0.9134`

つまり、reward scale 正規化により PPO の ratio/clip は安定化したが、policy の出力分布はやや deterministic に寄った。現時点では collapse ではないが、次の実験では entropy / max_prob を継続監視する。

### 6.4 semantic auxiliary も改善傾向

seed42 semantic eval では、yaku / terminal が大幅に改善した。

| metric | raw | scaled |
|---|---:|---:|
| terminal accuracy | 0.6017 | 0.6351 |
| yaku micro F1 | 0.4945 | 0.6777 |
| yaku macro F1 | 0.1065 | 0.1924 |
| yaku exact match | 0.1640 | 0.3533 |
| deal_in ROC AUC | 0.5539 | 0.6510 |
| deal_in PR AUC | 0.1887 | 0.2861 |

また、same-shard cross eval でも Tanyao / Yakuhai / win_called terminal の改善が見えた。これは、単なる selfplay 分布差ではなく、model 側の semantic head が改善したことを示す。

## 7. 結論

`reward.point_delta_scale=0.0001` は Stage2a 標準に採用する。

理由:

- value loss の異常スケールが解消した
- 3seed 平均で policy performance が改善した
- tail10 / tail20 が改善し、終盤平均でも効果がある
- yaku / terminal semantic も大幅改善した
- PPO clip / log_ratio diagnostics も改善した

新しい Stage02a 基準候補:

```text
policy_ratio=1.00
reward.point_delta_scale=0.0001
separated policy-only PPO
no anchor
tile_presence_flags=true
value_hidden_dims=[256,128]
semantic_aux enabled
```

## 8. 次アクション

### 8.1 CQ-0283 を実装する

`configs/stage2a_core_minimal_mixed_s1_baseline.yaml` に以下を追加する。

```yaml
reward:
  type: "point_delta"
  point_delta_scale: 0.0001
```

Stage2a selfplay / eval path への reward config propagation は CQ-0276 で実装済みなので、基本的には config 修正と smoke test で足りる。

### 8.2 次の改善候補

reward scale を標準化した後、次に検討する候補:

1. gradient norm diagnostics
   - value / terminal / yaku / policy が value trunk に流す gradient norm を確認
   - terminal aux が強すぎないかを見る
2. `gae_lambda`
   - 現在 `gamma=0.5`, `gae_lambda=0.0`
   - value が正常化したため、credit assignment を再検討する価値がある
3. `policy_ppo_epochs=2`
   - target_kl early stop と組み合わせて検討
4. entropy / max_prob 対策
   - scaled で方策が硬くなっているため、必要なら小さな entropy_coef を再検討
5. Stage02b ルール拡張
   - Stage02a 基準がかなり安定したため、ルール拡張へ進む条件は整いつつある
