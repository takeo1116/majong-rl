# Experiment Runbook: exp_033

作成日: 2026-05-09  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_033` は、ルール拡張へ進む前の Stage2a 最終確認実験である。

目的は、直近の修正をすべて反映した上で、現在の本命設定が 3seed で安定しているかを確認すること。

反映済みの重要修正:

- CQ-0282: rulebase baseline action を PPO ratio に混ぜない separated PPO
- CQ-0283: `reward.point_delta_scale=0.0001`
- CQ-0285: terminal loss の weighted mean 正規化
- CQ-0286: policy / value_semantic optimizer lr group 分離
- CQ-0287: target_kl early stop
- CQ-0288: dead weight だった `semantic_proj` 削除
- CQ-0289: lr_groups の適用範囲を PPO / imitation で分離

特に CQ-0289 により、今回の条件では以下が成立する。

```text
imitation warmstart: single lr = 1e-4
PPO phase: policy lr = 5e-4, value_semantic lr = 1e-2
```

これにより、`exp_032` までに残っていた「imitation も高 lr_groups で動いていた」という交絡を外す。

## 2. 背景

### 2.1 exp_032 の結果

`exp_032` では `P5x_VS100x` が seed42 で非常に強かった。

```text
P5x_VS100x seed42:
final  2.105
best   1.885
best10 2.044
tail10 2.098
tail20 2.105
```

しかし 3seed 化すると seed44 が弱く、平均では本命確定できなかった。

```text
P5x_VS100x 3seed:
final  mean 2.2017
best   mean 2.0433
best10 mean 2.1165
tail10 mean 2.1852
tail20 mean 2.1811
```

### 2.2 target_kl probe

弱かった seed44 に target_kl を入れると、大きく改善した。

```text
P5 seed44 baseline:
final  2.340
best   2.195
best10 2.224
tail10 2.293
tail20 2.280
score  26378.5

P5 seed44 target_kl:
final  2.115
best   2.070
best10 2.119
tail10 2.155
tail20 2.194
score  29554.0
```

これは、target_kl が policy lr 高め条件の悪い踏み込みを抑え、seed 安定性を改善する可能性を示す。

### 2.3 CQ-0288 / CQ-0289 の影響

`semantic_proj` は削除済みなので、semantic summary は terminal / yaku predictions のみで構成される。  
また、`lr_groups.apply_to=["ppo"]` により imitation は高 lr_groups の影響を受けない。

したがって `exp_033` は、`exp_032` の単純な続きではなく、より整理された最終構成として扱う。

## 3. 実験条件

3seed:

| label | seed | purpose |
|---|---:|---|
| `FINAL_P5_TKL_seed42` | 42 | baseline seed |
| `FINAL_P5_TKL_seed43` | 43 | stability seed |
| `FINAL_P5_TKL_seed44` | 44 | exp032 で弱かった seed の再確認 |

共通設定:

```yaml
feature_encoder:
  tile_presence_flags: true

model:
  value_hidden_dims: [256, 128]
  semantic_aux:
    enabled: true
    tile_presence_flags_semantic_only: false
    # policy_projection_dim は CQ-0288 で廃止/無視

training:
  lr: 0.0001
  value_loss_coef: 0.125
  clip_epsilon: 0.15
  entropy_coef: 0.0

  semantic_aux:
    enabled: true
    terminal_loss_coef: 0.1
    yaku_loss_coef: 0.05

  rule_mix:
    policy_ratio: 1.0
    save_baseline_actions: false

  rule_mix_learner:
    enabled: true
    ppo_mode: "separated"
    baseline_imitation_epochs: 0
    policy_ppo_epochs: 1
    allow_mixed_offpolicy_baseline: false

  policy_anchor:
    enabled: false
    coef: 0.0

  lr_groups:
    enabled: true
    apply_to: ["ppo"]
    policy: 0.0005
    value_semantic: 0.01
    default: 0.0001

  ppo_target_kl:
    enabled: true
    target: 0.03
    stop_multiplier: 1.5
    skip_minibatch_on_exceed: true

  diagnostics:
    gradient_norms:
      enabled: true
      max_batches_per_epoch: 4
      every_n_epochs: 1

reward:
  type: "point_delta"
  point_delta_scale: 0.0001

selfplay:
  policy_ratio: 1.0
  temperature: 1.0

multi_cycle:
  num_cycles: 60
  selfplay_matches_per_cycle: 200
```

## 4. 比較対象

```text
exp026 P100 scaled 3seed:
experiments/Stage02_CallUnlock/exp_026/report.md

TERM50x 3seed:
experiments/Stage02_CallUnlock/exp_028/report.md

VS_LR100x seed42:
runs/20260508_stage2a_exp030_vs_lr100x_seed42_f1c3da06

P5x_VS100x seed42/43/44 without target_kl:
experiments/Stage02_CallUnlock/exp_032/report.md

P5x_VS100x seed44 + target_kl probe:
runs/20260509_stage2a_targetkl_probe_p5x_vs100x_seed44_2a63572f
```

## 5. 実行方針

この runbook では設計のみを定義する。実行用 driver は別途 `scripts/local/stage2/exp_033_driver.py` として作る。

想定 driver:

```bash
./.venv/bin/python scripts/local/stage2/exp_033_driver.py
```

単発実行できるようにする場合:

```bash
EXP033_ONLY=FINAL_P5_TKL_seed42 ./.venv/bin/python scripts/local/stage2/exp_033_driver.py
EXP033_ONLY=FINAL_P5_TKL_seed43 ./.venv/bin/python scripts/local/stage2/exp_033_driver.py
EXP033_ONLY=FINAL_P5_TKL_seed44 ./.venv/bin/python scripts/local/stage2/exp_033_driver.py
```

validate-only:

```bash
EXP033_VALIDATE_ONLY=1 ./.venv/bin/python scripts/local/stage2/exp_033_driver.py
```

## 6. 見るべき指標

Performance:

- final avg_rank
- best avg_rank
- best10
- tail10
- tail20
- final win_rate
- final deal_in_rate
- final avg_score

PPO / stability:

- entropy
- max_prob_mean
- clip_fraction
- ratio_max
- log_ratio_p01 / log_ratio_p99
- target_kl_stop_count
- target_kl_skipped_minibatches
- target_kl_checked_minibatches
- `skipped / checked`

lr_groups scope:

- imitation phase の `optimizer_lr_groups.active_for_algorithm == false`
- PPO phase の `optimizer_lr_groups.active_for_algorithm == true`
- PPO phase の `policy lr == 0.0005`
- PPO phase の `value_semantic lr == 0.01`

semantic / gradient:

- terminal_loss
- yaku_loss
- value_loss
- gradient_norms aggregate ratios
- semantic eval は必要なら final checkpoint で別途実行

## 7. 成功条件

### Strong success

```text
3seed mean tail10 <= 2.13
3seed mean tail20 <= 2.13
3seed mean best10 <= 2.07
seed44 が exp032 baseline より明確に改善
```

この場合、Stage2a は一区切りとしてルール拡張へ進む。

### Moderate success

```text
3seed mean tail10/tail20 は exp026 P100 scaled と同程度
seed44 は target_kl で改善
大きな collapse なし
```

この場合も、ルール拡張へ進んでよい。Stage2a 内の追加 tuning は後回し。

### Failure

```text
seed 間ばらつきが大きい
seed44 が再び 2.25+ tail に悪化
target_kl skip が過剰で update 不足
entropy / max_prob が collapse 寄り
```

この場合、ルール拡張前に以下を検討する。

- policy lr を `0.0003` または `0.0004` に下げる
- `policy_ppo_epochs=2` と target_kl の組み合わせを試す
- Adam state carry-over (Claude review H-1) を別 CQ 化する

## 8. 判断

`exp_033` は Stage2a CallUnlock の締め実験とする。

`Moderate success` 以上なら、追加 tuning よりルール拡張を優先する。
