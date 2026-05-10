# Experiment Runbook: exp_034

作成日: 2026-05-10  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_034` は、ルール拡張へ進む前の Stage2a 安定設定を 3seed で確認する実験である。

`exp_033` では `policy_lr=5x` を試したが、seed43 で序盤の悪化と大きな振動が出た。seed42 は良かったものの、安定設定としては上振れ依存の疑いがある。

そのため本実験では、policy 側は保守的に `1x` に戻し、value/semantic 側だけ `100x` を維持する。

```text
policy lr         = 1e-4  (1x)
value_semantic lr = 1e-2  (100x)
target_kl         = enabled
lr_groups.apply_to = ["ppo"]
```

## 2. 背景

反映済みの重要修正:

- CQ-0282: rulebase baseline action を PPO ratio に混ぜない separated PPO
- CQ-0283: `reward.point_delta_scale=0.0001`
- CQ-0285: terminal loss の weighted mean 正規化
- CQ-0286: policy / value_semantic optimizer lr group 分離
- CQ-0287: target_kl early stop
- CQ-0288: dead weight だった `semantic_proj` 削除
- CQ-0289: lr_groups の適用範囲を PPO / imitation で分離

`exp_032` / `exp_033` からの判断:

- value/semantic 側の lr を上げることには改善余地がある。
- policy lr を `5x` まで上げると当たり seed では強いが、seed 間の分散が大きい。
- ルール拡張前の基準としては、上振れ性能より再現性を優先する。

## 3. 実験条件

3seed:

| label | seed | purpose |
|---|---:|---|
| `FINAL_P1_TKL_seed42` | 42 | baseline seed |
| `FINAL_P1_TKL_seed43` | 43 | stability seed |
| `FINAL_P1_TKL_seed44` | 44 | weak-seed robustness |

共通設定:

```yaml
feature_encoder:
  tile_presence_flags: true

model:
  value_hidden_dims: [256, 128]
  semantic_aux:
    enabled: true
    tile_presence_flags_semantic_only: false

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
    policy: 0.0001
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

P5x_VS100x / target_kl:
experiments/Stage02_CallUnlock/exp_033/runbook.md
```

## 5. 実行

```bash
./.venv/bin/python scripts/local/stage2/exp_034_driver.py
```

単発実行:

```bash
EXP034_ONLY=FINAL_P1_TKL_seed42 ./.venv/bin/python scripts/local/stage2/exp_034_driver.py
EXP034_ONLY=FINAL_P1_TKL_seed43 ./.venv/bin/python scripts/local/stage2/exp_034_driver.py
EXP034_ONLY=FINAL_P1_TKL_seed44 ./.venv/bin/python scripts/local/stage2/exp_034_driver.py
```

validate-only:

```bash
EXP034_VALIDATE_ONLY=1 ./.venv/bin/python scripts/local/stage2/exp_034_driver.py
```

## 6. 判断基準

Performance:

- final avg_rank
- best avg_rank
- best10
- tail10
- tail20

安定性:

- seed 間の final / tail20 分散
- late drift の有無
- target_kl stop count / skipped minibatches
- entropy / max_prob / ratio tail

採用基準:

- `policy_lr=5x` より seed 間の振動が小さいこと
- `policy_lr=1x` で十分な best / tail20 が出ること
- ルール拡張前の基準モデルとして説明しやすいこと
