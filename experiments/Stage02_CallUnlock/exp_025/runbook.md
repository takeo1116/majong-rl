# Experiment Runbook: exp_025

作成日: 2026-05-02  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_025` の目的は、`exp_024` で新 baseline 候補になった

```text
separated policy-only PPO
no-anchor
lr=1e-4
clip=0.15
tile_presence_flags=true
value_hidden_dims=[256,128]
```

を基準に、selfplay に混ぜる rule-based baseline agent の比率を下げても安定して改善するかを見ることである。

2026-05-03 追記: seed42 の 1seed probe で `policy_ratio=0.75` / `1.00` がどちらも崩れず、特に `0.75` が有望だったため、seed43/44 を追加して 3seed 比較に拡張する。

現状の `exp_024` は `training.rule_mix.policy_ratio=0.50` である。つまり selfplay の約半分は learned policy actor、約半分は rule-based baseline actor である。

CQ-0282 以降、baseline actor sample は PPO policy loss には使わない。したがって baseline actor の役割は、主に次の 2 つである。

- 報酬を取れる相手として卓環境を作る
- learned policy だけでは崩れやすい early selfplay を安定させる

一方で、baseline actor を多く混ぜるほど、PPO 更新に使える policy sample は減る。そこで `policy_ratio` を上げることで、on-policy sample を増やしてさらに伸びるかを確認する。

## 2. 背景

`exp_023` では、baseline actor sample を PPO ratio 付きで policy loss に混ぜていたことが long-run collapse の主因だったと判断した。

`exp_024` では、`separated policy-only PPO` の上に `tile_presence_flags=true + value_hidden_dims=[256,128]` を載せることで、小幅だが安定した改善が見えた。

3seed 平均:

| condition | final | best10 | tail10 | tail20 |
|---|---:|---:|---:|---:|
| exp023 baseline | 2.167 | 2.182 | 2.199 | 2.200 |
| exp024 on_wide | 2.142 | 2.130 | 2.169 | 2.173 |

次に確認したいのは、`policy_ratio=0.50` がまだ最適なのか、baseline actor を減らした方が伸びるのかである。

## 3. 今回の問い

1. `policy_ratio=0.75` は `exp_024 policy_ratio=0.50` より良いか
2. `policy_ratio=1.00`、つまり pure policy selfplay でも安定するか
3. baseline actor はまだ環境形成役として必要か
4. policy sample を増やすことで PPO diagnostics は悪化するか

## 4. 実験方針

### 4.1 reference

`exp_024` seed42 を reference として流用する。

| reference | seed | policy_ratio | 備考 |
|---|---:|---:|---|
| `exp024 Y_onwide_separated_seed42` | 42 | 0.50 | 再実行しない |

### 4.2 new probes

初期 probe として seed42 で 2 条件を実行した。

| label | seed | policy_ratio | 意味 |
|---|---:|---:|---|
| `P075_onwide_separated_seed42` | 42 | 0.75 | baseline actor を残しつつ policy sample を増やす |
| `P100_onwide_separated_seed42` | 42 | 1.00 | pure policy selfplay |

seed42 の結果を受け、次の 4 本を追加実行する。

| label | seed | policy_ratio | 意味 |
|---|---:|---:|---|
| `P075_onwide_separated_seed43` | 43 | 0.75 | 0.75 の再現性確認 |
| `P075_onwide_separated_seed44` | 44 | 0.75 | 0.75 の再現性確認 |
| `P100_onwide_separated_seed43` | 43 | 1.00 | pure policy selfplay の再現性確認 |
| `P100_onwide_separated_seed44` | 44 | 1.00 | pure policy selfplay の再現性確認 |

## 5. 固定条件

`exp_024` と同じにする。

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `training.multi_cycle.num_cycles = 60`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `training.policy_anchor.enabled = false`
- `training.policy_anchor.coef = 0.0`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.15`
- `training.entropy_coef = 0.0`
- `training.value_loss_coef = 0.125`
- `training.rule_mix.enabled = true`
- `training.rule_mix.save_baseline_actions = true`
- `training.rule_mix_learner.enabled = true`
- `training.rule_mix_learner.ppo_mode = "separated"`
- `training.rule_mix_learner.baseline_imitation_epochs = 0`
- `training.rule_mix_learner.policy_ppo_epochs = 1`
- `training.rule_mix_learner.allow_mixed_offpolicy_baseline = false`
- `feature_encoder.tile_presence_flags = true`
- `model.value_hidden_dims = [256,128]`
- `model.semantic_aux.tile_presence_flags_semantic_only = false`
- `model.semantic_aux.enabled = true`
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `selfplay.temperature = 1.0`

変えるもの:

- `training.rule_mix.policy_ratio`

## 6. 実行方式

ローカル driver で実行する。2026-05-03 以降の default は seed43/44 の残り 4 本のみを実行する。seed42 の完了済み run は `run_map.json` に保持されるが、再実行しない。

```bash
./.venv/bin/python scripts/local/stage2/exp_025_driver.py
```

単発実行:

```bash
EXP025_ONLY=P075_onwide_separated_seed42 \
  ./.venv/bin/python scripts/local/stage2/exp_025_driver.py
```

seed42 も含めて全 6 本を明示的に対象にする場合:

```bash
EXP025_INCLUDE_SEED42=1 \
  ./.venv/bin/python scripts/local/stage2/exp_025_driver.py
```

失敗時に即停止:

```bash
EXP025_STOP_ON_ERROR=1 \
  ./.venv/bin/python scripts/local/stage2/exp_025_driver.py
```

validate-only:

```bash
EXP025_VALIDATE_ONLY=1 \
  ./.venv/bin/python scripts/local/stage2/exp_025_driver.py
```

## 7. 主評価

`exp_024 seed42` と seed42 同士で比較する。

reference values:

| condition | final | best | best5 | best10 | tail5 | tail10 | tail20 | final win | final deal-in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp024 seed42 policy_ratio=0.50 | 2.170 | 2.015 | 2.125 | 2.130 | 2.132 | 2.139 | 2.158 | 0.2013 | 0.1818 |

優先順位:

1. `tail10 avg_rank`
2. `final avg_rank`
3. `best10 avg_rank`
4. `deal_in_rate`
5. `win_rate`

## 8. PPO diagnostics

`policy_ratio` を上げて環境が自己閉じしすぎる場合、以下に出る可能性がある。

- entropy 低下
- `max_prob_mean` 上昇
- `clip_fraction` 上昇
- `log_ratio_p01` 悪化
- `ratio_max` 悪化

`exp024 seed42` reference:

| metric | value |
|---|---:|
| entropy_last | 0.2721 |
| clip_last | 0.0858 |
| log_ratio_p01_last | -0.4125 |
| ratio_max_last | 3.5555 |
| max_prob_mean_last | 0.8868 |

## 9. 判定

### 採用寄り

- `policy_ratio=0.75` が `tail10` / `final` で reference を改善し、diagnostics も悪化しない
- `policy_ratio=1.00` も安定するなら、baseline actor 依存を下げられる可能性がある

### 保留

- `0.75` と `0.50` が同等
- `1.00` は良いが diagnostics が怪しい
- seed42 だけでは判断が割れる

### 見送り

- `0.75` / `1.00` ともに reference より悪い
- pure policy selfplay で entropy collapse や late drift が出る

## 10. 次アクション

- `0.75` が明確に良ければ、3seed 化する
- `1.00` が良ければ、pure policy selfplay を 3seed 化する
- どちらも悪ければ、`policy_ratio=0.50` を維持して Stage02b ルール拡張に進む
