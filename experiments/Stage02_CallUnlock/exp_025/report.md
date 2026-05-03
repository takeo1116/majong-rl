# Experiment Report: exp_025

作成日: 2026-05-03  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_025` では、`exp_024` の有望設定を固定し、selfplay に混ぜる learned policy actor の比率を上げたときに性能が改善するかを確認した。

比較対象:

- reference: `exp_024` `policy_ratio=0.50`
- probe: `policy_ratio=0.75`
- probe: `policy_ratio=1.00`

CQ-0282 以降、rule-based baseline actor の sample は PPO policy loss に使わない。したがって `policy_ratio` は、PPO に使える on-policy sample 量と、selfplay 卓環境に残す rule-based baseline actor 量のトレードオフを表す。

## 2. 固定条件

`exp_024` と同じ条件を使った。

- separated policy-only PPO
- no anchor
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.15`
- `training.entropy_coef = 0.0`
- `training.value_loss_coef = 0.125`
- `training.rule_mix_learner.ppo_mode = "separated"`
- `training.rule_mix_learner.baseline_imitation_epochs = 0`
- `training.rule_mix_learner.policy_ppo_epochs = 1`
- `training.rule_mix_learner.allow_mixed_offpolicy_baseline = false`
- `feature_encoder.tile_presence_flags = true`
- `model.value_hidden_dims = [256,128]`
- `model.semantic_aux.tile_presence_flags_semantic_only = false`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `training.multi_cycle.num_cycles = 60`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`

変更したもの:

- `training.rule_mix.policy_ratio`

## 3. Run 一覧

| condition | seed | run_dir |
|---|---:|---|
| P075 | 42 | `runs/20260502_stage2a_exp025_P075_onwide_separated_seed42_43b41c0b` |
| P075 | 43 | `runs/20260503_stage2a_exp025_P075_onwide_separated_seed43_78896cfa` |
| P075 | 44 | `runs/20260503_stage2a_exp025_P075_onwide_separated_seed44_e3e420f0` |
| P100 | 42 | `runs/20260502_stage2a_exp025_P100_onwide_separated_seed42_c00c1bdd` |
| P100 | 43 | `runs/20260503_stage2a_exp025_P100_onwide_separated_seed43_419f75a6` |
| P100 | 44 | `runs/20260503_stage2a_exp025_P100_onwide_separated_seed44_be2c3435` |

Reference:

- `exp_024` P050 3seed: `experiments/Stage02_CallUnlock/exp_024/run_map.json`

## 4. 主結果

### 4.1 3seed 平均

| condition | n | final | best | best5 | best10 | tail5 | tail10 | tail20 | win | deal-in | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P050 exp024 | 3 | 2.1417 | 2.0267 | 2.0550 | 2.0835 | 2.1700 | 2.1690 | 2.1729 | 0.2216 | 0.1908 | 28519.3 |
| P075 exp025 | 3 | 2.1600 | 1.9767 | 2.0403 | 2.0670 | 2.1513 | 2.1617 | 2.1498 | 0.2172 | 0.1863 | 28548.7 |
| P100 exp025 | 3 | 2.1150 | 2.0083 | 2.0337 | 2.0582 | 2.1710 | 2.1545 | 2.1409 | 0.2219 | 0.1880 | 29478.3 |

### 4.2 seed ばらつき

| condition | final_sd | best10_sd | tail10_sd | tail20_sd | win_sd | deal_sd |
|---|---:|---:|---:|---:|---:|---:|
| P050 exp024 | 0.0671 | 0.0224 | 0.0460 | 0.0386 | 0.0183 | 0.0089 |
| P075 exp025 | 0.0577 | 0.0140 | 0.0207 | 0.0173 | 0.0116 | 0.0093 |
| P100 exp025 | 0.0200 | 0.0291 | 0.0170 | 0.0264 | 0.0046 | 0.0084 |

### 4.3 seed 別詳細

| condition | seed | final | best | best_cycle | best10 | tail10 | tail20 | win | deal-in | entropy | max_prob | ratio_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P050 exp024 | 42 | 2.170 | 2.015 | 58 | 2.089 | 2.139 | 2.158 | 0.2013 | 0.1818 | 0.2944 | 0.8820 | 3.6685 |
| P050 exp024 | 43 | 2.190 | 2.010 | 26 | 2.059 | 2.146 | 2.144 | 0.2270 | 0.1996 | 0.2886 | 0.8814 | 3.9279 |
| P050 exp024 | 44 | 2.065 | 2.055 | 27 | 2.103 | 2.222 | 2.217 | 0.2366 | 0.1908 | 0.2716 | 0.8891 | 3.6843 |
| P075 exp025 | 42 | 2.115 | 1.970 | 36 | 2.061 | 2.143 | 2.133 | 0.2142 | 0.1933 | 0.2876 | 0.8853 | 6.0630 |
| P075 exp025 | 43 | 2.225 | 2.015 | 16 | 2.083 | 2.184 | 2.167 | 0.2300 | 0.1757 | 0.2749 | 0.8890 | 11.2660 |
| P075 exp025 | 44 | 2.140 | 1.945 | 57 | 2.057 | 2.158 | 2.150 | 0.2074 | 0.1900 | 0.2440 | 0.9025 | 7.6237 |
| P100 exp025 | 42 | 2.115 | 2.055 | 53 | 2.092 | 2.155 | 2.171 | 0.2171 | 0.1974 | 0.2434 | 0.9017 | 6.1827 |
| P100 exp025 | 43 | 2.095 | 1.965 | 35 | 2.038 | 2.137 | 2.131 | 0.2223 | 0.1852 | 0.3389 | 0.8621 | 16.7532 |
| P100 exp025 | 44 | 2.135 | 2.005 | 43 | 2.045 | 2.171 | 2.120 | 0.2263 | 0.1813 | 0.2255 | 0.9082 | 7.5570 |

## 5. PPO diagnostics

### 5.1 final cycle 平均

| condition | entropy | clip | log_ratio_p01 | ratio_max | max_prob | policy steps |
|---|---:|---:|---:|---:|---:|---:|
| P050 exp024 | 0.2849 | 0.0846 | -0.4248 | 3.7602 | 0.8842 | 61036 |
| P075 exp025 | 0.2688 | 0.0982 | -0.5453 | 8.3176 | 0.8923 | 91787 |
| P100 exp025 | 0.2693 | 0.0813 | -0.4294 | 10.1643 | 0.8907 | 123105 |

### 5.2 tail10 平均

| condition | entropy | clip | log_ratio_p01 | ratio_max | max_prob |
|---|---:|---:|---:|---:|---:|
| P050 exp024 | 0.3084 | 0.0874 | -0.4203 | 4.7884 | 0.8748 |
| P075 exp025 | 0.2648 | 0.0866 | -0.4786 | 8.8938 | 0.8933 |
| P100 exp025 | 0.2704 | 0.0874 | -0.4765 | 10.4236 | 0.8901 |

## 6. 解釈

### 6.1 P100 は成立している

`policy_ratio=1.00` は rule-based baseline actor を selfplay から完全に外す pure learned-policy selfplay である。それにもかかわらず、60 cycle × 3seed で collapse せず、final / best10 / tail20 が最良になった。

これは重要な結果である。imitation で初期化した後、PPO 段階は learned policy のみで selfplay を回しても学習が成立している。

### 6.2 P075 は best が強いが final は不安定

P075 は best 平均が最も良い。

| condition | best |
|---|---:|
| P050 | 2.0267 |
| P075 | 1.9767 |
| P100 | 2.0083 |

一方で final は P100 より悪く、seed43 で `final=2.225` まで沈んだ。P075 はピーク性能は高いが、終盤維持は P100 ほど安定していない。

### 6.3 P100 は diagnostics が完全に安全ではないが、性能上は最有力

P100 は `ratio_max` が高い。これは少数 sample で大きな ratio が出ていることを示す。一方で、`clip_fraction` と `log_ratio_p01` は P075 より悪くない。

- P075 final `clip=0.0982`, `log_ratio_p01=-0.5453`
- P100 final `clip=0.0813`, `log_ratio_p01=-0.4294`

`max_prob` は P050 より高いが、collapse と判断するほどではない。現時点では P100 を次の基準にしてよい。

## 7. 結論

`policy_ratio=1.00` を次の Stage02a 基準候補にする。

理由:

- final avg_rank が最良
- best10 / tail20 も最良
- seed ばらつきが小さい
- rule-based baseline actor に依存しない構成として綺麗
- PPO 段階で pure learned-policy selfplay が成立している

`policy_ratio=0.75` は peak 性能が高いため完全には棄却しないが、次の probe 基準としては P100 を優先する。

## 8. 次アクション

ClaudeCode review で指摘された `reward.point_delta_scale` 問題を確認する。

現状、Stage2a config には `reward:` ブロックがなく、C++ default の `point_delta_scale=1.0` が使われている可能性が高い。Stage1 では `point_delta_scale=0.0001` を使っているため、Stage2a の reward / return / value target が想定より 10000 倍大きくなっている可能性がある。

次の 1seed probe:

```text
base: exp025 P100
change only:
  reward.type = "point_delta"
  reward.point_delta_scale = 0.0001
seed: 42
```

他の条件は変えない。

判定では、policy performance だけでなく以下も見る。

- value_loss が大きく下がるか
- entropy / clip / log_ratio / max_prob が悪化しないか
- semantic eval の yaku/terminal confidence が改善するか
- deal_in risk AUC / PR AUC が改善するか

この probe が良ければ 3seed 化し、悪ければ P100 の raw-scale 条件を暫定標準として Stage02b ルール拡張へ進む判断をする。
