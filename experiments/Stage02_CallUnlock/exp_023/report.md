# Experiment Report: exp_023

作成日: 2026-05-02  
Stage: `Stage02_CallUnlock`

参照:

- `experiments/Stage02_CallUnlock/exp_023/runbook.md`
- `experiments/Stage02_CallUnlock/exp_022/report.md`
- `experiments/Stage02_CallUnlock/exp_022/independent_review.md`
- `experiments/Stage02_CallUnlock/exp_021/report.md`

## 1. 要約

`exp_023` は、`exp_022` の long-run collapse の主因候補として浮上した **mixed PPO に baseline actor sample を PPO ratio 付きで混ぜている設計** を検証した実験である。

`exp_022` では、`B no-anchor lr1e-4 clip0.15` が短期的には強かったが、60 cycle まで伸ばすと `c40` 以降に 3 seed すべてで collapse した。

`exp_023` では、rule-based baseline agent は selfplay の卓に残しつつ、learner の PPO policy update は `actor_type="policy"` sample のみに限定した。

実験差分:

```yaml
training:
  rule_mix_learner:
    ppo_mode: "separated"
    baseline_imitation_epochs: 0
    policy_ppo_epochs: 1
```

結論:

- 3 seed すべて正常完了
- `exp_022` の `c40` 以降 collapse は消えた
- 3 seed 平均 final は `3.132 -> 2.167` に改善
- 3 seed 平均 tail10 は `2.985 -> 2.199` に改善
- best / best5 / best10 も `exp_022` より改善
- `log_ratio_p01`, `ratio_max`, `clip_fraction`, `entropy_last`, `max_prob_mean` が大幅に正常化
- mixed PPO に baseline actor sample を PPO ratio 付きで混ぜていたことが collapse の主要因だったと判断する

これは Stage2a RL の現時点で最も大きい改善であり、今後の baseline は `separated policy-only PPO` に寄せるべきである。

## 2. 背景

### 2.1 exp_022 の症状

`exp_022` は、`exp_021` で有望だった `B no-anchor lr1e-4 clip0.15` を 60 cycle まで延長した実験である。

3 seed 平均:

| 指標 | avg_rank |
|---|---:|
| imitation | 2.352 |
| final | 3.132 |
| best | 2.147 |
| best5 | 2.233 |
| best10 | 2.271 |
| tail5 | 3.076 |
| tail10 | 2.985 |
| tail20 | 2.816 |

cycle window:

| Window | avg_rank |
|---|---:|
| c00-c09 | 2.341 |
| c10-c19 | 2.320 |
| c20-c29 | 2.303 |
| c30-c39 | 2.343 |
| c40-c49 | 2.648 |
| c50-c59 | 2.985 |

`exp_022` は best / best5 / best10 では改善を出していたが、tail では完全に崩壊した。

### 2.2 independent review の指摘

`independent_review.md` では、mixed PPO の baseline sample の扱いが主因候補として指摘された。

現状の mixed PPO では baseline actor sample について:

```text
action source: rule-based baseline
old_log_prob: learned policy がその rule action に付けた log_prob
```

になっていた。

しかし PPO ratio は本来、旧方策が実際に sample した action に対して計算する。

```text
ratio = pi_new(a|s) / pi_old(a|s)
```

baseline action は learned policy から sample されていないため、これを PPO ratio 付き policy loss に混ぜると off-policy な混合になる。

policy が尖ると、rule baseline が選ぶ action に learned policy が極端に低い確率を付けることがある。その状態で learner 更新を行うと `ratio` が爆発しうる。

`entropy_coef=0.003` probe の cycle59 diagnostics は、この構造と整合していた。

| metric | value |
|---|---:|
| `max_prob_mean` | 0.9737 |
| `max_prob_p95` | 1.0000 |
| `log_ratio_p01` | -35.2108 |
| `log_ratio_p99` | 0.8532 |
| `ratio_p99` | 2.3472 |
| `ratio_max` | 7.89e6 |

## 3. 実験条件

`exp_023` の新規条件は `S: noanchor_lr1e4_clip015_separated_policy_only`。

全 seed 共通:

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
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix.save_baseline_actions = true`
- `training.rule_mix_learner.enabled = true`
- `training.rule_mix_learner.ppo_mode = "separated"`
- `training.rule_mix_learner.baseline_imitation_epochs = 0`
- `training.rule_mix_learner.policy_ppo_epochs = 1`
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

run:

| seed | run_dir |
|---:|---|
| 42 | `runs/20260501_stage2a_exp023_S_separated_noanchor_lr1e4_clip015_seed42_8728a44a` |
| 43 | `runs/20260501_stage2a_exp023_S_separated_noanchor_lr1e4_clip015_seed43_9433fd5f` |
| 44 | `runs/20260501_stage2a_exp023_S_separated_noanchor_lr1e4_clip015_seed44_e75fb085` |

## 4. 主結果

avg_rank は低いほど良い。

### 4.1 seed 別結果

| seed | final | best | best5 | best10 | tail5 | tail10 | tail20 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.155 | c58 1.975 | 2.115 | 2.143 | 2.129 | 2.153 | 2.169 |
| 43 | 2.265 | c57 2.135 | 2.248 | 2.265 | 2.250 | 2.273 | 2.272 |
| 44 | 2.080 | c25 2.010 | 2.104 | 2.138 | 2.150 | 2.171 | 2.159 |

読み:

- 3 seed すべてで final / tail が良い
- seed42/43 は best が後半 `c57-c58` に出ている
- seed44 は best が `c25` だが、tail も良好に維持している
- `exp_022` のような c40 以降の collapse は見えない

### 4.2 exp022 mixed との比較

3 seed 平均:

| condition | final | best | best5 | best10 | tail5 | tail10 | tail20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp022 mixed | 3.132 | 2.147 | 2.233 | 2.271 | 3.076 | 2.985 | 2.816 |
| exp023 separated | 2.167 | 2.040 | 2.156 | 2.182 | 2.176 | 2.199 | 2.200 |
| diff | -0.965 | -0.107 | -0.077 | -0.089 | -0.900 | -0.786 | -0.617 |

読み:

- tail 系が劇的に改善
- best 系も改善
- final が `2.167` まで改善し、collapse 後ではなく最終 checkpoint も採用可能な水準になった

## 5. cycle window

3 seed 平均:

| condition | c00-c09 | c10-c19 | c20-c29 | c30-c39 | c40-c49 | c50-c59 |
|---|---:|---:|---:|---:|---:|---:|
| exp022 mixed | 2.341 | 2.320 | 2.303 | 2.343 | 2.648 | 2.985 |
| exp023 separated | 2.302 | 2.274 | 2.264 | 2.219 | 2.201 | 2.199 |

読み:

- exp022 mixed は `c40-c49` から悪化し、`c50-c59` で collapse
- exp023 separated は `c30-c59` で安定して良い
- 後半ほど悪くなるのではなく、後半でも改善帯を維持している

seed 別:

| seed | c00-c09 | c10-c19 | c20-c29 | c30-c39 | c40-c49 | c50-c59 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.299 | 2.274 | 2.282 | 2.200 | 2.184 | 2.153 |
| 43 | 2.275 | 2.312 | 2.347 | 2.298 | 2.272 | 2.273 |
| 44 | 2.334 | 2.235 | 2.162 | 2.159 | 2.147 | 2.171 |

3 seed すべてで `c50-c59` が `2.15〜2.27` に収まっている。

## 6. win / deal-in

| condition | final win_rate | final deal_in_rate | tail10 win_rate | tail10 deal_in_rate |
|---|---:|---:|---:|---:|
| exp022 mixed avg | 0.0744 | 0.2110 | 0.0981 | 0.2038 |
| exp023 separated avg | 0.2279 | 0.1928 | 0.2304 | 0.1861 |

読み:

- collapse で落ちていた win_rate が大きく回復
- deal_in_rate も少し改善
- 主改善は win_rate の回復
- これは `exp_022` の collapse が「放銃爆増」ではなく「勝てなくなる」形だったことと整合する

## 7. PPO diagnostics

`exp023 separated` は `CQ-0281` 後の run なので、log_ratio / max_prob diagnostics が入っている。

### 7.1 終盤 diagnostics

3 seed 平均:

| metric | exp023 separated avg |
|---|---:|
| `entropy_last` | 0.2841 |
| `clip_last` | 0.0897 |
| `log_ratio_p01_last` | -0.4537 |
| `ratio_max_last` | 6.0594 |
| `max_prob_mean_last` | 0.8853 |

比較用に、`entropy_coef=0.003 mixed probe` seed42 の cycle59 は以下だった。

| metric | entropy probe mixed seed42 |
|---|---:|
| `entropy_last` | 0.0651 |
| `clip_last` | 0.3426 |
| `log_ratio_p01` | -35.2108 |
| `ratio_max` | 7.89e6 |
| `max_prob_mean` | 0.9737 |

読み:

- `log_ratio_p01` の極端な負 tail が消えた
- `ratio_max` が常識的な範囲に戻った
- `clip_fraction` が大きく下がった
- entropy が高めに維持された
- `max_prob_mean` は上がっているが、collapse 水準ではない

### 7.2 seed 別終盤 diagnostics

| seed | entropy_last | clip_last | log_ratio_p01_last | ratio_max_last | max_prob_mean_last |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.3116 | 0.1055 | -0.4564 | 4.52 | 0.8736 |
| 43 | 0.3022 | 0.0769 | -0.4086 | 7.33 | 0.8768 |
| 44 | 0.2385 | 0.0866 | -0.4960 | 6.33 | 0.9056 |

3 seed すべてで diagnostics が安定している。

## 8. 解釈

### 8.1 主因仮説は強く支持された

`exp_023` の結果は、independent review の主張を強く支持する。

```text
mixed PPO に baseline actor sample を PPO ratio 付きで混ぜていたことが、
exp022 の long-run collapse の主要因だった。
```

理由:

- 差分はほぼ `ppo_mode="separated"` と `baseline_imitation_epochs=0` のみ
- baseline agent は selfplay の卓には残っている
- baseline sample は保存されている
- PPO policy update だけが policy sample に限定された
- その結果、collapse が 3 seed で消えた
- ratio tail / entropy / clip diagnostics も同時に正常化した

### 8.2 baseline agent は「環境形成役」としては有効

今回、baseline agent は selfplay から除外していない。

したがって、当初の懸念であった:

```text
報酬をきちんと取ってくれる agent がいないと学習が進まないのではないか
```

に対しては、以下の整理ができる。

- baseline agent を卓に混ぜること自体は有効
- ただし baseline actor の action を PPO ratio policy loss に混ぜるのは危険
- baseline action を学習に使うなら、PPO ではなく imitation / auxiliary route に分けるべき

### 8.3 no-anchor は再評価されるべき

`exp_022` では no-anchor が長期 collapse したため、anchor なしは危険に見えていた。

しかし `exp_023` では no-anchor のまま 60 cycle tail が安定した。

つまり、問題は no-anchor 自体ではなく、mixed PPO baseline sample の扱いだった可能性が高い。

今後は:

```text
separated policy-only PPO + no-anchor
```

を新しい Stage2a baseline として扱うのが自然である。

## 9. 次アクション

### 9.1 CQ

次に切るべき CQ は以下。

```text
CQ-0282: Stage2a rule_mix_learner default を separated policy-only PPO に変更する
```

目的:

- baseline actor sample を PPO policy loss / ratio diagnostics に混ぜない
- rule-based baseline agent は selfplay opponent / environment shaper として残す
- baseline sample を使う場合は imitation / auxiliary route に分離する

受け入れ条件:

- default config の `training.rule_mix_learner.ppo_mode` を `"separated"` に変更
- default config の `training.rule_mix_learner.baseline_imitation_epochs` を `0` に明示
- `policy_ppo` learner は `actor_type="policy"` sample のみを使う
- `mixed_ppo` を使う場合は明示 opt-in とし、warning または diagnostics に off-policy 注意を出す
- summary/report に learner stage が `policy_ppo` として記録される

### 9.2 次実験

`exp_024` 候補:

1. `separated policy-only PPO` を新 baseline として 60 cycle 追加検証
2. `target_kl early stop` を追加し、さらに安定化するかを見る
3. baseline imitation を弱く戻す実験

ただし、`exp_023` の時点で long-run stability は大きく改善しているため、次にやるなら target_kl よりも、まず report / CQ / config 整理を優先する。

## 10. 結論

`exp_023` の結論:

1. `separated policy-only PPO` は 3 seed で collapse を解消した
2. `tail10` は `2.985 -> 2.199` に改善した
3. `final` は `3.132 -> 2.167` に改善した
4. best 系も mixed PPO より良い
5. `log_ratio_p01`, `ratio_max`, `clip_fraction`, entropy が正常化した
6. mixed PPO に baseline actor sample を PPO ratio 付きで混ぜる設計が、exp022 collapse の主要因だったと判断する
7. 今後の Stage2a baseline は `separated policy-only PPO` に寄せるべきである

これは Stage2a の RL 更新安定化における大きな前進である。
