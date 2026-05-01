# Experiment Runbook: exp_023

作成日: 2026-05-01  
Stage: `Stage02_CallUnlock`

## 1. 目的

`exp_023` の目的は、`exp_022` で観測された long-run collapse の主因候補である **mixed PPO に baseline actor sample を PPO ratio 付きで混ぜている設計** を検証することである。

具体的には、rule-based baseline agent は selfplay の卓には残しつつ、learner の PPO policy update は `actor_type="policy"` sample のみに限定する。

これにより、次を切り分ける。

```text
baseline agent が卓にいること自体が必要なのか
baseline agent の action を PPO ratio 付き policy loss に混ぜることが問題なのか
```

`exp_022` では no-anchor 条件が短期的には強かったが、60 cycle では `c40` 以降に collapse した。`exp_023` では、no-anchor の改善力を保ちつつ、baseline sample 由来の ratio 暴走を外すことで long-run stability が改善するかを見る。

## 2. 背景

### 2.1 exp_021 / exp_022 の結果

`exp_021` では、`B: no-anchor, lr=1e-4, clip=0.15` が 30 cycle 条件で最も良かった。

| Group | imitation | final | best | best5 | tail5 | tail10 |
|---|---:|---:|---:|---:|---:|---:|
| A anchor075 lr1e-4 clip0.15 | 2.352 | 2.360 | 2.188 | 2.311 | 2.392 | 2.355 |
| B no-anchor lr1e-4 clip0.15 | 2.352 | 2.327 | 2.148 | 2.246 | 2.288 | 2.303 |

`exp_022` では、この B 条件を 60 cycle まで延長した。

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

読み:

- no-anchor B は短期的に強い
- `c20-c30` 付近までは imitation より良い
- しかし `c40` 以降に 3 seed すべてで collapse
- collapse は win_rate 低下、entropy 低下、clip_fraction 上昇と同期

### 2.2 entropy_coef=0.003 probe

`exp_022` 後、`entropy_coef=0.003` の seed42 probe を実行した。

run:

```text
runs/20260501_stage2a_entropy_probe_noanchor_ec003_seed42_2c8100e5
```

比較:

| condition | final | best | best5 | best10 | tail10 |
|---|---:|---:|---:|---:|---:|
| exp022 seed42 entropy=0.0 | 2.910 | 2.185 | 2.269 | 2.307 | 2.785 |
| entropy=0.003 probe | 3.255 | 2.185 | 2.315 | 2.353 | 3.005 |

診断:

| condition | entropy_last | clip_last |
|---|---:|---:|
| entropy=0.0 | 0.0662 | 0.3378 |
| entropy=0.003 | 0.0651 | 0.3426 |

`entropy_coef=0.003` は collapse を止めなかった。したがって、単純な entropy bonus よりも、ratio / update 構造側の問題を優先して調べる。

### 2.3 independent review の指摘

`experiments/Stage02_CallUnlock/exp_022/independent_review.md` では、collapse の主因候補として以下が挙げられた。

現状の mixed PPO では、baseline actor sample について:

```text
action source: rule-based baseline
old_log_prob: learned policy がその rule action に付けた log_prob
```

になっている。

しかし PPO ratio は本来、旧 policy が実際に sample した action に対して、

```text
ratio = pi_new(a|s) / pi_old(a|s)
```

を計算する前提である。

baseline actor の action は learned policy から sample されていないため、この sample を PPO ratio 付き policy loss に混ぜると、off-policy な混合になる。

特に policy が尖ると、rule baseline が選んだ action に learned policy が非常に低い確率を付けることがある。このとき `old_log_prob` が極端に小さくなり、learner 更新後の `ratio` が爆発しうる。

`entropy_coef=0.003` probe の終盤 diagnostics はこの構造と整合する。

| metric | cycle59 value |
|---|---:|
| `max_prob_mean` | 0.9737 |
| `max_prob_p95` | 1.0000 |
| `log_ratio_p01` | -35.2108 |
| `log_ratio_p99` | 0.8532 |
| `ratio_p99` | 2.3472 |
| `ratio_max` | 7.89e6 |

特に discard branch の `max_prob_mean` は終盤 0.99 近くまで上がっており、collapse の主戦場は discard policy と見られる。

## 3. 今回の問い

`exp_023` で答えたい問いは以下。

1. baseline actor sample を PPO ratio 更新から外すと、`c40` 以降の collapse は軽減するか
2. no-anchor B の短期改善力は維持されるか
3. `log_ratio_p01`, `ratio_max`, `clip_fraction`, `max_prob_mean` は改善するか
4. baseline agent を卓に残すだけで、報酬を取れる相手/環境として十分機能するか
5. mixed PPO collapse の主因が baseline sample の ratio 混入だったと言えるか

## 4. 実験方針

既存 `exp_022` を historical reference とし、新規には `separated policy-only PPO` 条件を 3 seed で実行する。

### 4.1 reference: exp_022 mixed PPO

再実行しない。既存 run を比較対象として使う。

| seed | run_dir |
|---:|---|
| 42 | `runs/20260501_stage2a_exp022_Blong_noanchor_lr1e4_clip015_seed42_9d2e15a6` |
| 43 | `runs/20260501_stage2a_exp022_Blong_noanchor_lr1e4_clip015_seed43_ce37fabb` |
| 44 | `runs/20260501_stage2a_exp022_Blong_noanchor_lr1e4_clip015_seed44_34ec042d` |

### 4.2 new: exp_023 separated policy-only PPO

新規に 3 seed 実行する。

条件名:

```text
S: noanchor_lr1e4_clip015_separated_policy_only
```

差分:

```yaml
training:
  rule_mix_learner:
    ppo_mode: "separated"
    baseline_imitation_epochs: 0
    policy_ppo_epochs: 1
```

意味:

- rule-based baseline agent は selfplay に混ぜる
- baseline actor sample は shard に保存される
- baseline imitation stage は実行しない
- PPO policy update は `actor_type="policy"` sample のみに限定する
- baseline actor sample は PPO ratio / policy loss に入れない

`training.rule_mix.save_baseline_actions` は `true` のままにする。

理由:

- baseline sample が生成されていることを actor_type counts で確認できる
- 後から baseline sample の `old_log_prob` 分布を解析できる
- learner 側で filter されているかを検証しやすい

## 5. 条件定義

全条件共通:

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `training.multi_cycle.num_cycles = 60`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `training.policy_anchor.enabled = false`
- `training.policy_anchor.coef = 0.0`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.15`
- `training.entropy_coef = 0.0`
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
- `evaluation.num_workers = 10`
- `training.imitation_eval.num_workers = 10`
- expected shard semantics: `sample_semantics_version = 3`

新規条件:

| label | seed | purpose |
|---|---:|---|
| `S_separated_noanchor_lr1e4_clip015_seed42` | 42 | exp022 seed42 mixed PPO との直接比較 |
| `S_separated_noanchor_lr1e4_clip015_seed43` | 43 | exp022 seed43 mixed PPO との直接比較 |
| `S_separated_noanchor_lr1e4_clip015_seed44` | 44 | exp022 seed44 mixed PPO との直接比較 |

## 6. 必須観測

### 6.1 Performance

各 run で以下を集計する。

- imitation avg_rank / win_rate / deal_in_rate
- final avg_rank / win_rate / deal_in_rate
- best cycle
- best5 window
- best10 window
- tail5
- tail10
- tail20

cycle window:

| Window | meaning |
|---|---|
| c00-c09 | early |
| c10-c19 | middle-1 |
| c20-c29 | exp021/022 と比較できる short-run best zone |
| c30-c39 | transition |
| c40-c49 | exp022 collapse onset |
| c50-c59 | final tail |

### 6.2 PPO diagnostics

`CQ-0281` diagnostics を重視する。

全体 + discard/call branch 別に以下を見る。

- `entropy`
- `clip_fraction`
- `log_ratio_p01`, `log_ratio_p05`, `log_ratio_p50`, `log_ratio_p95`, `log_ratio_p99`
- `ratio_p99`, `ratio_max`
- `max_prob_mean`, `max_prob_p95`, `max_prob_p99`
- `advantage_pos_frac`, `advantage_neg_frac`
- `clip_fraction_adv_pos`, `clip_fraction_adv_neg`
- `mixed_ppo` / `policy_ppo` sample counts

特に重視する比較:

| metric | exp022 mixed symptom | exp023 expected if hypothesis correct |
|---|---:|---:|
| `log_ratio_p01` | `-20〜-38` | 大幅に浅くなる |
| `ratio_max` | `1e6+` tail | 大幅に下がる |
| `clip_fraction` | `0.30+` | 下がる |
| `max_prob_mean` | `0.97+` | 上昇が遅れる / 止まる |
| `entropy_last` | `0.03〜0.13` | 高めに維持 |
| `tail10 avg_rank` | `2.985` | 明確に改善 |

### 6.3 actor_type / learner path validation

実験後、各 cycle で以下を確認する。

- selfplay shard には `actor_type=policy` と `actor_type=baseline` が存在する
- learner stage は `policy_ppo` として記録される
- `baseline_imitation` は `executed=false` または存在しない / updates 0
- PPO diagnostics の sample count が policy samples のみを対象にしている

この確認は重要。`ppo_mode="separated"` が意図通り効いていない場合、実験解釈が崩れる。

## 7. 判定基準

### 7.1 成功

以下を満たすなら、mixed PPO baseline ratio 混入が主因だった可能性が高い。

- `tail10` が exp022 mixed より明確に改善
- `c40-c59` の collapse が軽減
- `log_ratio_p01` の極端な負 tail が消える / 浅くなる
- `ratio_max` が大幅に下がる
- `clip_fraction` が後半 `0.30+` に張り付かない
- `max_prob_mean` の上昇が遅くなる

目安:

```text
tail10 <= 2.45: strong success
tail10 <= 2.60: partial success
tail10 > 2.80: collapse remains
```

### 7.2 失敗

以下なら、baseline ratio 混入だけでは説明できない。

- `tail10` が exp022 mixed と同程度に悪い
- `c40` 以降に同じように collapse
- policy-only sample でも `log_ratio_p01` が極端に落ちる
- discard `max_prob_mean` が 0.97+ まで上がる

この場合、次は以下を検討する。

1. `target_kl` early stop 実装
2. lagged anchor 再導入
3. selfplay temperature schedule
4. value diagnostics / GAE 設定見直し

### 7.3 短期性能だけ落ちる場合

もし collapse は改善するが best / best5 が悪化する場合:

- baseline sample を PPO に混ぜていたことが短期改善を加速していた可能性がある
- ただし長期安定性の代償が大きかった可能性もある

この場合は、baseline sample を PPO ratio に入れず、別経路の imitation loss として弱く使う設計を検討する。

候補:

```text
baseline_imitation_epochs=1
or baseline imitation loss coef を小さくする
```

ただし、まずは `baseline_imitation_epochs=0` で clean に切り分ける。

## 8. 期待される結果パターン

### Pattern A: collapse が大幅に改善

```text
best / best5 は exp022 と同等
tail10 が 2.4〜2.6 程度まで改善
log_ratio_p01 / ratio_max が正常化
```

解釈:

- mixed PPO baseline sample ratio 混入が主因
- 次は separated policy-only を新 baseline にする
- その後 target_kl を追加してさらに安定化を狙う

### Pattern B: collapse は改善するが短期性能が落ちる

```text
best が 2.20〜2.30 へ悪化
tail10 は exp022 より良い
```

解釈:

- baseline sample は短期改善に寄与していた
- ただし PPO ratio 経由で使うのは危険
- baseline sample を imitation / auxiliary route で使う設計を検討

### Pattern C: collapse が残る

```text
best は良いが c40 以降崩れる
log_ratio tail も壊れる
```

解釈:

- baseline sample 混入だけではない
- policy sample だけでも PPO update が強すぎる
- `target_kl` / lr decay / anchor / temperature を優先

### Pattern D: 全体的に悪い

```text
best も tail も悪い
```

解釈:

- baseline sample を policy loss から外すと learning signal が不足
- baseline imitation stage を弱く入れる必要がある
- ただし mixed PPO へ戻すのではなく、PPO と imitation を分離する方向を維持する

## 9. 次アクション

`exp_023` の結果に応じて以下。

1. `S separated policy-only` が良い場合:
   - `CQ-0282`: mixed PPO baseline sample policy-loss exclusion / separated default 化
   - `CQ-0283`: target_kl early stop
   - `exp_024`: separated + target_kl 60cycle

2. `S` が collapse する場合:
   - `CQ-0283`: target_kl early stop を先に実装
   - `exp_024`: noanchor + target_kl probe

3. `S` が安定するが弱い場合:
   - baseline sample を imitation loss として弱く使う実験
   - `baseline_imitation_epochs=1` または imitation coef 付き設計

## 10. 実行メモ

runbook 作成時点では driver 未作成。

想定 override の核:

```bash
training.rule_mix_learner.ppo_mode='"separated"'
training.rule_mix_learner.baseline_imitation_epochs=0
training.rule_mix_learner.policy_ppo_epochs=1
training.rule_mix.save_baseline_actions=true
```

その他は `exp_022` B-long と同一。


## 11. Probe-first 運用

実行負荷を抑えるため、最初は 1 seed probe として seed42 のみ実行する。

### 11.1 seed42 probe

まず以下を実行する。

```text
S_separated_noanchor_lr1e4_clip015_seed42
```

比較対象は `exp_022` seed42。

| condition | final | best | best5 | best10 | tail10 |
|---|---:|---:|---:|---:|---:|
| exp022 mixed seed42 | 2.910 | 2.185 | 2.269 | 2.307 | 2.785 |

probe の判定:

- `tail10 <= 2.60` なら改善ありとして 3seed 化を推奨
- `tail10 <= 2.45` なら strong improvement
- `tail10 > 2.80` なら mixed PPO baseline ratio 混入だけでは collapse を説明しにくい

加えて、以下の diagnostics を seed42 mixed と比較する。

- `log_ratio_p01`
- `ratio_max`
- `clip_fraction`
- `max_prob_mean`
- `entropy_last`
- discard branch の `max_prob_mean`

### 11.2 3seed 化の条件

seed42 probe で以下のいずれかを満たした場合、seed43/44 を追加で実行する。

1. `tail10` が exp022 seed42 より明確に改善する
2. `c40-c59` の collapse が軽くなる
3. `log_ratio_p01` / `ratio_max` が明確に改善する
4. final は悪くても best5 / tail20 が改善し、構造的に見込みがある

### 11.3 report 方針

seed42 probe が悪い場合は、`exp_023/report.md` または `exp_022/report.md` に probe 結果として追記し、3seed 化は行わない。

seed42 probe が良い場合は、`exp_023` を正式実験として seed43/44 まで実行し、3seed 平均で report を作成する。
