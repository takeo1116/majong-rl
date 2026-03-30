# Experiment Runbook: exp_004

作成日: 2026-03-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_002/runbook.md`
- `experiments/Stage02_CallUnlock/exp_002/report.md`
- `experiments/Stage02_CallUnlock/exp_003/report.md`
- `configs/stage2a_core_minimal_mixed_search_baseline.yaml`

## 1. 背景

`exp_002` では、Stage02a A `core_minimal` を固定して

- `ppo_mode="mixed"`
- `ppo_mode="separated"`

を比較した。

その結果、

- `separated`: 20 cycle を通して安定
- `mixed`: `policy_ratio=0.25`, `baseline_sample_weight=0.5` でも discard drift が再発し不採用

という結論になった。

一方で `exp_003` により、`separated` baseline 上では A/B/C の feature 比較が安定に回ること、
さらに Stage02a learner の imitation throughput も実験可能水準まで改善したことが確認できた。

したがって次の主課題は、**Stage02a で mixed PPO を安定に回せる条件を探すこと**である。

## 2. 問い

Stage02a A `core_minimal` において、mixed PPO を安定化させるには次のどの要因が効くか。

1. policy sample 比率を増やすこと (`policy_ratio`)
2. baseline sample の重みをさらに下げること (`baseline_sample_weight`)
3. anchor を強めること (`policy_anchor.coef`)
4. PPO の update を穏やかにすること (`lr`, `clip_epsilon`, `max_grad_norm`)

## 3. 仮説

現時点の仮説は次の順。

1. `mixed` が壊れる主因は、baseline 混入率がまだ高く、discard branch が off-policy 的に引っ張られていること
2. `baseline_sample_weight=0.5` でも baseline 側の PPO 寄与がまだ強い可能性がある
3. `anchor_kl_discard` を見る限り、anchor が弱く discard drift を止め切れていない可能性がある
4. それでも崩れる場合、`lr=3e-4`, `clip_epsilon=0.15` は Stage02 mixed に対して強すぎる可能性がある

## 4. この実験の位置づけ

この `exp_004` は、feature 比較ではなく **mixed 安定化探索** である。

- 対象は A `core_minimal` 固定
- 目的は「20 cycle 完走した」だけでなく、「PPO 指標が安定し eval が崩れない mixed 条件」を見つけること
- `exp_002` の `mixed` 失敗 run と `separated` 安定 run を参照点とする
- 今回は追加実装なし、既存 config override のみで切り分ける

## 5. 参照点（既存 run の流用）

### Stable control

- `exp_002` の A2 `separated` control
- `ppo_mode="separated"`
- final `avg_rank=2.555`, `win_rate=0.2312`
- final `ratio_mean=1.0035`, `clip_fraction=0.2462`, `anchor_kl_discard=0.0688`

### Unstable mixed reference

- `exp_002` の A1 `mixed` reference
- `ppo_mode="mixed"`
- `policy_ratio=0.25`, `baseline_sample_weight=0.5`
- final `avg_rank=3.45`, `win_rate=0.0484`
- final `ratio_mean=413438`, `clip_fraction=0.4898`, `anchor_kl_discard=5.6866`

これらは新規実行せず、比較参照点として流用する。

## 6. 共通条件

ベース config:

- `configs/stage2a_core_minimal_mixed_search_baseline.yaml`

固定条件:

- A `core_minimal` feature set
- `training.rule_mix_learner.ppo_mode = "mixed"`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`
- `training.imitation_eval.eval_each_chunk = false`
- `training.imitation_eval.num_matches = 50`
- `training.policy_anchor.reference = "imitation_fixed"`
- selfplay / eval workers = `10`, `worker_num_threads = 1`

意図:

- imitation 直後 eval も保存し、
  - imitation 直後は良いのに PPO で崩れるのか
  - そもそも imitation から差が出ているのか
  を後から切り分けられるようにする

## 7. 比較条件

### M1 `policy_ratio=0.50, bsw=0.50`

- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.50`
- anchor / lr / clip は baseline のまま

狙い:

- policy sample 比率を上げるだけで mixed が安定するかを見る

### M2 `policy_ratio=0.50, bsw=0.25`

- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`

狙い:

- baseline sample の PPO 寄与をさらに弱めると discard drift が改善するかを見る

### M3 `M2 + stronger_anchor`

- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.coef = 1.0`

狙い:

- discard drift を anchor 強化で抑え込めるかを見る

### M4 `M3 + softer_ppo_step`

- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.coef = 1.0`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.10`
- `training.max_grad_norm = 0.30`

狙い:

- mixed がまだ壊れるなら、update 強度そのものが主因かを切り分ける

## 8. 実行順

1. M1
2. M2
3. M3
4. M4

理由:

- まず off-policy 混入率の問題を切る
- 次に baseline sample 重みを切る
- それでもだめなら anchor 強化を見る
- 最後に update 強度まで落として mixed を救えるかを見る

## 9. 成功判定

最低条件:

- 20 cycle 完走
- learner loss が終盤で吹き上がらない
- `ratio_mean` が 1 近傍に留まる
- `clip_fraction` が 0.3 台前半以下に収まる
- `anchor_kl_discard` が 0.2 前後以下に収まる
- eval `avg_rank` / `win_rate` が後半で崩れ切らない

望ましい条件:

- imitation_eval が A control と大きくは変わらず、PPO 後も悪化しない
- `cycle_00` から `cycle_19` にかけて eval を維持または改善する

## 10. 判定の見方

### mixed が安定したとみなす条件

- `ratio_mean` が異常値にならない
- `clip_fraction` が高止まりしない
- `anchor_kl_discard` が 1 を大きく超えて増えない
- final eval が `separated` control と同程度か、それに近い

### 次に進む条件

- M1 or M2 で既に安定するなら、その条件を mixed baseline 候補にする
- M3 で初めて安定するなら、Stage02 mixed では anchor 強化が必須とみなす
- M4 でも壊れるなら、次は hyperparameter ではなく mixed 設計自体の見直しを検討する

## 11. 実行方法

### Driver

```bash
./.venv/bin/python scripts/local/stage2/exp_004_driver.py
```

### 個別 run のベース config

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_search_baseline.yaml \
  --base-dir runs \
  --override \
  'experiment.name="stage2a_exp004_M1_pr050_bsw05_seed42"' \
  'experiment.global_seed=42'
```

他の条件は driver が上書きする。

## 12. 期待するアウトプット

- `experiments/Stage02_CallUnlock/exp_004/run_map.json`
- report に転記する主要数値
- imitation 直後 eval と final eval の比較

## 13. 次アクション判定

### mixed 安定条件が見つかった場合

- その条件を `exp_005` などで再確認
- 必要なら 2-3 seed に広げる
- その後、Stage02 の baseline 更新候補にする

### mixed 安定条件が見つからなかった場合

- mixed 設計自体の見直しへ進む
- 候補:
  - discard / optional で別の mixed 方針
  - baseline sample を PPO ではなく BC 寄りに混ぜる
  - actor_type 別 diagnostics の追加
