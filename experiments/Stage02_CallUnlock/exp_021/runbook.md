# Experiment Runbook: exp_021

作成日: 2026-04-30  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_020/report.md`
- `experiments/Stage02_CallUnlock/exp_020/run_map.json`
- `runs/20260430_stage2a_anchor_probe_noanchor_seed42_af41f851/summary.json`
- `docs/CHANGE_QUEUE.md`

## 1. 背景

`exp_020` では、`CQ-0274`〜`CQ-0279` 修正後の A2 baseline を 3 seed で取り直した。

結果は次の通り。

- 全 seed で imitation より良い best checkpoint に到達
- best cycle は `c20 / c20 / c21` に集中
- しかし final / tail では性能保持に失敗
- PPO diagnostics は壊れていない
- late drift は 3 seed でかなり再現性がある

この結果から、現状の問題は「RL が学べない」ではなく、**良い policy 領域に入った後に保持できないこと**と整理した。

その後、1 seed probe として `policy_anchor.enabled=false` の no-anchor 条件を seed42 で試した。

probe 結果:

| 条件 | imitation | final | best | best5 | tail5 | tail10 |
|---|---:|---:|---:|---:|---:|---:|
| anchor075 seed42 | 2.365 | 2.440 | c20 2.225 | c19-c23 2.327 | 2.455 | 2.392 |
| noanchor seed42 | 2.365 | 2.390 | c15 2.190 | c03-c07 2.308 | 2.337 | 2.340 |

性能面では no-anchor が有望だった。

一方で PPO diagnostics は黄色信号だった。

| 条件 | ratio_mean avg/last/min | clip_fraction avg/last/max |
|---|---:|---:|
| anchor075 seed42 | 1.0048 / 1.0007 / 0.9926 | 0.1101 / 0.0964 / 0.1672 |
| noanchor seed42 | 0.9669 / 0.9206 / 0.9206 | 0.2169 / 0.2510 / 0.2510 |

つまり no-anchor は late drift を弱める可能性があるが、anchor を外した分、PPO 更新が強く/偏りやすくなっている可能性がある。

## 2. 今回の問い

`exp_021` で答えたい問いは 3 つある。

1. no-anchor は 3 seed でも `anchor075` より late drift を弱めるのか
2. no-anchor の PPO diagnostics 悪化は seed42 固有か、それとも再現するのか
3. no-anchor が有望な場合、更新を弱めるなら `clip_epsilon` と `lr` のどちらが有効か

重要なのは、単に no-anchor を採用するかではなく、**anchor の有無と update strength を分離して見ること**である。

## 3. 実験方針

既存 run を最大限使い回す。

### 3.1 使い回す run

#### A: anchor075 baseline, 3 seed

`exp_020` の 3 seed をそのまま使う。

| seed | run_dir |
|---:|---|
| 42 | `runs/20260430_stage2a_exp020_A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed42_0a26a46e` |
| 43 | `runs/20260430_stage2a_exp020_A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed43_6fde9bf9` |
| 44 | `runs/20260430_stage2a_exp020_A2_semaux_light_vhalf_tenpaifix_prnorm_fixedrl_seed44_dac201a9` |

#### B: noanchor baseline, seed42

probe の seed42 を使い回す。

| seed | run_dir |
|---:|---|
| 42 | `runs/20260430_stage2a_anchor_probe_noanchor_seed42_af41f851` |

### 3.2 新規実行する run

今回新規に回すのは 8 run。

| group | label | seeds | new runs | 目的 |
|---|---|---:|---:|---|
| B | `noanchor_lr1e4_clip015` | 43, 44 | 2 | no-anchor 効果の 3 seed 化 |
| C | `noanchor_lr1e4_clip010` | 42, 43, 44 | 3 | no-anchor + PPO clip 縮小 |
| D | `noanchor_lr5e5_clip015` | 42, 43, 44 | 3 | no-anchor + LR 半減 |

合計評価では、既存 4 run + 新規 8 run = 12 run 相当を比較する。

## 4. 条件定義

全条件共通:

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `training.multi_cycle.num_cycles = 30`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
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
- `sample_semantics_version = 3`

### A. `anchor075_lr1e4_clip015`

使い回し: exp_020

- `training.policy_anchor.enabled = true`
- `training.policy_anchor.coef = 0.75`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.15`

### B. `noanchor_lr1e4_clip015`

seed42 は probe を使い回し、seed43/44 を新規実行する。

- `training.policy_anchor.enabled = false`
- `training.policy_anchor.coef = 0.0`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.15`

### C. `noanchor_lr1e4_clip010`

no-anchor のまま PPO clip を狭める。

- `training.policy_anchor.enabled = false`
- `training.policy_anchor.coef = 0.0`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.10`

狙い:

- no-anchor seed42 probe で `clip_fraction` が `0.25` まで上がった
- clip を狭めることで大きな policy ratio 更新を抑えられるかを見る

注意:

- `clip_epsilon` を狭めると `clip_fraction` 自体は上がる可能性がある
- 見るべきは `clip_fraction` の絶対値だけではなく、`ratio_mean` の偏り、tail performance、best保持である

### D. `noanchor_lr5e5_clip015`

no-anchor のまま LR を半分にする。

- `training.policy_anchor.enabled = false`
- `training.policy_anchor.coef = 0.0`
- `training.lr = 0.00005`
- `training.clip_epsilon = 0.15`

狙い:

- no-anchor seed42 probe では `ratio_mean` が後半 `0.9206` まで下がった
- LR を落として、policy drift を滑らかにできるかを見る
- clip 縮小よりも直接的に update magnitude を下げる条件として置く

## 5. 実行対象一覧

### 新規 B: noanchor, seed43/44

| label | seed | overrides |
|---|---:|---|
| `stage2a_exp021_B_noanchor_lr1e4_clip015_seed43` | 43 | noanchor, lr=1e-4, clip=0.15 |
| `stage2a_exp021_B_noanchor_lr1e4_clip015_seed44` | 44 | noanchor, lr=1e-4, clip=0.15 |

### 新規 C: noanchor + clip010, seed42/43/44

| label | seed | overrides |
|---|---:|---|
| `stage2a_exp021_C_noanchor_lr1e4_clip010_seed42` | 42 | noanchor, lr=1e-4, clip=0.10 |
| `stage2a_exp021_C_noanchor_lr1e4_clip010_seed43` | 43 | noanchor, lr=1e-4, clip=0.10 |
| `stage2a_exp021_C_noanchor_lr1e4_clip010_seed44` | 44 | noanchor, lr=1e-4, clip=0.10 |

### 新規 D: noanchor + lr half, seed42/43/44

| label | seed | overrides |
|---|---:|---|
| `stage2a_exp021_D_noanchor_lr5e5_clip015_seed42` | 42 | noanchor, lr=5e-5, clip=0.15 |
| `stage2a_exp021_D_noanchor_lr5e5_clip015_seed43` | 43 | noanchor, lr=5e-5, clip=0.15 |
| `stage2a_exp021_D_noanchor_lr5e5_clip015_seed44` | 44 | noanchor, lr=5e-5, clip=0.15 |

## 6. 必須観測

各 run で以下を集計する。

### 6.1 performance

- imitation avg_rank / win_rate / deal_in_rate
- final avg_rank / win_rate / deal_in_rate
- best cycle
- best5 window
- tail5
- tail10
- cycle20-24 平均
- cycle25-29 平均

### 6.2 late drift

各 seed / condition で以下を計算する。

- `best_rank -> tail5_rank`
- `best5_rank -> tail5_rank`
- `cycle20-24 -> cycle25-29`
- final - imitation
- tail5 - imitation

特に `anchor075` と比較して、no-anchor 系で late drift がどれだけ減るかを見る。

### 6.3 PPO diagnostics

- `ratio_mean` avg / last / min / max
- `ratio_std` avg / last / max
- `clip_fraction` avg / last / max
- `advantage_std`
- `policy_loss`
- `value_loss`
- `entropy`
- `anchor_kl_discard` は A のみ参考値

no-anchor 系の重点:

- `ratio_mean` が 0.95 未満に沈み続けるか
- `clip_fraction` が 0.25 以上に張り付くか
- C/D でその傾向が改善するか

## 7. 判定基準

### B noanchor が有望と言える条件

- 3 seed 平均で tail5 が anchor075 より良い
- 3 seed 平均で tail10 が anchor075 より良い
- best は同等以上
- final が imitation より大きく悪化しない
- `ratio_mean` / `clip_fraction` が seed42 と同程度以下で、崩壊しない

### C clip010 が有望と言える条件

- B より tail5 / tail10 が同等以上
- B より `ratio_mean` の沈み込みが弱い
- B より final が安定
- best 到達力を大きく失わない

### D lr5e5 が有望と言える条件

- B より `ratio_mean` が 1.0 近くに戻る
- B より `clip_fraction` が下がる
- B と同等以上の tail5 / tail10
- best 到達が遅くなっても、30 cycle 内で十分な性能に届く

## 8. 期待される解釈パターン

### Pattern 1: B が良く、C/D は不要

no-anchor が本命。teacher anchor が late drift の主因だった可能性が高い。

次は no-anchor を practical baseline 候補にする。

### Pattern 2: B は良いが diagnostics が悪く、D が安定

teacher anchor は外した方がよいが、update strength を下げる必要がある。

次の baseline 候補は `noanchor_lr5e5_clip015`。

### Pattern 3: B は良いが C がさらに安定

teacher anchor は外し、PPO trust region を狭めるのがよい。

次の baseline 候補は `noanchor_lr1e4_clip010`。

### Pattern 4: B/C/D すべて anchor075 より悪い

no-anchor seed42 は偶然か、eval noise の可能性が高い。

次は no-anchor ではなく、anchor decay / lagged anchor / LR decay を検討する。

### Pattern 5: no-anchor 系は best は強いが tail が不安定

anchor の有無だけでは不十分。

次は cycle-based schedule が本命になる。

## 9. 実行コマンド方針

driver は `scripts/local/stage2/exp_021_driver.py`。

2026-04-30 の途中停止後、driver は残り専用に上書き済み。デフォルト実行では、完了済みの A/B は再実行せず、C/D の 6 run だけを実行する。

2026-04-30 18:23 の continuation で `C_noanchor_lr1e4_clip010_seed42` が cycle eval 中に `eval_worker_9: non-zero exit code -11` で停止した。学習本体ではなく parallel eval subprocess 側の失敗と見ているが、速度優先のため C/D continuation run も `evaluation.num_workers=10` / `training.imitation_eval.num_workers=10` のまま実行する。

代わりに、driver は continue-on-error に変更した。1 run が eval worker crash などで失敗しても `run_map.json` に `failed` として記録し、次の条件へ進む。最後に failed run が 1 件以上あれば exit code は 1 にする。従来通り最初の失敗で止めたい場合は `EXP021_STOP_ON_ERROR=1` を指定する。

満たす条件:

- `EXP021_ONLY=<label>` で単発実行可能
- `EXP021_STOP_ON_ERROR=1` で従来通り最初の failure で停止可能
- 既存 reuse run は driver では実行しない
- `run_map.json` に reuse / new を明示
- 残り C/D 6 run のみを実行対象にする
- 各 run の `training.policy_anchor.enabled` / `coef` / `lr` / `clip_epsilon` を明示的に override する
- C/D continuation run は速度優先で `evaluation.num_workers=10` / `training.imitation_eval.num_workers=10` を明示的に override する


実行コマンド:

```bash
./.venv/bin/python scripts/local/stage2/exp_021_driver.py
```

1 条件だけ実行:

```bash
EXP021_ONLY=B_noanchor_lr1e4_clip015_seed43 \
  ./.venv/bin/python scripts/local/stage2/exp_021_driver.py
```

共通 command skeleton:

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
  'experiment.name="<experiment_name>"' \
  'experiment.global_seed=<seed>' \
  'training.multi_cycle.num_cycles=30' \
  'training.value_loss_coef=0.125' \
  'training.policy_anchor.enabled=false' \
  'training.policy_anchor.coef=0.0' \
  'training.lr=<lr>' \
  'training.clip_epsilon=<clip_epsilon>' \
  'evaluation.num_workers=10' \
  'training.imitation_eval.num_workers=10' \
  'model.semantic_aux.enabled=true' \
  'model.semantic_aux.policy_projection_dim=16' \
  'model.semantic_aux.tile_presence_flags_semantic_only=false' \
  'model.value_hidden_dims=[128,64]' \
  'training.semantic_aux.enabled=true' \
  'training.semantic_aux.terminal_loss_coef=0.1' \
  'training.semantic_aux.yaku_loss_coef=0.05' \
  'feature_encoder.tile_presence_flags=false' \
  'selfplay.temperature=1.0'
```

## 10. 期待アウトプット

- `experiments/Stage02_CallUnlock/exp_021/runbook.md`
- `experiments/Stage02_CallUnlock/exp_021/run_map.json`
- `experiments/Stage02_CallUnlock/exp_021/driver_logs/*.log`
- 新規 8 run の `summary.json`
- 既存 4 run と合わせた 12 run 比較表
- 後続の `experiments/Stage02_CallUnlock/exp_021/report.md`

## 11. この設計の理由

単純に no-anchor を seed43/44 だけ追加するのは安いが、seed42 probe で既に diagnostics の黄色信号が見えている。

そのため exp_021 では、no-anchor の multi-seed 確認だけでなく、**no-anchor を practical baseline にするなら必要になりそうな update-strength 調整**も同時に見る。

この設計なら、次のどれかを判断できる。

- teacher anchor は本当に late drift の原因か
- no-anchor はそのまま採用できるか
- no-anchor には LR 半減が必要か
- no-anchor には clip 縮小が必要か
- anchor 系に戻って別の schedule を考えるべきか

時間はかかるが、次の分岐を一気に減らせるデータになる。
