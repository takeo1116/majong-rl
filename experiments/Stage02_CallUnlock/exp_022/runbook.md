# Experiment Runbook: exp_022

作成日: 2026-05-01  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_021/report.md`
- `experiments/Stage02_CallUnlock/exp_021/run_map.json`
- `experiments/Stage02_CallUnlock/exp_020/report.md`
- `docs/CHANGE_QUEUE.md`

## 1. 背景

`exp_021` では、現行 fixedRL 条件で `teacher anchor` を外した no-anchor 系を比較した。

3 seed 平均では、`B: no-anchor, lr=1e-4, clip=0.15` が最も良かった。

| Group | imitation | final | best | best5 | tail5 | tail10 |
|---|---:|---:|---:|---:|---:|---:|
| A anchor075 lr1e-4 clip0.15 | 2.352 | 2.360 | 2.188 | 2.311 | 2.392 | 2.355 |
| B no-anchor lr1e-4 clip0.15 | 2.352 | 2.327 | 2.148 | 2.246 | 2.288 | 2.303 |
| C no-anchor lr1e-4 clip0.10 | 2.352 | 2.355 | 2.165 | 2.257 | 2.313 | 2.303 |
| D no-anchor lr5e-5 clip0.15 | 2.330 | 2.363 | 2.190 | 2.253 | 2.361 | 2.338 |

`B` は best / best5 / tail5 / tail10 で `A` を上回り、tail 平均でも imitation より改善していた。

一方で、`B` には以下の未解決点がある。

- best 一点から tail5 への戻りは残っている
- no-anchor は `clip_fraction` が高い
- no-anchor は entropy が大きく低下する
- 30 cycle では、今見えている best が天井なのか、まだ伸びる途中なのか判断できない

`exp_022` では、条件を増やさず、`B` をそのまま 60 cycle まで延長して長期挙動を見る。

## 2. 今回の問い

`exp_022` で答えたい問いは以下。

1. `B` の 30 cycle 時点の best 付近が現設定の天井なのか
2. 60 cycle まで回すとさらに改善するのか
3. 長期化すると entropy 低下 / clip 高止まりで崩れるのか
4. tail が安定して imitation より良い状態を保てるのか
5. best 一点ではなく best5 / best10 / tail10 で見ても改善が続くのか

ここでは特徴量追加や entropy bonus はまだ入れない。まず no-anchor baseline 自体の長期曲線を確認する。

## 3. 実験方針

`exp_021` の B 条件を 3 seed で fresh に 60 cycle 実行する。

既存 `exp_021` B run の continuation は使わない。

理由:

- 現在の runner は任意 cycle から安全に resume する前提では設計していない
- fresh 60 cycle の方が run_map / summary / cycle 統計が単純
- 30 cycle との比較は `exp_021` の B を historical reference として行う

## 4. 条件定義

全条件共通:

- config: `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `training.multi_cycle.num_cycles = 60`
- `training.multi_cycle.selfplay_matches_per_cycle = 200`
- `training.policy_anchor.enabled = false`
- `training.policy_anchor.coef = 0.0`
- `training.lr = 0.0001`
- `training.clip_epsilon = 0.15`
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
- expected shard semantics: `sample_semantics_version = 3`

## 5. 実行対象

| label | seed | 目的 |
|---|---:|---|
| `stage2a_exp022_Blong_noanchor_lr1e4_clip015_seed42` | 42 | exp_021 B seed42 の長期再確認 |
| `stage2a_exp022_Blong_noanchor_lr1e4_clip015_seed43` | 43 | exp_021 B seed43 の長期再確認 |
| `stage2a_exp022_Blong_noanchor_lr1e4_clip015_seed44` | 44 | exp_021 B seed44 の長期再確認 |

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

cycle window は以下を必ず出す。

| Window | 意味 |
|---|---|
| c00-c09 | early |
| c10-c19 | middle-1 |
| c20-c29 | exp_021 と直接比較できる late-30 |
| c30-c39 | extension-1 |
| c40-c49 | extension-2 |
| c50-c59 | final-tail |

### 6.2 天井判定

以下を見る。

- `best cycle` が c30 以降に移動するか
- `best5 / best10` が exp_021 B を上回るか
- `c30-c59` 平均が `c00-c29` 平均を上回るか
- `tail10` が exp_021 B の tail10 より良いか

判定目安:

| 観測 | 解釈 |
|---|---|
| best が c30 以降に更新され、tail10 も改善 | 30 cycle 時点は天井ではない |
| best は更新されるが tail は悪い | 伸びるが保持できない |
| best / best5 / tail が c20-c30 付近で頭打ち | 現設定の天井に近い |
| c40 以降で明確に悪化 | entropy 低下 / over-update / policy collapse を疑う |

### 6.3 Drift

各 seed で以下を計算する。

- `best -> tail5`
- `best5 -> tail5`
- `best10 -> tail10`
- `c20-c29 -> c30-c39`
- `c30-c39 -> c40-c49`
- `c40-c49 -> c50-c59`

特に `best5 -> tail5` と `best10 -> tail10` を重視する。best 一点は eval noise を強く受けるため、window 指標で判断する。

### 6.4 PPO diagnostics

各 run / window で以下を集計する。

- `ratio_mean` avg / last / min / max
- `ratio_std` avg / last / max
- `clip_fraction` avg / last / max
- `advantage_mean`
- `advantage_std`
- `entropy` avg / last / min

見るポイント:

- `ratio_mean` がさらに 1 から下がるか
- `clip_fraction` が 0.25 以上で高止まりするか
- entropy が底打ちするか、それとも下がり続けるか
- entropy 低下と avg_rank 悪化が同期するか

### 6.5 Win / Deal-in

avg_rank だけでなく、以下も見る。

- win_rate
- deal_in_rate
- win_rate - deal_in_rate
- tail window で win_rate が落ちていないか
- tail window で deal_in_rate が上がっていないか

## 7. 参照 baseline

`exp_021` B の 3 seed 平均を参照値とする。

| 指標 | exp_021 B |
|---|---:|
| imitation | 2.352 |
| final | 2.327 |
| best | 2.148 |
| best5 | 2.246 |
| tail5 | 2.288 |
| tail10 | 2.303 |
| ratio_avg | 0.967 |
| ratio_last | 0.947 |
| clip_avg | 0.208 |
| clip_last | 0.238 |
| entropy_avg | 0.370 |
| entropy_last | 0.233 |

## 8. 期待される分岐

### Pattern A: 60 cycle でさらに改善し、tail も良い

例:

- best < 2.10
- best5 < 2.20
- tail10 < 2.25
- c50-c59 が c20-c29 より良い

解釈:

- no-anchor B はまだ天井ではない
- 次はより長い run または entropy / lr schedule の軽い安定化
- B を Stage02 practical baseline に昇格してよい

### Pattern B: best は更新するが tail が悪い

例:

- best < 2.10
- tail10 > 2.30
- entropy が下がり続ける
- clip_fraction が高止まり

解釈:

- 改善余地はあるが保持に失敗している
- 次は entropy bonus / lr decay / lagged anchor / anchor decay

### Pattern C: c20-c30 付近で頭打ち

例:

- best / best5 が exp_021 B と同程度
- c30 以降で改善しない
- tail は大崩れしない

解釈:

- 現モデル・現特徴・現報酬での近い天井かもしれない
- 次は feature / model capacity / reward / value 側を見る

### Pattern D: 長期化で崩れる

例:

- c40 以降で avg_rank が悪化
- entropy が極端に低下
- win_rate が下がるか deal_in_rate が上がる

解釈:

- no-anchor は短期改善には有効だが長期安定性が足りない
- anchor decay / lagged anchor / entropy bonus の優先度が高い

## 9. 実行コマンド方針

別途 driver を作る。

期待する driver:

- `scripts/local/stage2/exp_022_driver.py`
- `experiments/Stage02_CallUnlock/exp_022/run_map.json` を生成/更新
- `experiments/Stage02_CallUnlock/exp_022/driver_logs/` にログ保存
- `EXP022_ONLY=<label>` で単発実行可能
- 1 run failure で全体を止めない continue-on-error 形式
- `EXP022_STOP_ON_ERROR=1` で従来通り即停止可能

共通 command skeleton:

```bash
./.venv/bin/python -m mahjong_rl.cli \
  --config configs/stage2a_core_minimal_mixed_s1_baseline.yaml \
  --base-dir runs \
  --override \
  'experiment.name="<experiment_name>"' \
  'experiment.global_seed=<seed>' \
  'training.multi_cycle.num_cycles=60' \
  'training.multi_cycle.selfplay_matches_per_cycle=200' \
  'training.policy_anchor.enabled=false' \
  'training.policy_anchor.coef=0.0' \
  'training.lr=0.0001' \
  'training.clip_epsilon=0.15' \
  'training.value_loss_coef=0.125' \
  'model.value_hidden_dims=[128,64]' \
  'model.semantic_aux.enabled=true' \
  'model.semantic_aux.policy_projection_dim=16' \
  'model.semantic_aux.tile_presence_flags_semantic_only=false' \
  'training.semantic_aux.enabled=true' \
  'training.semantic_aux.terminal_loss_coef=0.1' \
  'training.semantic_aux.yaku_loss_coef=0.05' \
  'feature_encoder.tile_presence_flags=false' \
  'selfplay.temperature=1.0'
```

## 10. 事前注意

`CQ-0280` は未実装である。

そのため cycle eval worker crash が再発した場合、その run は失敗する可能性がある。driver は batch-level continue-on-error にするが、run 内で eval だけ retry する機構はまだない。

発症率は低そうなので、今回の exp_022 はこのまま実行してよい。ただし crash が複数回出た場合は、先に `CQ-0280` を実装する。
