# Experiment Report: exp_001

作成日: 2026-03-29  
参照: `experiments/Stage02_CallUnlock/exp_001/runbook.md`, `experiments/Stage02_CallUnlock/exp_001/run_map.json`

## 1. 要約

`exp_001` は当初、Stage02a の最初の baseline 比較として

- `A core_minimal`
- `B stage1style_context`
- `C stage1style_context_plus_danger`

を同一 scaffold 上で比較する実験として開始した。

ただし、実際に走らせてみると今回の主問題は feature 差ではなく、**Stage02a の long-run PPO 安定性**だった。

現時点での結論は以下。

1. **A/B/C の feature 比較結果はまだ採用しない**
2. **imitation と short smoke は正常化できた**
3. **long-run では A/B 共通で PPO が後半 cycle で不安定化した**
4. 現在は比較実験より先に、**PPO で壊れない条件を探す段階**へ切り替える

つまり `exp_001` は、現時点では「A/B/C の勝敗を出した実験」ではなく、**Stage02a baseline 比較を始める前に安定性問題が露出した実験**として扱う。

## 2. 当初の実験目的

当初の目的は、Stage02a の mainline scaffold を固定した上で、特徴量セットだけを比較することだった。

- `A core_minimal`
  - `shanten_hint`
  - `discard_ukeire_hint`
  - `current_shanten`
  - `shape_hint`
  - `turn_context`
- `B stage1style_context`
  - A + `opponent_current_shanten`
  - A + `opponent_tenpai_flag`
- `C stage1style_context_plus_danger`
  - B + `danger_mask`

共通 scaffold は概ね以下。

- Stage02a 3-branch model (`discard / optional / value`)
- `multi_chunk_imitation`
- `tie_aware_best_set` imitation
- `imitation_value_warmstart`
- `policy_anchor(reference="imitation_fixed")`
- grouped GAE PPO
- CPU parallel selfplay / eval (`worker_num_threads=1`)

## 3. 実験前に起きたこと

### 3.1 correctness / throughput の整備

long run に入る前に、Stage02a の correctness と CPU 使用率に関する修正をかなり入れた。

主なもの:

- nested `worker_*/shard_*.parquet` を learner が読めるよう修正
- Stage02a parallel worker に thread cap を導入
- selfplay / eval の buffer 再利用と `inference_mode()` 導入
- optional 単発推論 fast path
- eval no-value fast path
- `FlatFeatureEncoder` scratch buffer
- summary / diagnostics の 0.0 保持修正
- optional imitation diagnostics の batched 化
- imitation chunk timing の分離記録
- Stage02a mixed PPO を Stage1 parity に寄せる修正
  - `baseline_sample_weight`
  - advantage 全体一括正規化
  - pure-baseline mixed PPO guard

### 3.2 short smoke は通った

修正後、short smoke では以下を確認できた。

- imitation loss が `0.0000` ではなく正常値になる
- Stage02a multi-cycle が最後まで完走する
- selfplay / learner / eval が non-zero step / update で通る
- optional path も `call_count` が十分出る
- `num_workers=10` でも CPU 使用率はおおむね `50%` 前後まで改善

この時点で、少なくとも

- shard 読み込み不良
- CPU oversubscription
- optional fast path 導入による意味論破壊

のような初期不具合は、主因ではなくなったと判断した。

## 4. long-run A/B/C で実際に起きたこと

### 4.1 A `core_minimal`

- run label: `A_core_minimal` （対応は `experiments/Stage02_CallUnlock/exp_001/run_map.json` を参照）
- 結果: **失敗**

A は imitation 自体は正常に完了したが、PPO 後半 cycle で learner loss が急激に悪化した。

代表的には以下。

- cycle 18: `learner loss = 21649.9074`
- cycle 19: `2802324.3454`
- cycle 20: `162782932.3668`
- cycle 24: `8646723963970442.0000`
- cycle 29: `nan`

最終的には eval worker が

- `tile_type 0 は合法打牌ではありません`

で全落ちした。これは eval 側の単独バグというより、**学習済み model が無効 discard を出すほど壊れた**と読むのが自然。

### 4.2 B `stage1style_context`

- run label: `B_stage1style_context` （対応は `experiments/Stage02_CallUnlock/exp_001/run_map.json` を参照）
- 結果: **完走したが採用不可**

B は最後まで走ったが、後半 PPO diagnostics が不健全だった。

final summary では概ね以下。

- `policy_loss = 330481545558.8459`
- `value_loss = 248144.77919396813`
- `ratio_mean = 4.756067375312917e+17`
- `ratio_std = 1.3179196188611925e+20`
- `clip_fraction = 0.6806994483868823`
- `anchor_kl_discard = 7.706935095047325`
- `anchor_kl_optional = 0.09550217233665355`

つまり、B は「完走」はしたが、**比較結果として採用できる安定 run ではない**。

### 4.3 C `stage1style_context_plus_danger`

- run label: `C_stage1style_context_plus_danger` （対応は `experiments/Stage02_CallUnlock/exp_001/run_map.json` を参照）
- 結果: **途中停止**

C は imitation chunk 0 の時点で手動停止した。

- `chunk 0: steps=659974 loss=0.4366`

ここまでにクラッシュは出ていないが、A/B が共通 scaffold で不健全だったため、C を続けても同じ regime に入る可能性が高いと判断して止めた。

## 5. 今回分かったこと

### 5.1 問題は feature 比較より PPO scaffold 側にある

A と B が同じ方向に壊れているため、まず疑うべきは

- `core_minimal` か `context` か
- `danger_mask` が良いか悪いか

ではなく、**Stage02a PPO の共通条件**である。

今回の `exp_001` では、A/B/C の feature 差を議論する前に、まず PPO 安定化が必要だと分かった。

### 5.2 先に壊れているのは discard branch 寄り

観測上は、optional より discard 側の drift が大きい。

- A は最終的に無効 discard を出した
- B でも `anchor_kl_discard` が大きく、`anchor_kl_optional` はかなり小さい

したがって、現時点の見立てでは

- **主導して崩れているのは discard branch**
- value branch もかなり関与している
- optional branch は主犯には見えない

という整理になる。

### 5.3 Stage1 と Stage02a には実装差分があった

source review で、Stage1 の安定 run と比べて Stage02a にはいくつか重要差分が見つかった。

- mixed PPO 時の baseline weighting が Stage1 parity で入っていなかった
- advantage 正規化が Stage1 と違い、minibatch 依存になっていた
- `policy_ratio=0.0` なのに `ppo_mode="mixed"` を無警告で通していた

この3点は今回すでに修正済み。

したがって、**次の rerun は旧 long-run と同条件ではない**。

## 6. 今回入れた発散対策

今回までに、少なくとも以下を Stage02a に入れた。

### 6.1 PPO 安定化

- `baseline_sample_weight` 導入
- discard / optional 両 branch の weighted mean
- advantage の全体一括正規化
- pure-baseline mixed PPO を `ValueError` で reject
- `value_loss_type` を weighted mean でも維持

### 6.2 imitation / diagnostics / timing

- optional imitation diagnostics の batched 化
- chunk timing の分離
  - `data_generation_sec`
  - `learner_sec`
  - `diagnostics_sec`
  - `chunk_total_sec`

### 6.3 throughput / observability

- optional fast path
- eval no-value path
- thread cap
- buffer 再利用
- nested shard 読み込み対応

## 7. 現時点の判断

1. **`exp_001` の A/B/C long-run は比較結果としては不採用**
2. **A/B/C feature 比較は一旦停止**
3. 次は `A` 固定で、PPO で壊れない条件を探す
4. 安定条件が見つかってから、改めて A/B/C 比較へ戻る

要するに、`exp_001` は途中で目的が変わった。

- 当初: feature comparison
- 現在: Stage02a PPO stability search への入り口

この切り替えは妥当だと考えている。

## 8. 次アクション

次は feature 比較ではなく、`A core_minimal` 固定で PPO 条件比較を行う。

優先度が高い候補は以下。

1. `A`, `rule_mix.policy_ratio=0.25`, `baseline_sample_weight=0.5`, `ppo_mode="mixed"`
2. `A`, `rule_mix.policy_ratio=0.25`, `baseline_sample_weight=0.5`, `ppo_mode="separated"`

共通:

- `policy_anchor.reference="imitation_fixed"`
- `policy_anchor.coef=0.5`
- `num_cycles=20`

主に見たい指標:

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- `policy_loss`
- `value_loss`
- eval `avg_rank`

この2本で「mixed を安全に使えるか」「separated の方が安定か」をまず見る。

## 9. 実行結果の現時点整理

| 条件 | status | 解釈 |
|---|---|---|
| `A core_minimal` | failed | PPO 後半で発散。最終的に eval で illegal discard |
| `B stage1style_context` | completed but unusable | 完走したが PPO diagnostics が極端で比較結果として不採用 |
| `C stage1style_context_plus_danger` | interrupted | imitation 途中で手動停止。比較材料に使わない |

## 10. 実行対応表

| seed | role | run label | 参照 | 備考 |
|---|---|---|---|---|
| 42 | A | `A_core_minimal` | `experiments/Stage02_CallUnlock/exp_001/run_map.json` | cycle 29 で NaN / illegal discard |
| 42 | B | `B_stage1style_context` | `experiments/Stage02_CallUnlock/exp_001/run_map.json` | 完走したが divergence で不採用 |
| 42 | C | `C_stage1style_context_plus_danger` | `experiments/Stage02_CallUnlock/exp_001/run_map.json` | imitation chunk 0 で手動停止 |
