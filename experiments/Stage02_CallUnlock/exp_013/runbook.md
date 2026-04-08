# Experiment Runbook: exp_013

作成日: 2026-04-03  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_011/report.md`
- `experiments/Stage02_CallUnlock/exp_012/report.md`
- `experiments/Stage02_CallUnlock/exp_011/runbook.md`
- `experiments/Stage02_CallUnlock/exp_012/runbook.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

## 1. 背景

`exp_011` では、feature 修正後の条件で semantic auxiliary を再評価し、

- actor-relative full observation
- full path `riichi` / `menzen`

の追加により `A1/A2` が `C0` より改善することを確認した。

一方で `exp_012` では、`A1` を固定して `training.value_loss_coef` を sweep し、

- `value_loss_coef = 0.125` (`vhalf`) が final performance の新基準として有望
- ただし semantic collapse の主因は `value_loss_coef` だけではなさそう

という結果を得た。

その後、semantic learning を素直にするためにモデル構造と terminal target をさらに整理した。

## 2. 今回の再実験前に入った修正

今回の `exp_013` は、`exp_012` の `A1_vhalf_v0125` を基準条件としつつ、以下の修正を反映した上で `C0 / A1 / A2` を再比較する。

### 2.1 Stage2a direct hint branch (`CQ-0265`)

`shanten_hint` / `discard_ukeire_hint` を Stage2a の shared trunk 入力から外し、Stage01 と同じ思想の discard direct hint branch に戻した。

具体的には:

- `discard` path のみが tile-wise local scorer + context gate で利用
- `optional / value / semantic` には流さない
- Stage2a ではこの 2 hint を mandatory とする

これにより、discard 専用の局所ヒントが semantic/value 側を汚染しない構造になった。

### 2.2 terminal semantic の 5-class 化 (`CQ-0266`)

terminal semantic label を次の 5 class に再定義した。

- `win_menzen`
- `win_called`
- `draw_tenpai`
- `deal_in`
- `other_non_dealin`

従来の 8-class にあった、policy にとって意味の薄い細分化を整理し、

- 自分が面前で和了できるか
- 自分が副露して和了できるか
- 形式聴牌で終われるか
- 放銃するか
- それ以外か

という、行動に直結しやすい終局差だけを残した。

この変更に合わせて、

- outcome → terminal label mapping
- selfplay / imitation teacher label 生成
- terminal head 出力次元
- semantic eval / confidence diagnostics

もすべて更新済みである。

## 3. 今回の問い

今回の runbook で答えたい問いは次の 5 つである。

1. `CQ-0265/0266` 後でも `C0 / A1 / A2` は stable に完走するか
2. `A1_semaux_default` は、新 terminal 5-class と direct hint 切り離し後に再び `C0` を明確に上回るか
3. `win_called` の relevant subset で、以前より `p(win_called)` が上がるか
4. open + tenpai + actual `win_called` subset で、以前不自然に高かった `p(win_menzen)` が下がるか
5. それでも semantic が改善しないなら、次の本命は loss / sampling / anchor 側だと言えるか

## 4. 実験方針

今回は **`exp_012` の `vhalf` を新しい共通基準** とし、`C0 / A1 / A2` を比較する。

つまり、差分は次の 2 系統に限定される。

- `semantic_aux` の有無と強さ
- `CQ-0265/0266` による最新の構造整理が効いた状態

`value_loss_coef` は全条件で固定し、再度 semantic aux の有無を比較する。

## 5. 比較条件

### C0: control + vhalf + structfix

- semantic aux 無効
- `training.value_loss_coef = 0.125`
- 最新の feature / direct hint / terminal 5-class を使う control

### A1: semantic aux default + vhalf + structfix

- semantic aux 有効
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.2`
- `training.semantic_aux.yaku_loss_coef = 0.1`
- `training.value_loss_coef = 0.125`

### A2: semantic aux light + vhalf + structfix

- semantic aux 有効
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`
- `training.value_loss_coef = 0.125`

## 6. 共通固定条件

全条件共通:

- `core_minimal`
- `full observation`
- mixed PPO
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.policy_anchor.coef = 0.75`
- `training.value_loss_coef = 0.125`
- `training.lr = 1e-4`
- `training.clip_epsilon = 0.15`
- `training.max_grad_norm = 0.50`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`
- seed `42`

encoder / model 側では、最新の構造を前提にする。

- actor-relative full observation
- full path `riichi`
- full path `menzen`
- Stage2a discard direct hints (`shanten_hint`, `discard_ukeire_hint`)
- terminal semantic 5-class

## 7. 必須観測

### 7.1 通常性能

- `imitation_eval`
- `final`
- `best cycle`
- `tail-5 average`

特に見る差分:

- `final - imitation_eval`
- `A1 - C0`
- `A2 - C0`
- `A1 - A2`

### 7.2 PPO 安定性

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- learner loss の暴れ方
- 20 cycle 完走可否

### 7.3 semantic diagnostics

少なくとも `A1` について、以下を取る。

- `checkpoint_imitation.pt`
- `checkpoint_cycle_05.pt`
- `checkpoint_learner.pt`

見るもの:

- terminal accuracy
- `win_called` / `draw_tenpai` / `deal_in` の support と recall
- terminal label-conditioned confidence
- `actual_win_called_and_tenpai_and_open` subset の平均予測分布
- `p(win_called)`
- `p(win_menzen)`
- `all_samples` 平均との差

### 7.4 成功判定に直結する subset 診断

前回の `exp_012` では、`actual_win_called_and_tenpai_and_open` でも

- `p(win_called)` がほぼ全体平均と変わらない
- `p(win_menzen)` が open 状態にもかかわらずそれなりに残る

という問題があった。

今回は少なくとも、次が改善しているかを確認する。

- `actual_win_called_and_tenpai_and_open` で `p(win_called)` が前回の `~0.12` より明確に上がる
- 同 subset で `p(win_menzen)` が前回の `~0.07` より明確に下がる
- subset と `all_samples` / `actual_ron_bystander` 全体平均との差が開く

## 8. 成功判定

今回の再実験は、少なくとも次を満たせば成功とみなす。

1. 全条件が stable に完走する
2. `A1` または `A2` が `C0` を final `avg_rank` で上回る
3. `A1` の relevant subset で `win_called` 識別が前回より改善する
4. terminal 5-class 化により、open 状態で不可能な class への確率配分が減る

## 9. 典型的な解釈

### ケース 1: `A1` が再び明確に強く、subset 診断も改善

解釈:

- direct hint の切り離しと terminal 5-class 化が効いた
- 次は multi-seed か、A1 を基準にさらなる安定化を検討する

### ケース 2: policy は改善するが semantic subset はまだ弱い

解釈:

- semantic trunk の補助効果はある
- ただし terminal head の保持 / sampling / loss 設計はまだ課題
- 次は anchor / imbalance / sampling を検討する

### ケース 3: ほとんど改善しない

解釈:

- feature / target 整理だけでは足りない
- semantic learning の主因は学習設計側にある
- 次は loss / replay / sampling の根本対策へ進む

## 10. 実行コマンド

全条件:

```bash
./.venv/bin/python scripts/local/stage2/exp_013_driver.py
```

1 条件だけ:

```bash
EXP013_ONLY=A1_semaux_default_vhalf_structfix ./.venv/bin/python scripts/local/stage2/exp_013_driver.py
```
