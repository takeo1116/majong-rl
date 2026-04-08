# Experiment Runbook: exp_010

作成日: 2026-04-01  
Stage: `Stage02_CallUnlock`

参照:
- `experiments/Stage02_CallUnlock/exp_009_bugfix/report.md`
- `experiments/Stage02_CallUnlock/exp_009_bugfix/bug_report.md`
- `experiments/Stage02_CallUnlock/exp_008/report.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

## 1. 背景

`exp_009_bugfix` では、semantic auxiliary trunk の初回比較として

- `C0_r3_control`
- `A1_semaux_default`
- `A2_semaux_light`

の 3 条件を 1 seed で比較し、方向としては `A1_semaux_default` が最も有望という結果になった。

ただし、その後の調査で、この比較には open-hand 周りの大きい交絡が混入していたことが分かった。

- rule-based teacher の discard が open-hand を closed-hand 前提の shanten / acceptance で評価していた
- evaluator baseline seat も同じ問題を持っていた
- `FlatFeatureEncoder` の `current_shanten` / `shanten_hint` / `discard_ukeire_hint` / opponent `danger_mask` も open-hand 整合が崩れていた
- その結果、imitation teacher data では `win_called` が 0 だった

これらは `exp_009_bugfix/bug_report.md` にまとめた bugfix (`CQ-0259`〜`CQ-0261`) で修正済みであり、修正後は Stage2 selfplay smoke で `win_called` が再び生成されることを確認した。

したがって次に必要なのは、**bugfix 後の teacher / evaluator / feature semantics の上で、`exp_009` の比較をそのまま再実行すること**である。

## 2. 問い

今回の runbook で答えたい問いは次の 4 つである。

1. open-hand bugfix 後でも、R3 系 mixed PPO は `C0` / `A1` / `A2` の全条件で stable に回るか
2. bugfix 後に同じ比較を取り直しても、`A1_semaux_default` は依然として本命か
3. bugfix 後の imitation teacher / selfplay shard では、本当に `win_called` support が復活しているか
4. `A1` の semantic head は、以前より open-hand を含む terminal / yaku prediction を学びやすくなっているか

## 3. 基本方針

今回の再実行では、semantic auxiliary trunk 以外の新要素を増やさない。

比較する条件は `exp_009_bugfix` と同じ 3 条件に固定する。
差分は

- semantic aux の有無
- semantic aux loss 係数

だけに限定し、open-hand bugfix は **全条件共通の基盤修正**として扱う。

また、最初は seed `42` の 1 本だけを回す。
ここで

- `win_called` support 復活
- semantic 診断の改善
- `A1` の有望性維持

が見えた条件だけを次段階で multi-seed 化する。

## 4. 比較条件

### C0: bugfix 後 R3 control

- semantic aux 無効
- bugfix 後の teacher / evaluator / feature を使った control

### A1: bugfix 後 semantic aux default

- `model.semantic_aux.enabled = true`
- `training.semantic_aux.enabled = true`
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.2`
- `training.semantic_aux.yaku_loss_coef = 0.1`

### A2: bugfix 後 semantic aux light

- semantic aux は有効
- `model.semantic_aux.policy_projection_dim = 16`
- `training.semantic_aux.terminal_loss_coef = 0.1`
- `training.semantic_aux.yaku_loss_coef = 0.05`

## 5. 共通固定条件

全条件共通:

- A `core_minimal`
- `full observation`
- mixed PPO
- `training.rule_mix.policy_ratio = 0.50`
- `training.rule_mix_learner.baseline_sample_weight = 0.25`
- `training.policy_anchor.reference = "imitation_fixed"`
- `training.policy_anchor.coef = 0.75`
- `training.lr = 1e-4`
- `training.clip_epsilon = 0.15`
- `training.max_grad_norm = 0.50`
- `training.multi_cycle.num_cycles = 20`
- `training.multi_cycle.eval_each_cycle = true`
- `training.imitation_eval.enabled = true`
- seed `42`

意図:

- `exp_008` / `exp_009_bugfix` で stable と確認できた R3 条件を土台にする
- semantic aux の効果以外の差分を増やさない
- 今回は bugfix の影響を見るため、PPO ハイパラ探索には進まない

## 6. 今回の追加観測

`exp_009_bugfix` からの再実行なので、通常の eval 指標に加えて以下を必ず確認する。

### 6.1 `win_called` support

最低限、各条件について次を確認する。

- imitation data に `win_called` が存在するか
- selfplay shard に `win_called` が存在するか
- `call` decision row の中にも `win_called` が存在するか

ここは今回の bugfix の直接確認であり、0 なら再実行の意味が薄れる。

### 6.2 semantic head diagnostics

`A1` については、少なくとも

- imitation checkpoint
- final checkpoint

の 2 点で semantic eval を取る。

見たいもの:

- terminal accuracy
- `win_menzen` / `win_called` の support と recall
- yaku micro / macro F1
- winner-only exact match

この診断は `scripts/local/stage2/semantic_head_eval.py` を使う前提とする。

### 6.3 stability

以下が R3 と同程度に健全かを見る。

- `ratio_mean`
- `clip_fraction`
- `anchor_kl_discard`
- learner loss の暴れ方
- 20 cycle 完走可否

### 6.4 性能

主に次を比較する。

- `imitation_eval`
- `cycle_00`
- `final`
- 可能なら `tail-5 average`

特に見る差分:

- `final - imitation_eval`
- `cycle_00 - imitation_eval`

## 7. 成功判定

今回の再実行の成功は、少なくとも次を満たすこととする。

1. 全条件が stable に完走する
2. imitation data / selfplay shard に `win_called` support が復活している
3. `A1` または `A2` の `final eval` が、`C0` より `imitation_eval` からの改善量で上回る
4. `A1` の semantic diagnostics で、以前の `win_called support = 0` 状態が解消している

ここでの主目的は、

- semantic aux の比較を open-hand bugfix 後の正しい teacher 上で取り直すこと
- semantic trunk が open-hand を含む terminal / yaku prediction を本当に学び始めるかを見ること

である。

## 8. 典型的な解釈

### ケース 1: `A1` が stable で、bugfix 後も最良

解釈:

- semantic auxiliary は bugfix 後も有望
- `exp_009_bugfix` の結論は大筋維持
- 次は `A1` を 3 seeds に広げる

### ケース 2: `C0` が大きく改善し、`A1` の優位が消える

解釈:

- 以前見えていた差の一部は teacher / feature bug による交絡だった可能性が高い
- semantic aux は再評価が必要
- まず bugfix 後 control の性能を基準線として固定する

### ケース 3: `A1` は性能差が小さいが semantic diagnostics は改善

解釈:

- semantic trunk 自体は以前より meaningfully 学べている
- policy 利用の仕方や reward 側はまだ別問題
- 次は summary の使い方や branch attribution を見る

### ケース 4: `win_called` は復活したが semantic head がまだ学べない

解釈:

- teacher supply は直った
- ただし semantic auxiliary の head / loss / data balance がまだ弱い
- 次は semantic head 設計の見直しを検討する

## 9. 実装・実行方針

ベース config は R3 系 mixed baseline を使い、override で semantic aux 条件を切る。

想定管理ファイル:

- `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `experiments/Stage02_CallUnlock/exp_010/run_map.json`
- `experiments/Stage02_CallUnlock/exp_010/report.md`

想定ラベル:

- `C0_r3_control_bugfix`
- `A1_semaux_default_bugfix`
- `A2_semaux_light_bugfix`

### 9.1 事前確認

実行前に少なくとも次を確認する。

- `.venv` の通常 import で `_mahjong_core` が `meld_count` 付きシグネチャを持つこと
- Stage2 selfplay smoke で `win_called` が出ること

ここが崩れていると、再実行結果の解釈が再び不安定になる。

## 10. 実行方式

想定実行は 3 条件連続 driver とする。

想定所要時間:

- 1 run あたり約 1〜2 時間
- 3 run 合計で約 3〜6 時間
- 追加で A1 semantic diagnostics に 10〜30 分程度

実行コマンド:

```bash
./.venv/bin/python scripts/local/stage2/exp_010_driver.py
```

1 条件だけ試す場合:

```bash
EXP010_ONLY=A1_semaux_default_bugfix ./.venv/bin/python scripts/local/stage2/exp_010_driver.py
```

## 11. 今回やらないこと

- branch-swap
- multi-seed
- reward shaping
- han / fu auxiliary
- non-detach summary
- partial observation
- call policy の追加設計変更

## 12. 次アクション判定

### 良い結果だった場合

- `A1` を 3 seeds に広げる
- その後 branch-swap で optional 改善が立つかを見る

### 中立だった場合

- `C0` を新しい基準線として固定し、semantic head 診断をさらに詰める
- summary 利用や loss 設計を見直す

### 悪い結果だった場合

- teacher bugfix 後は semantic aux の優位が崩れたと判断する
- まず control を基準に Optional 学習の次の論点を整理する

## 13. 作成前チェック

- [x] 既存実験との条件重複を確認し、`exp_009_bugfix` の再実行として位置づけた
- [x] 絶対パスを書いていない
- [x] `runs/` 配下を恒久参照先として書いていない
- [x] bugfix 内容と再実行理由を `bug_report.md` にまとめた
- [x] 再実行で追加確認したい項目 (`win_called`, semantic diagnostics) を明記した
