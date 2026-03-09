# Experiment Runbook: exp_024

最終更新: 2026-03-10  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: joint imitation と value head 専用 current shanten 特徴が、value/target 品質と PPO 後悪化を改善するかを一晩で切り分ける

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `linear_decay + scale=0.01 + mode=both` を暫定標準 reward とする
  - `exp_023` で、`improve` 群の advantage mean は負、`worsen` 群の advantage mean は正で、reward shaping だけでは learner signal 整合が回復しないと確認した
- なぜ今この比較をするのか:
  - 現在の本命仮説は「value が未学習/表現不足で、advantage が系統的に歪んでいる」
  - そのため、imitation で value を warm start し、さらに value 専用 current shanten 特徴を入れることで改善するかを見たい
- この実験で更新したい判断:
  - joint imitation は入れるべきか
  - value current_shanten は追加価値があるか

## 1. この実験の問い

1. imitation で policy と value を同時学習すると、`shanten_diag` の符号整合と PPO 後悪化は改善するか。
2. value head 専用 current shanten 特徴は、joint imitation の上に追加価値を持つか。
3. `imitation_value_warmstart.coef` は `0.1` と `0.5` のどちらが自然か。

## 2. 実験方針

### 2.1 比較軸
- A: baseline
  - `training.imitation_value_warmstart.enabled=false`
  - `model.value_features.current_shanten.enabled=false`
- B: joint imitation small
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.1`
  - `model.value_features.current_shanten.enabled=false`
- C: joint imitation medium
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.5`
  - `model.value_features.current_shanten.enabled=false`
- D: joint imitation small + value current_shanten
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.1`
  - `model.value_features.current_shanten.enabled=true`
- E: joint imitation medium + value current_shanten
  - `training.imitation_value_warmstart.enabled=true`
  - `training.imitation_value_warmstart.coef=0.5`
  - `model.value_features.current_shanten.enabled=true`

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- model / encoder:
  - `feature_encoder.shanten_hint.enabled=true`
  - `training.imitation_loss_mode=tie_aware_best_set`
- reward 条件:
  - `reward.shaping.shanten_delta.enabled=true`
  - `reward.shaping.shanten_delta.scale=0.01`
  - `reward.shaping.shanten_delta.mode=both`
  - `reward.shaping.shanten_delta.schedule.type=linear_decay`
- 固定 override:
  - `imitation.num_workers=10`
  - `selfplay.imitation_matches=25`
  - `training.imitation_epochs=4`
  - `selfplay.num_matches=200`
  - `selfplay.num_workers=10`
  - `selfplay.policy_ratio=1.0`
  - `selfplay.save_baseline_actions=false`
  - `evaluation.mode=rotation`
  - `evaluation.rotation_seats=[0,1,2,3]`
  - `evaluation.num_matches=30`
  - `evaluation.num_workers=10`
  - `training.epochs=4`
  - `training.lr=0.0001`
  - `training.value_loss_coef=0.25`
  - `training.batch_size=256`
  - `training.gamma=0.99`
  - `training.gae_lambda=0.95`
  - `training.entropy_coef=0.01`
  - `training.clip_epsilon=0.2`
  - `training.device=cuda`
  - `selfplay.inference_device=cpu`
  - `evaluation.inference_device=cpu`

### 2.3 交絡回避
- 何を固定するか:
  - reward 条件、policy 側特徴、PPO learner ハイパラ、evaluation 条件、seeds
- 何を変えてよいか:
  - `imitation_value_warmstart.enabled`
  - `imitation_value_warmstart.coef`
  - `model.value_features.current_shanten.enabled`
- reuse を使わない理由:
  - imitation checkpoint 自体が条件差で変わる
  - selfplay 分布と learner 診断も条件差で変わる
  - 今回は全条件 full run で取る

## 3. 実行方式

### 3.1 実行単位
- batch
- driver

### 3.2 既存実験からの流用
- 参照可能な既存 run:
  - なし（条件自体は近いが、新規 joint imitation / current shanten 実装後の成果物が必要）
- 流用するもの:
  - なし
- 新規実行するもの:
  - A, B, C, D, E の 5 条件すべて
- 実データ確認:
  - `runs/` 配下の参照 run が残っているか: 問わない
  - 残っていない場合、必要な値が `report.md` に転記されているか: 問わない
- 再実行が必要な理由:
  - 新実装後の `summary/batch_summary/shanten_diag/imitation value_loss` を取り直す必要がある

### 3.3 run_map
- `experiments/exp_024/run_map.json` を driver の正とする
- report には条件名と batch_dir の対応表を転記する

## 4. 実行コマンド

driver を正とする。

```bash
python3 scripts/local/exp_024_driver.py
```

参考: CLI override は deep dotted 記法を使う。

## 5. 成功判定

### 5.1 共通
- `summary.json.success == true`
- 必須成果物:
  - `summary.json`
  - `config.yaml`
  - `metrics/train_metrics.json`

### 5.2 評価成果物
- `eval/eval_rotation.json`
- `eval/eval_diff.json`
- `batch_summary.json`

### 5.3 追跡キー
- `summary.phase_stats.imitation.value_loss`
- `summary.phase_stats.imitation.imitation_value_warmstart`
- `summary.model_features.value_features.current_shanten.enabled`
- `summary.phase_stats.learner.ppo_diag.shanten_diag`
- `summary.phase_stats.learner.ppo_diag.value_error_*`
- `summary.phase_stats.learner.ppo_diag.clip_fraction`

## 6. 主評価と副評価

### 6.1 主評価
- `shanten_diag` の符号整合
  - `improve.advantage.mean` がより正方向へ寄るか
  - `worsen.advantage.mean` がより負方向へ寄るか
  - `improve/worsen` の `value_error` が改善するか
- `eval_before -> eval` の悪化幅

### 6.2 副評価
- imitation phase:
  - `value_loss`
  - `teacher_top1_match_rate`
  - `teacher_best_set_hit_rate`
- after 指標:
  - `avg_rank`
  - `avg_score`
  - `win_rate`
  - `deal_in_rate`
- learner 補助指標:
  - `ratio_*`
  - `clip_fraction`

### 6.3 比較優先順
- `shanten_diag` 整合
- `Δavg_rank`
- `Δavg_score`
- `Δdeal_in_rate`
- `Δwin_rate`

## 7. 集計方法

- どのファイルを正とするか:
  - `runs/<batch_dir>/batch_summary.json`
  - 必要に応じて各 run の `summary.json`
- mean/std の単位:
  - seed=5 の run mean/std
- seed 対応の取り方:
  - batch `runs[*]` を条件内で集約
- offline 集計が必要なもの:
  - 原則なし
  - report 作成時に `shanten_diag` と imitation 指標を batch から再集約する

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - `current_shanten.enabled=true` 条件の runner/eval 経路
  - deep dotted override の typo
- 長時間実行時の注意:
  - 条件数が 5 本なので夜間実行前提
- 交絡要因:
  - reward 条件は完全固定
  - PPO learner 条件も完全固定
- 再開方針:
  - driver は resume 未対応
  - 中断時は条件単位でどこまで完了したかを log / run_map で確認する
- 計算時間見積もり:
  - 約 5.5〜7 時間

## 9. レポートに必ず含める項目

- 条件一覧
- 実行対応表
- `shanten_diag` 主診断表
- imitation 指標表（`value_loss`, teacher 指標）
- `eval_before -> eval` 表
- after 指標表
- 結論
- 次アクション

## 10. 次アクション判定

- どの結果なら採用:
  - `shanten_diag` が改善し、`eval_before -> eval` も悪化縮小
  - その条件を暫定標準として次段へ進める
- どの結果なら却下:
  - imitation `value_loss` だけ増えて、teacher 指標と `eval_diff` が悪化
- どの結果なら追加診断:
  - joint imitation は効くが current shanten が効かない
  - current shanten は効くが coef 感度が不安定
- 次に回すべき実験:
  - 良かった条件を固定し、残る `shanten_diag` の歪みをさらに value/target 観点で診断する

## 11. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] 参照する既存 run の実データが残っているか、または必要値が `report.md` に転記済みかを確認した
- [x] 再実行する条件について、流用しない理由を明記した
