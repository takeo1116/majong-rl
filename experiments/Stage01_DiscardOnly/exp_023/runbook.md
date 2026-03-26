# Experiment Runbook: exp_023

最終更新: 2026-03-09  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: reward shaping が learner の更新信号まで届いているかを、`shanten_diag` を使って直接診断する

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `linear_decay + scale=0.01 + mode=both` が現時点の暫定標準 reward 条件
  - sparse reward は PPO 悪化の主因の一つだった
  - それでも PPO 後悪化は完全には消えておらず、value / target 側が次の本命候補
- なぜ今この比較をするのか:
  - 今回から `ppo_diag.shanten_diag` が使えるので、改善打牌/悪化打牌ごとの `advantage/return/value_error` を直接見られる
  - まずは「reward shaping が learner の更新信号まで届いているか」を確認したい
- この実験で更新したい判断:
  - reward shaping が improve/worsen 群の advantage を望ましい方向へ動かしているか
  - もし動いていないなら、value/target 側の問題をかなり強く疑ってよいか

## 1. この実験の問い

1. baseline reward と比較して、標準 shaping reward は `shanten_improve` 群の advantage をより正寄りにできているか
2. baseline reward と比較して、標準 shaping reward は `shanten_worsen` 群の advantage をより負寄りにできているか
3. shaping を入れても improve/worsen 群の advantage 整合が弱い場合、`value_error` がその原因候補として見えるか

## 2. 実験方針

### 2.1 比較軸
- A: baseline reward
  - `point_delta` のみ
  - ただし `shanten_diag` 取得のため `shanten_delta.enabled=true, scale=0.0` を入れる
- B: 標準 shaping reward
  - `linear_decay + scale=0.01 + mode=both`

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- model / encoder:
  - `FlatFeatureEncoder`
  - `MLPPolicyValueModel`
  - `feature_encoder.shanten_hint={"enabled":true}`
- 固定 override:
  - `training.imitation_loss_mode=tie_aware_best_set`
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
  - imitation 条件
  - learner ハイパラ
  - evaluation 条件
  - encoder / model
- 何を変えてよいか:
  - reward 条件のみ
- reuse を使わない理由:
  - reward 条件が self-play / learner target の両方を変えるため、self-play から丸ごと変わる

## 3. 実行方式

### 3.1 実行単位
- batch + driver

### 3.2 既存実験からの流用
- 参照可能な既存 run:
  - `exp_021` A（baseline reward）
  - `exp_022` B（標準 shaping reward）
- 流用するもの:
  - なし
- 新規実行するもの:
  - A, B を両方新規実行
- 実データ確認:
  - `runs/` 配下の参照 run が残っているか:
    - 現時点では残っている
  - 残っていない場合、必要な値が `report.md` に転記されているか:
    - 主評価値は転記済み
    - ただし今回は新規の `shanten_diag` を見たいので流用不可
- 再実行が必要な理由:
  - `CQ-0146/0147` 実装後の `shanten_diag` を新しい成果物として取得したい
  - 旧 run には今回欲しい診断統計が存在しない

### 3.3 run_map
- `experiments/exp_023/run_map.json` に条件A/Bと batch_dir を保存
- report には batch_dir と条件対応表を転記する

## 4. 実行コマンド

```bash
# 条件A: baseline reward（観測用 zero-scale shaping）
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    'feature_encoder.shanten_hint={"enabled":true}' \
    training.imitation_loss_mode=tie_aware_best_set \
    imitation.num_workers=10 \
    selfplay.imitation_matches=25 \
    training.imitation_epochs=4 \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    evaluation.mode=rotation \
    evaluation.rotation_seats=[0,1,2,3] \
    evaluation.num_matches=30 \
    evaluation.num_workers=10 \
    training.epochs=4 \
    training.lr=0.0001 \
    training.value_loss_coef=0.25 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu \
    reward.shaping.shanten_delta.enabled=true \
    reward.shaping.shanten_delta.scale=0.0 \
    reward.shaping.shanten_delta.mode=both \
    reward.shaping.shanten_delta.schedule.type=linear_decay
```

```bash
# 条件B: 標準 shaping reward
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    'feature_encoder.shanten_hint={"enabled":true}' \
    training.imitation_loss_mode=tie_aware_best_set \
    imitation.num_workers=10 \
    selfplay.imitation_matches=25 \
    training.imitation_epochs=4 \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    evaluation.mode=rotation \
    evaluation.rotation_seats=[0,1,2,3] \
    evaluation.num_matches=30 \
    evaluation.num_workers=10 \
    training.epochs=4 \
    training.lr=0.0001 \
    training.value_loss_coef=0.25 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu \
    reward.shaping.shanten_delta.enabled=true \
    reward.shaping.shanten_delta.scale=0.01 \
    reward.shaping.shanten_delta.mode=both \
    reward.shaping.shanten_delta.schedule.type=linear_decay
```

## 5. 成功判定

### 5.1 共通
- `summary.json.success == true`
- 必須成果物:
  - `summary.json`
  - `notes.md`
  - `config.yaml`
  - `metrics/train_metrics.json`

### 5.2 評価成果物
- `eval_before/eval_rotation.json`
- `eval/eval_rotation.json`
- `eval/eval_diff.json`
- `batch_summary.json`
- `batch_table.csv`

### 5.3 追跡キー
- `summary.phase_stats.learner.ppo_diag.shanten_diag`
- `metrics/train_metrics.json -> ppo_diag.shanten_diag`
- `batch_summary.json.runs[*].learner_diag.shanten_diag`
- `summary.phase_stats.selfplay.reward_composition`
- `summary.phase_stats.selfplay.reward_shaping`（条件Bのみ）

## 6. 主評価と副評価

### 6.1 主評価
- 今回の主評価は通常の `eval_before -> eval` ではなく、まず `shanten_diag` の整合を見る
- 特に最優先で見る項目:
  - `improve.advantage.mean`
  - `improve.advantage.positive_ratio`
  - `worsen.advantage.mean`
  - `worsen.advantage.negative_ratio`
  - `improve.value_error.mean`
  - `worsen.value_error.mean`

### 6.2 副評価
- 通常の主指標:
  - `Δavg_rank`
  - `Δavg_score`
  - `Δdeal_in_rate`
  - `Δwin_rate`
- after 指標
- `clip_fraction`, `ratio_mean/std`, `value_error_mean/std`
- reward 内訳

### 6.3 比較優先順
- `improve/worsen advantage 整合 -> improve/worsen value_error -> Δavg_rank -> Δavg_score -> Δdeal_in_rate -> Δwin_rate`

## 7. 集計方法

- どのファイルを正とするか:
  - `batch_summary.json`
  - 各 run の `summary.json`
  - 各 run の `metrics/train_metrics.json`
- mean/std の単位:
  - seed 単位
- seed 対応の取り方:
  - A/B の batch_summary から seed ごとに比較
- offline 集計が必要なもの:
  - なし

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - reward shaping override
- 長時間実行時の注意:
  - 2 条件 full batch のため数時間かかる
- 交絡要因:
  - reward 条件以外を固定済み
- 再開方針:
  - driver 再実行
- 計算時間見積もり:
  - 約 2.5〜3.5 時間

## 9. レポートに必ず含める項目

- 条件一覧
- 実行対応表
- `shanten_diag` 比較表
  - improve / same / worsen の `advantage/return/value_error`
- 通常主評価表
- reward 内訳表
- learner 診断表
- 結論
- 次アクション

## 10. 次アクション判定

- どの結果なら採用:
  - B で improve 群がより正寄り、worsen 群がより負寄りになり、通常指標も改善方向なら
- どの結果なら却下:
  - B で `shanten_diag` の整合が改善せず、通常指標も改善しない場合
- どの結果なら追加診断:
  - shaping により通常指標は改善するが `value_error` が大きく残る場合
- 次に回すべき実験:
  - value 入力改善仮説の検証
  - 例: value に現在シャンテン数を直接入れる案の CQ / 実験

## 11. 作成前チェック

- [x] 既存実験との条件重複を確認し、流用可否を判断した
- [x] 参照する既存 run の実データが残っているか、または必要値が `report.md` に転記済みかを確認した
- [x] 再実行する条件について、流用しない理由を明記した
