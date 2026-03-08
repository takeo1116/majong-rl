# experiments/exp_019/runbook.md

最終更新: 2026-03-08  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **PPO が imitation 初期方策を壊す主因が「更新強度」ではなく「reward / target / advantage の質」にあるかを、baseline と weak-lr の learner 診断統計比較で切り分ける**

---

## 0. この実験の位置づけ

- 直前までの結論:
  - `shanten_hint=true` と `tie_aware_best_set` は imitation 側では有効
  - ただし PPO を入れると平均では依然として悪化方向
  - `exp_018` で `epochs=2` は明確に悪化、`lr=5e-5` は learner 統計を穏やかにするが主評価は baseline を更新できなかった
- なぜ今この比較をするのか:
  - 「更新を弱めれば直る」仮説はかなり削れた
  - 次は **何の統計が悪化と一緒に動くか** を明確にする必要がある
- この実験で更新したい判断:
  - PPO 悪化の第一候補が
    - `ratio/clip` 由来の更新強度なのか
    - `advantage/return/value_error` 由来の target 品質なのか
    - reward の sparse / heavy-tail 構造なのか
  を絞り込む

## 1. この実験の問い

1. baseline と weak-lr で、`eval_before -> eval` の悪化幅と一緒に動く learner 診断統計はどれか。
2. `lr=5e-5` で `clip_fraction` / `ratio` tail が下がっても after 改善に繋がらないのは、`advantage / return / value_error` が依然として悪いからか。
3. 現在の reward / round_over reward 分布は、PPO が打牌改善を学ぶには sparse すぎるか。

## 2. 実験方針

### 2.1 比較軸
- A: baseline
  - `training.epochs=4`
  - `training.lr=0.0001`
- B: weak-lr
  - `training.epochs=4`
  - `training.lr=0.00005`

### 2.2 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- model / encoder:
  - `feature_encoder.shanten_hint='{"enabled":true}'`
  - `training.imitation_loss_mode=tie_aware_best_set`
  - `hidden_dims=[256,128]`
  - `value_heads=["round_delta"]`
- 固定 override:
  - `imitation.num_workers=10`
  - `selfplay.imitation_matches=25`
  - `training.imitation_epochs=4`
  - `selfplay.num_matches=200`
  - `selfplay.num_workers=10`
  - `selfplay.policy_ratio=1.0`
  - `selfplay.save_baseline_actions=false`
  - `evaluation.mode=rotation`
  - `evaluation.rotation_seats='[0,1,2,3]'`
  - `evaluation.num_matches=30`
  - `evaluation.num_workers=10`
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
  - imitation 方針
  - self-play データ
  - eval 条件
  - value / entropy / clip / batch_size / gamma / gae
- 何を変えてよいか:
  - `training.lr` のみ
- reuse を使う場合の理由:
  - self-play 分布ノイズを消して、learner 差だけを見たい

## 3. 実行方式

### 3.1 実行単位
- reuse + driver
- seed ごとに REF を 1 本作成し、そこから A/B を分岐

### 3.2 reuse を使う場合
- 参照 run の作り方:
  - full run（`imitation,selfplay,learner,eval`）
  - ただし主目的は `imitation,selfplay,eval_before` の固定成果物を作ること
- `--reuse-phases`:
  - `imitation,selfplay,eval_before`
- 参照元と分岐先で一致必須のキー:
  - `feature_encoder.shanten_hint`
  - `training.imitation_loss_mode`
  - `selfplay.*`
  - `evaluation.mode`
  - `evaluation.rotation_seats`
  - `evaluation.num_matches`
  - `evaluation.num_workers`
  - `model / encoder`

### 3.3 run_map
- `experiments/exp_019/run_map.json` をローカル管理
- report には最終的に
  - `seed`
  - `role`
  - `run_dir`
  - `source_run_dir`
  を転記する

## 4. 実行コマンド

```bash
# seed=42 REF
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    feature_encoder.shanten_hint='{"enabled":true}' \
    training.imitation_loss_mode=tie_aware_best_set \
    imitation.num_workers=10 \
    selfplay.imitation_matches=25 \
    training.imitation_epochs=4 \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=30 \
    evaluation.num_workers=10 \
    training.value_loss_coef=0.25 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.epochs=4 \
    training.lr=0.0001 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

```bash
# seed=42 A (baseline reuse)
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --reuse-from runs/<REF_RUN_DIR_FOR_SEED_42> \
  --reuse-phases imitation,selfplay,eval_before \
  --override \
    experiment.global_seed=42 \
    experiment.phases='["learner","eval"]' \
    feature_encoder.shanten_hint='{"enabled":true}' \
    training.imitation_loss_mode=tie_aware_best_set \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=30 \
    evaluation.num_workers=10 \
    training.value_loss_coef=0.25 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.epochs=4 \
    training.lr=0.0001 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

```bash
# seed=42 B (weak-lr reuse)
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --reuse-from runs/<REF_RUN_DIR_FOR_SEED_42> \
  --reuse-phases imitation,selfplay,eval_before \
  --override \
    experiment.global_seed=42 \
    experiment.phases='["learner","eval"]' \
    feature_encoder.shanten_hint='{"enabled":true}' \
    training.imitation_loss_mode=tie_aware_best_set \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=30 \
    evaluation.num_workers=10 \
    training.value_loss_coef=0.25 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.epochs=4 \
    training.lr=0.00005 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

備考:
- 実運用では `scripts/local/exp_019_driver.py` を使う。
- `exp_018` と同様に driver で `run_map.json` / `driver_logs/` を管理する。

## 5. 成功判定

### 5.1 共通
- `summary.json.success == true`
- 必須成果物:
  - `summary.json`
  - `notes.md`
  - `config.yaml`

### 5.2 評価成果物
- `eval_before/eval_rotation.json`
- `eval/eval_rotation.json`
- `eval/eval_diff.json`

### 5.3 追跡キー
- `summary.phase_stats.learner`
- `summary.phase_stats.learner.ppo_diag`
- `metrics/train_metrics.json`
- `summary.reuse_info`
- `summary.phase_action`

## 6. 主評価と副評価

### 6.1 主評価
- `eval_before -> eval` の delta
  - `Δavg_rank`
  - `Δavg_score`
  - `Δdeal_in_rate`
  - `Δwin_rate`

### 6.2 副評価
- after 指標:
  - `avg_rank`
  - `avg_score`
  - `win_rate`
  - `deal_in_rate`
- 補助指標:
  - `clip_fraction`
  - `ratio_mean/std/p90/p99`
  - `advantage_mean/std/p90/p99`
  - `return_mean/std/p90/p99`
  - `old_value_mean/std`
  - `value_error_mean/std/p90/p99`
  - REF self-play 統計
  - REF reward / round_over reward 分布

### 6.3 比較優先順
- `Δavg_rank -> Δavg_score -> Δdeal_in_rate -> Δwin_rate`

## 7. 集計方法

- どのファイルを正とするか:
  - run 対応: `experiments/exp_019/run_map.json`
  - 主評価: 各 run の `eval/eval_diff.json`
  - after 指標: 各 run の `eval/eval_rotation.json`
  - learner 診断: `metrics/train_metrics.json` の `ppo_diag`
  - self-play 統計: REF の `summary.json.phase_stats.selfplay`
- mean/std の単位:
  - seed 単位平均
- seed 対応の取り方:
  - REF / A / B を seed ごとに揃えて比較
- offline 集計が必要なもの:
  - REF shard の `reward`
  - REF shard の `round_over_reward`

## 8. 想定リスクと回避

- 実行失敗しやすい箇所:
  - reuse 条件のキー不一致
  - driver 側の成果物検証
- 長時間実行時の注意:
  - `eval_before + eval` が支配的
  - 5 seeds でも数時間スケール
- 交絡要因:
  - self-play を再生成すると learner 差が見えにくくなる
- 再開方針:
  - driver が止まったら、完了済み run_dir と `run_map.json` を確認して再開
- 計算時間見積もり:
  - REF 5 本 + reuse 10 本
  - おおむね `2.5〜3.5時間` を想定

## 9. レポートに必ず含める項目

- 条件一覧
- 実行対応表
- 主評価表
- 副評価表
- learner 診断統計表
- REF self-play 統計
- REF reward / round_over reward 分布
- 結論
- 次アクション

## 10. 次アクション判定

- どの結果なら採用:
  - B が主評価で A を上回り、かつ learner 診断統計の改善が after 指標にも結びつく
- どの結果なら却下:
  - B が `clip_fraction / ratio` を下げても主評価で A を更新できない
- どの結果なら追加診断:
  - A/B とも悪化し、reward / value / advantage 側の異常が目立つ
- 次に回すべき実験:
  - reward / target / value 設計を主題にした CQ or 実験
