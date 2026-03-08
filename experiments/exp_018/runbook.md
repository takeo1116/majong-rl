# experiments/exp_018/runbook.md（Runbook 18）

最終更新: 2026-03-08  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **PPO が imitation 初期方策を壊す理由を、self-play 分布・reward 分布・learner 診断統計・`eval_before -> eval` の変化をまとめて比較し、主因が「更新強度」なのか「更新方向 / target の質」なのかを切り分ける**

---

## 0. この実験の位置づけ

ここまでで分かったことは次の通り。

- `shanten_hint` は教師再現に効いている
- `tie_aware_best_set` は imitation-only では有望
- しかし warm start + PPO では、strict / tie-aware の差は決定的にならず、平均では依然として悪化方向
- よって現時点の主問題は **PPO 側** にある可能性が高い

さらに現状確認と追加実装により、次が見られるようになった。

- self-play 統計（run 単位）
- shard からの reward / round_over reward のオフライン集計
- `eval_before -> eval` の run 単位差分
- learner 診断統計
  - `advantage_*`
  - `return_*`
  - `old_value_*`
  - `value_error_*`
  - `ratio_*`
  - `clip_fraction`

したがって次は、**tie-aware を固定**し、PPO 更新強度だけを変えて、
- どの条件で壊れ方が弱まるか
- learner 内部統計がどう変わるか
を比較する。

---

## 1. この実験の問い

今回知りたいのは次の3点。

### 1.1 更新強度を弱めると、`eval_before -> eval` の悪化は小さくなるか
比較条件:
- baseline: `epochs=4, lr=1e-4`
- weak-epochs: `epochs=2, lr=1e-4`
- weak-lr: `epochs=4, lr=5e-5`

### 1.2 更新強度を弱めたとき、learner 診断統計はどう変わるか
特に見たいのは:
- advantage のスケール
- return と old_value のズレ
- value error の大きさ
- ratio 分布
- `clip_fraction`

### 1.3 reward / self-play 分布に極端な偏りがないか
- self-play データが PPO に不向きではないか
- reward が sparse / heavy-tail すぎないか
- `round_over` 報酬が過度に支配的ではないか

---

## 2. 実験方針

### 2.1 tie-aware を固定する理由
exp_017 で、
- imitation-only では tie-aware が有望
- しかし PPO で優位が消える
ということが見えた。

したがって今回は、
**imitation 方針差をもう増やさず、PPO 側の違いだけに集中する**。

固定:
- `feature_encoder.shanten_hint='{"enabled":true}'`
- `training.imitation_loss_mode=tie_aware_best_set`

### 2.2 selfplay 再利用を使う理由
今回は learner 差だけを見たいので、self-play 分布ノイズをなるべく減らす。

そのため、
- seed ごとに 1 本だけ参照 run を作る
- そこから `imitation,selfplay,eval_before` を再利用
- learner/eval だけ 3 条件で分岐
とする。

### 2.3 実験の全体構成
seed ごとに:
1. 参照 run（REF）を 1 本作る
2. そこから 3 条件に分岐
   - A: baseline
   - B: weak-epochs
   - C: weak-lr

---

## 3. 比較条件

### 3.1 共通固定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- `feature_encoder.shanten_hint='{"enabled":true}'`
- `training.imitation_loss_mode=tie_aware_best_set`
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

### 3.2 learner 比較条件
- A baseline:
  - `training.epochs=4`
  - `training.lr=0.0001`
- B weak-epochs:
  - `training.epochs=2`
  - `training.lr=0.0001`
- C weak-lr:
  - `training.epochs=4`
  - `training.lr=0.00005`

### 3.3 なぜ `evaluation.num_matches=30` か
今回はスクリーニング寄りの診断 runbook であり、主役は
- learner 診断統計
- 条件間の大きな傾向
である。

時間効率を優先し、まずは `30` で比較する。  
最終確認が必要なら次段で `50` に戻す。

---

## 4. 実行方式

### 4.1 参照 run（REF）
各 seed で 1 本だけ full run を作る。  
この run は
- imitation
- selfplay
- learner
- eval
を含むが、**以後の再利用元** として使うのが主目的である。

### 4.2 再利用 run
各 seed について、REF から
- `imitation`
- `selfplay`
- `eval_before`
を再利用し、
- `learner`
- `eval`
だけを条件別に回す。

### 4.3 run_map
`experiments/exp_018/run_map.json` を正とする。  
最低限、以下を保持する。

- `seed`
- `role`（ref / A / B / C）
- `run_dir`
- `source_run_dir`
- `epochs`
- `lr`

---

## 5. 実行コマンド

## 5.1 参照 run（seed=42 の例）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    imitation.num_workers=10 \
    selfplay.imitation_matches=25 \
    training.imitation_epochs=4 \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    feature_encoder.shanten_hint='{"enabled":true}' \
    training.imitation_loss_mode=tie_aware_best_set \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=30 \
    evaluation.num_workers=10 \
    training.lr=0.0001 \
    training.epochs=4 \
    training.value_loss_coef=0.25 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

## 5.2 条件A baseline（seed=42 の例）
```bash
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
    training.lr=0.0001 \
    training.epochs=4 \
    training.value_loss_coef=0.25 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

## 5.3 条件B weak-epochs（seed=42 の例）
```bash
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
    training.lr=0.0001 \
    training.epochs=2 \
    training.value_loss_coef=0.25 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

## 5.4 条件C weak-lr（seed=42 の例）
```bash
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
    training.lr=0.00005 \
    training.epochs=4 \
    training.value_loss_coef=0.25 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

## 5.5 seed=43..46
上記と同様に、各 seed で
- 参照 run 1 本
- A/B/C の再利用 run 3 本
を実行する。

**注意**
- `--reuse-from` は `--seeds` と併用不可
- 必ず seed ごとに単発実行する
- `run_map.json` で対応を管理する

---

## 6. 成功判定

### 6.1 参照 run
- `summary.json.success == true`
- `summary.phase_status.imitation == "success"`
- `summary.phase_status.selfplay == "success"`
- `summary.phase_status.learner == "success"`
- `summary.phase_status.eval == "success"`
- `summary.encoder_features.shanten_hint == true`
- `summary.phase_stats.imitation.imitation_loss_mode == "tie_aware_best_set"`
- `summary.phase_stats.eval.eval_mode == "rotation"`
- `eval_before/eval_rotation.json`
- `eval/eval_rotation.json`
- `eval/eval_diff.json`
が存在する

### 6.2 再利用 run
- `summary.json.success == true`
- `summary.reuse_info.reused_phases` に `imitation,selfplay,eval_before`
- `summary.phase_stats.imitation.imitation_loss_mode == "tie_aware_best_set"`
- `summary.phase_stats.eval.eval_mode == "rotation"`
- `eval/eval_diff.json` が存在
- `eval_diff` の主要4指標 delta が非 `null`
- learner 診断統計が run 成果物から読める
  - `advantage_*`
  - `return_*`
  - `old_value_*`
  - `value_error_*`
  - `ratio_*`
  - `clip_fraction`

### 6.3 条件比較に必要な最低項目
各 run で少なくとも以下が取得できること。
- `Δavg_rank`
- `Δavg_score`
- `Δdeal_in_rate`
- `Δwin_rate`
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`
- learner 診断統計
- self-play 統計
- reward オフライン集計元の shard

---

## 7. 集計方法

## 7.1 主比較
条件 A/B/C について、seed ごとに run を対応づけて比較する。

### 主評価
- `eval_before -> eval` の差分
  - `Δavg_rank`
  - `Δavg_score`
  - `Δdeal_in_rate`
  - `Δwin_rate`

### 副評価
- after 指標
  - `avg_rank`
  - `avg_score`
  - `win_rate`
  - `deal_in_rate`

## 7.2 learner 診断統計
run 単位または batch_summary の `runs[*].learner_diag` 相当から比較する。

最低限見るもの:
- `advantage_mean`, `advantage_std`, `advantage_p90`, `advantage_p99`
- `advantage_positive_ratio`, `advantage_negative_ratio`
- `return_mean`, `return_std`, `return_p90`, `return_p99`
- `old_value_mean`, `old_value_std`
- `value_error_mean`, `value_error_std`, `value_error_p90`, `value_error_p99`
- `ratio_mean`, `ratio_std`, `ratio_p90`, `ratio_p99`
- `clip_fraction`
- `policy_loss`, `value_loss`, `entropy`, `num_updates`, `total_steps`

## 7.3 self-play 統計
`summary.phase_stats.selfplay` から run 単位で比較する。

- `policy_wins`
- `policy_deal_ins`
- `policy_draws`
- `tsumo_count`
- `ron_count`
- `ryukyoku_count`
- `num_rounds`
- `total_steps`

## 7.4 reward / round_over reward 分布
これは **オフライン集計** で確認する。  
対象は参照 run の self-play shard。

見るもの:
- reward:
  - mean / std / min / max / p50 / p90 / p99
- round_over reward:
  - mean / std / min / max / p50 / p90 / p99
- `round_over` sample 数

---

## 8. 期待する読み方

### 8.1 weak-epochs / weak-lr で delta が改善する場合
- PPO の **更新強度** が強すぎた可能性が高い
- tie-aware 初期方策は一定の価値があるが、baseline PPO が壊しすぎていた

### 8.2 delta は改善しないが learner 統計が大きく変わる場合
- 更新の強さだけでなく
  - advantage / return のスケール
  - value 誤差
  - ratio / clip の挙動
が問題
- 方向の悪さや target quality の問題を疑う

### 8.3 clip_fraction が高い / ratio tail が重い場合
- PPO 更新が clip に強く当たりすぎている
- 更新幅過大のサイン

### 8.4 value_error が大きい場合
- value 推定が target に追いついていない
- policy 更新が noisy になる原因の可能性

### 8.5 reward 分布が極端に sparse / heavy-tail の場合
- reward 設計や round_over 報酬が policy 改善に向いていない可能性
- credit assignment の弱さを疑う

---

## 9. レポートに必ず含める項目

### 9.1 実行対応
- `experiments/exp_018/run_map.json` を正とする
- 各 seed の
  - REF
  - A
  - B
  - C
  の対応表

### 9.2 条件別比較表
条件 A/B/C について
- `Δavg_rank`
- `Δavg_score`
- `Δdeal_in_rate`
- `Δwin_rate`
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 9.3 learner 診断統計比較表
最低限:
- `advantage_mean/std/p90/p99`
- `return_mean/std/p90/p99`
- `old_value_mean/std`
- `value_error_mean/std/p90/p99`
- `ratio_mean/std/p90/p99`
- `clip_fraction`

### 9.4 self-play 統計
- `policy_wins`
- `policy_deal_ins`
- `policy_draws`
- `tsumo_count`
- `ron_count`
- `ryukyoku_count`
- `num_rounds`

### 9.5 reward 分布
各 seed の REF から集計し、条件比較の前提として報告する。
- reward mean/std/min/max/p50/p90/p99
- round_over reward mean/std/min/max/p50/p90/p99
- `round_over` sample 数

### 9.6 時間
- 参照 run の平均時間
- 再利用 run の平均時間
- learner/eval の支配率

### 9.7 解釈
以下に答えること。
1. weak-epochs / weak-lr は baseline より壊れにくいか
2. その差は learner 診断統計で説明できるか
3. reward / self-play 分布に明らかな異常があるか
4. 次に見るべきは
   - 更新強度の更なる調整か
   - reward / target 設計か
   - self-play 分布改善か

---

## 10. 最終判断ルール

### ケースA
- weak-epochs または weak-lr で delta が明確に改善
- learner 診断でも
  - ratio tail 縮小
  - `clip_fraction` 低下
  - value_error 改善
などが見える

→ 主因は **更新強度過大** 寄り  
→ PPO を弱めた条件で次段探索

### ケースB
- delta 改善は小さい
- learner 診断で
  - value_error 大
  - advantage / return 分布が荒い
  - reward が heavy-tail
が目立つ

→ 主因は **target / value / reward 設計** 寄り  
→ reward / value 側の設計見直しへ

### ケースC
- 条件差が小さく
- self-play 統計や reward 分布のばらつきが大きい

→ 主因は **self-play 分布ノイズ** 寄り  
→ self-play 条件や再利用戦略を見直す

---

## 11. 実行上の注意

- `--reuse-from` は `--seeds` と併用不可
- 必ず seed ごとに単発実行する
- REF → A/B/C の対応を `run_map.json` で厳密に管理する
- reward 分布は REF shard から集計する
- self-play を再生成しないことで、learner 差だけを比較しやすくする
- この runbook は診断目的なので、まずは `evaluation.num_matches=30` を使う
- 良い方向が見えたら次段で `50` に戻して確認する

---

## 12. この runbook の意味

この runbook の目的は、単に
- `epochs=2` が良いか
- `lr=5e-5` が良いか
を選ぶことではない。

本当に知りたいのは、

> **PPO は imitation 初期方策を「強く更新しすぎて壊している」のか、  
> それとも「そもそも悪い方向へ更新している」のか**

である。

この runbook により、
- 更新強度
- learner 診断統計
- reward / self-play 分布
をまとめて見て、次の改善対象を絞る。

---