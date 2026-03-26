# experiments/exp_014/runbook.md（Runbook 14）

最終更新: 2026-03-07  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **シャンテン補助特徴（shanten hint）on/off を imitation-only 相当で比較し、初期方策の到達点を改善できるか確認する**

---

## 0. この実験の位置づけ

ここまでの実験で、learner 側の主要ノブはかなり詰まってきた。

現時点の暫定 baseline は以下。

- `training.lr=0.0001`
- `training.epochs=4`
- `training.value_loss_coef=0.25`
- `training.clip_epsilon=0.2`
- `training.batch_size=256`
- `training.gae_lambda=0.95`

一方で、依然として以下の課題が残っている。

- 絶対性能はまだ低い
- imitation は効くが、まだ初期到達点が十分高くない
- PPO は改善するが、still 少し壊しがち

また、過去の個人実験では、

> **各打牌候補について、「この牌を切るとシャンテンが改善するか」を特徴量として入れると非常に効いた**

という経験がある。

今回、`FlatFeatureEncoder` に最小のシャンテン補助特徴が実装されたため、  
まずは **imitation-only 相当** でその効果を切り出して確認する。

---

## 1. この実験の意図

### 1.1 何を知りたいのか
今回知りたいのは、シャンテン補助特徴が

> **imitation learning の到達点を明確に押し上げるか**

である。

ここではまだ PPO まで含めず、  
まず **「初期方策の質を上げる補助特徴として価値があるか」** を確認する。

### 1.2 なぜ imitation-only 相当にするのか
いきなり PPO まで含めると、

- 特徴量の効果
- PPO の効果
- PPO による破壊/上積み

が混ざって解釈しにくくなる。

したがって最初は、
- シャンテン補助特徴なし
- シャンテン補助特徴あり

を **imitation 後の評価** で直接比較する。

### 1.3 今回の問い
- shanten hint は imitation-only の到達点を改善するか
- 改善するなら
  - `avg_rank`
  - `avg_score`
  - `win_rate`
  - `deal_in_rate`
のどこに出るか
- 入力次元増加に対して、学習や評価は安定して完走するか

---

## 2. 実験方針

### 2.1 比較対象
- A: `shanten_hint=off`（現行特徴量）
- B: `shanten_hint=on`（シャンテン補助特徴あり）

### 2.2 実行方式
現行実装では、`imitation` を使う場合 `selfplay` phase が必要。  
したがって、imitation-only 相当の最短経路として以下を使う。

- `experiment.phases='["imitation","selfplay","eval"]'`
- `selfplay.num_matches=0`

これにより、
- imitation は実行される
- PPO learner は実行されない
- `eval` は **imitation 後モデル** の評価になる

### 2.3 評価の扱い
この経路では `eval_before` は出ない。  
したがって主評価は `eval_before -> eval` の差分ではなく、**最終評価（after 指標）そのもの** を比較する。

見る指標:
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

---

## 3. 実験条件

### 3.1 共通条件
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- `experiment.phases='["imitation","selfplay","eval"]'`
- `imitation.num_workers=10`
- `selfplay.imitation_matches=25`
- `training.imitation_epochs=4`
- `selfplay.num_matches=0`
- `selfplay.num_workers=10`
- `selfplay.policy_ratio=1.0`
- `selfplay.save_baseline_actions=false`
- `evaluation.mode=rotation`
- `evaluation.rotation_seats='[0,1,2,3]'`
- `evaluation.num_matches=50`
- `evaluation.num_workers=10`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

### 3.2 比較条件
- A: `feature_encoder.shanten_hint='{"enabled":false}'`
- B: `feature_encoder.shanten_hint='{"enabled":true}'`

### 3.3 shanten hint 設定の注意
CLI override では、3段ドットキーは使えない。  
したがって on/off は以下のように **JSON 値** で渡す。

- OFF: `feature_encoder.shanten_hint='{"enabled":false}'`
- ON: `feature_encoder.shanten_hint='{"enabled":true}'`

---

## 4. 実験規模

今回は learner ノブ比較ではなく、**特徴量の有効性確認**である。  
ただし、将来の本採用判断につながる可能性があるため、最初から判断力を持たせる。

- `5 seeds`
- `rotation eval`
- `evaluation.num_matches=50`

---

## 5. 実行コマンド

### 5.1 条件A: shanten hint off
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    experiment.phases='["imitation","selfplay","eval"]' \
    imitation.num_workers=10 \
    selfplay.imitation_matches=25 \
    training.imitation_epochs=4 \
    selfplay.num_matches=0 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    feature_encoder.shanten_hint='{"enabled":false}' \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 5.2 条件B: shanten hint on
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    experiment.phases='["imitation","selfplay","eval"]' \
    imitation.num_workers=10 \
    selfplay.imitation_matches=25 \
    training.imitation_epochs=4 \
    selfplay.num_matches=0 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    feature_encoder.shanten_hint='{"enabled":true}' \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

---

## 6. 成功判定

### 6.1 各 batch の最低条件
- `success_count == 5/5`
- `aggregate.eval_mode == "rotation"`

### 6.2 各 run の確認項目
- `summary.json.success == true`
- `summary.phase_status.imitation == "success"`
- `summary.phase_status.selfplay == "success"`
- `summary.phase_status.eval == "success"`
- `summary.encoder_features.shanten_hint` が意図通り
- `summary.encoder_features.input_dim` が記録されている
- `summary.phase_stats.eval.eval_mode == "rotation"`
- `eval/eval_rotation.json` が存在する

### 6.3 selfplay.num_matches=0 の扱い
- `selfplay` phase 自体は走る
- `total_matches=0` は正常
- ここでの `selfplay` は imitation-only 相当経路を成立させるための最小経路である

---

## 7. 集計方法

この実験は通常の multi-seed batch 実行が使える。  
したがって、条件 A/B それぞれについて

- `batch_summary.json` の `aggregate`
- `runs[*].eval_metrics`

を使って比較する。

### 7.1 主確認先
- `summary.json` → `phase_stats.eval`
- `eval/eval_rotation.json`（run 単位）
- `batch_summary.json.aggregate`（条件集約）

### 7.2 主な集計指標
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

---

## 8. 主な評価項目

### 8.1 最優先
今回は after 指標そのものを主評価にする。

見る順序:
1. `avg_rank`
2. `avg_score`
3. `win_rate`
4. `deal_in_rate`

### 8.2 補助確認
- `imitation_loss` が summary/run artifact から読めるなら参考として記録
- `encoder_features.shanten_hint`
- `encoder_features.input_dim`
- `notes.md` の `shanten_hint=on/off, input_dim=...`

### 8.3 実行時間
- imitation 時間
- eval 時間
- total 時間

特徴量追加で時間が大きく悪化していないかも確認する。

---

## 9. 結果の読み方

### 9.1 条件B（shanten hint on）が明確に良い場合
- シャンテン補助特徴は imitation-only 到達点を押し上げる
- flat + MLP の現構成では、強い補助特徴の価値が高い
- 次は **warm start + PPO で on/off 比較** に進む

### 9.2 条件A/B がほぼ同等の場合
- 少なくとも imitation-only では決定打になっていない
- learner 側の改善や別特徴量を優先する余地がある
- ただし一部指標だけ改善しているなら次段比較を検討する

### 9.3 条件B が悪い場合
- shortcut 依存や表現干渉の可能性がある
- 今の形での導入は見送る
- 必要なら入れ方を弱める / auxiliary 化を検討する

---

## 10. 判定ルール

### 10.1 採用判断
以下を満たす場合、shanten hint は有望とみなす。

- `avg_rank` が改善
- `avg_score` が改善
- `win_rate` が改善
- `deal_in_rate` が悪化しすぎない

### 10.2 次段へ進める基準
次の **warm start + PPO 比較** に進める条件は、

- `avg_rank` または `avg_score` が明確に改善
- `win_rate` が改善または同等
- `deal_in_rate` の悪化が小さい
- 実行時間増が許容範囲

### 10.3 差が僅差の場合
差が僅差なら、
- `avg_rank`
- `avg_score`
を優先し、
- 実行時間
- 学習安定性
を補助的に見る

---

## 11. レポートに必ず含める項目

- batch_dir（A/B）
- 成功率
- `aggregate.eval_mode`
- `encoder_features.shanten_hint`
- `encoder_features.input_dim`
- after 指標（mean ± std）
  - `avg_rank`
  - `avg_score`
  - `win_rate`
  - `deal_in_rate`
- 可能なら imitation loss
- 所要時間
  - imitation
  - eval
  - total
- shanten hint on/off の比較解釈
- 次アクション

---

## 12. この実験の副次目的

この runbook には、shanten hint の有効性確認に加えて、

> **新特徴量 on/off が現行の runbook / report / summary 運用で追跡可能か**

を確認する副次目的がある。

確認したい点:
- `summary.json` で on/off が一意に分かるか
- `input_dim` が追跡できるか
- `notes.md` の記録が十分か
- report で比較しやすいか

---

## 13. 次のアクション

### 13.1 shanten hint on が良い場合
- 次は **warm start + PPO で on/off 比較**
- そのときは learner baseline を固定して、特徴量差だけを見る

### 13.2 shanten hint on が微妙だが一部良い場合
- 追加比較をするか
- 入れ方を弱めるか
- 補助特徴ではなく auxiliary target にするか
を検討する

### 13.3 shanten hint on が悪い場合
- 現形での導入は見送る
- gamma など残ノブに戻るか
- 別の特徴量（ukeire / safety）はまだ入れない

---

## 14. メモ

- 今回は「本採用」ではなく、**補助特徴の有効性確認実験**
- PPO まで含めず、まず imitation-only で切り出して見る
- ここで明確な改善が出るなら、強い補助特徴導入の価値がかなり高い

---