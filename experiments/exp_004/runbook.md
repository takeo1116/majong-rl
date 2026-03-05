# experiments/exp_004/runbook.md（Runbook 4）

最終更新: 2026-03-05  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **seat バイアスを除去（rotation eval）**した上で、A（PPO）と E（軽量 warm start）の優劣を固める

---

## 0. 位置づけ

Runbook 3 で得られた結論（10 seed）:
- warm start（軽量 E）が最終到達点で優位傾向
- ただし評価が単席 eval であり seat バイアスが残る懸念

Runbook 4 では rotation eval を使って、**A vs E の結論を固める**。

---

## 1. 前提

- multi-seed batch 実行が利用可能
- rotation eval の集約は `eval_metrics` を通して batch 集約で扱える
- worker 標準は `imitation/selfplay/eval = 10`
- 交絡排除のため、A/Eで以下を揃える:
  - `selfplay.policy_ratio=1.0`
  - `selfplay.save_baseline_actions=false`

---

## 2. 重要な注記（rotation 評価の実効並列度）

現在の実装では、rotation 評価時の worker 割当は「席ごとに `evaluation.num_workers // 4`」で分配される。  
そのため `evaluation.num_workers=10` を指定しても、実効 worker 数は以下になる。

- `10 // 4 = 2`
- 実効 worker = `2 × 4席 = 8`

この Runbook では **A/E の両方で同じ設定を使う**ため比較の公平性は保たれるが、  
「`num_workers=10` がそのまま10並列で動く」わけではない点に注意する。

---

## 3. 比較対象

### A（基準）
- config: `configs/stage1_full_flat_mlp_ppo.yaml`
- warm start: なし（PPO直学習）

### E（軽量 warm start, 交絡排除版）
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- warm start: あり（軽量）
  - `selfplay.imitation_matches=25`
  - `training.imitation_epochs=4`
  - `imitation.num_workers=10`

---

## 4. 実験条件（共通）

### 4.1 seeds（まず5本）
- `42,43,44,45,46`

### 4.2 worker（統一）
- `imitation.num_workers=10`（Eのみ必要）
- `selfplay.num_workers=10`
- `evaluation.num_workers=10`（※rotationでは実効8 worker）

### 4.3 実験予算（Runbook3と同等）
- `selfplay.num_matches=200`
- `training.epochs=4`

### 4.4 評価（rotation）
- `evaluation.mode="rotation"`
- `evaluation.rotation_seats=[0,1,2,3]`
- `evaluation.num_matches=50`（推奨）
  - ※差が曖昧なら 100 へ増やして確証を取る

### 4.5 device（標準）
- `training.device="cuda"`（利用可能なら）
- `selfplay.inference_device="cpu"`
- `evaluation.inference_device="cpu"`

---

## 5. 実行コマンド

### 5.1 A（PPO直学習, rotation）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.epochs=4 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 5.2 E（軽量 imitation→PPO, rotation, 交絡排除）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44,45,46 \
  --override \
    imitation.num_workers=10 \
    selfplay.imitation_matches=25 \
    training.imitation_epochs=4 \
    selfplay.num_matches=200 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.epochs=4 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

補足:
- `eval_strict` を厳密に踏襲するなら `evaluation.num_matches=100` に変更する。

---

## 6. 実行コスト目安（概算）

rotation eval は single の約4倍程度になりやすい。

- 1 seed あたり `eval_before + eval` で **約9〜12分**程度の見込み
- 5 seed で **45〜60分**以上 + selfplay/learner/imitation 分

---

## 7. 成功判定

各 run / batch で以下を確認する。

- `summary.json.success == true`
- `eval_metrics.eval_mode == "rotation"`
- run_dir に `eval/eval_rotation.json` が存在する（参考）
- batch_dir に `batch_summary.json` / `batch_table.csv` が出る

---

## 8. 比較の見方

比較する主指標（rotation総合）:
- `avg_rank`（小さいほど良い）
- `avg_score`（大きいほど良い）
- `win_rate`（大きいほど良い）
- `deal_in_rate`（小さいほど良い）

見る場所:
- `batch_summary.json`
- `batch_table.csv`

信頼性を見る:
- SE / CI / outlier（対応していれば）

---

## 9. 判定ルール（Runbook4の結論）

### 9.1 Eが優位と判断
- `avg_rank` と `avg_score` が平均で改善し、CIがAと重なりにくい
- `win_rate` が明確に上がる or `deal_in_rate` が下がる

### 9.2 差が曖昧な場合
- `evaluation.num_matches=100` に上げて同じ比較を再実行（5 seedのまま）
- それでも曖昧なら、seedを10本に拡張して判断

---

## 10. 次のステップ（Runbook4後）

- E優位が固まった場合:
  - **PPO段の上積み改善（sweep）**へ移行（Eを起点）
- 差が曖昧/逆転した場合:
  - warm start の設定（matches/epochs/filter）を再探索
  - または PPO直学習を標準へ戻す

---
