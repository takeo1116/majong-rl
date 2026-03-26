# experiments/exp_006/runbook.md（Runbook 6）

最終更新: 2026-03-06  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **Runbook 5 で最有力だった条件 C（`training.lr=0.0001`）を、5 seed / rotation eval 50 で再確認する**

---

## 0. 位置づけ

Runbook 5 で確認できたこと:

- E（軽量 imitation→PPO）を起点にした小規模 sweep では、
  **`training.lr=0.0001`** が最良候補だった
- 特に `eval_before -> eval` の悪化が最も小さく、
  `avg_rank` / `avg_score` / `deal_in_rate` は改善方向だった
- after の最終指標も 4 条件中で最良だった

Runbook 6 では、この条件 C を  
**より信頼できる条件（5 seed / rotation eval 50）で確証取りする**。

---

## 1. 実験条件

### 1.1 ベース設定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`

### 1.2 seed
- `42,43,44,45,46`

### 1.3 固定条件
- `imitation.num_workers=10`
- `selfplay.imitation_matches=25`
- `training.imitation_epochs=4`
- `selfplay.num_matches=200`
- `selfplay.num_workers=10`
- `selfplay.policy_ratio=1.0`
- `selfplay.save_baseline_actions=false`
- `evaluation.mode=rotation`
- `evaluation.rotation_seats='[0,1,2,3]'`
- `evaluation.num_matches=50`
- `evaluation.num_workers=10`
- `training.epochs=4`
- `training.lr=0.0001`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

### 1.4 rotation eval に関する注記
`evaluation.num_workers=10` を指定しても、rotation 評価では実装上の実効 worker 数は 8 になる。  
ただし今回は単一条件の確証取りなので問題ない。

---

## 2. 実行コマンド

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
    training.lr=0.0001 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

---

## 3. 成功判定

batch 実行後、以下を確認する。

- `success_count == 5/5`
- `aggregate.eval_mode == "rotation"`
- 各 run の `runs[].eval_diff` が存在する
  - これは `eval_before -> eval` の差分が計算済みであることを意味する
- 必要なら各 `run_dir` の `eval/` 配下と `run.log` を参照する

---

## 4. 主な確認項目

### 4.1 最優先
`eval_before -> eval` の平均差分が、少なくとも **悪化していない**かを確認する。

見る指標:
- `Δavg_rank`
- `Δavg_score`
- `Δwin_rate`
- `Δdeal_in_rate`

### 4.2 次に見るもの
after の最終指標:
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 4.3 self-play 統計
各 run の `summary.json.phase_stats.selfplay` から、少なくとも以下を確認する。

- `policy_wins`
- `policy_deal_ins`
- `policy_draws`
- `tsumo_count`
- `ron_count`
- `ryukyoku_count`

必要なら `selfplay/worker_*/round_results.jsonl` も確認する。

---

## 5. 判定ルール

### 5.1 条件 C を新しい標準候補とする条件
以下を満たせば、`training.lr=0.0001` を新しい標準候補とみなす。

- `Δavg_rank <= 0` に近い、または負
- `Δavg_score >= 0` に近い、または正
- `Δdeal_in_rate <= 0`
- after の最終指標も E の従来設定以上である

### 5.2 条件 C がまだ不安定な場合
- 5 seed でも delta が悪化寄りなら、次は
  - `training.epochs=2`
  - `training.entropy_coef` 減少
  - `training.value_loss_coef` 調整
の順で見る

### 5.3 条件 C が有望だった場合
次はこの順で進む。

1. `training.lr=0.0001` を新 baseline とする
2. 追加 sweep
   - `training.epochs=2`
   - `training.entropy_coef` 減少
   - 必要なら `training.value_loss_coef`
3. さらに良ければ 10 seed へ拡張

---

## 6. レポートに必ず含める項目

- batch_dir
- 成功率
- after 指標（mean ± std）
  - `avg_rank`
  - `avg_score`
  - `win_rate`
  - `deal_in_rate`
- `eval_before -> eval` の平均差分
  - `Δavg_rank`
  - `Δavg_score`
  - `Δwin_rate`
  - `Δdeal_in_rate`
- self-play 統計
  - `policy_wins`
  - `policy_deal_ins`
  - `policy_draws`
  - `tsumo_count`
  - `ron_count`
  - `ryukyoku_count`
- 所要時間
  - imitation
  - selfplay
  - eval_before
  - learner
  - eval
  - total

---

## 7. メモ

- この実験の目的は「さらに勝つ設定を探す」前に、
  **条件 C が本当に改善方向かを固めること**
- 今回は after の最終性能だけでなく、
  **delta（before→after）** を主に見る
- self-play 統計もあわせて確認して、
  何が改善/悪化しているかを読む

---