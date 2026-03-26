# experiments/exp_005/runbook.md（Runbook 5）

最終更新: 2026-03-06  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **E（軽量 imitation→PPO）を起点に、PPO 段が方策を壊さない設定を探す**

---

## 0. 位置づけ

Runbook 4 で分かったこと:

- A（PPO直学習）より E（軽量 warm start）が rotation eval でも優位
- ただし E の `eval_before -> eval` では PPO 段が悪化させている可能性が高い
- imitation を増やせば伸びるわけではなく、現状は
  - `imitation_matches=25`
  - `training.imitation_epochs=4`
  が最良候補

Runbook 5 では、**E を固定し、PPO 更新を弱める方向の小規模 sweep** を行う。

---

## 1. 基本方針

### 1.1 比較の主目的
最終 after 指標も見るが、まず重視するのは

- `eval_before -> eval` の `delta`

である。

目標はまず
- **悪化しない**
こと。  
次に
- **改善する**
ことを狙う。

### 1.2 固定するもの
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44`
- `imitation.num_workers=10`
- `selfplay.imitation_matches=25`
- `training.imitation_epochs=4`
- `selfplay.num_matches=200`
- `selfplay.num_workers=10`
- `selfplay.policy_ratio=1.0`
- `selfplay.save_baseline_actions=false`
- `evaluation.mode=rotation`
- `evaluation.rotation_seats='[0,1,2,3]'`
- `evaluation.num_workers=10`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

### 1.3 worker に関する注記
rotation eval では `evaluation.num_workers=10` を指定しても、実装上の実効 worker 数は 8 になる。  
ただし全条件で同じ設定を使うため、比較の公平性は保たれる。

---

## 2. sweep 条件

### 条件A: baseline
現状の E 設定そのまま。

### 条件B: lr↓
- `training.lr=0.0002`

### 条件C: lr↓↓
- `training.lr=0.0001`

### 条件D: clip↓
- `training.clip_epsilon=0.1`

### この4条件を選ぶ理由
現状の悪化は「PPO 更新が強すぎる」典型に見えるため、まずは

- 学習率を下げる
- clip を小さくする

の2軸を試すのが最短である。

---

## 3. 実験予算

### 3.1 推奨評価予算
まずは **小規模探索** として以下を採用する。

- `evaluation.num_matches=20`

理由:
- 3 seed × 4 条件でも現実的な時間に収まる
- まず傾向を見る段階だから

### 3.2 次段
良さそうな条件が見つかったら、同条件で

- seed を 5 本へ拡張
- `evaluation.num_matches=50`

として確証を取りに行く。

---

## 4. 比較条件一覧

| 条件 | 追加 override |
|---|---|
| A | なし |
| B | `training.lr=0.0002` |
| C | `training.lr=0.0001` |
| D | `training.clip_epsilon=0.1` |

---

## 5. 実行コマンド

### 5.1 条件A: baseline
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44 \
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
    evaluation.num_matches=20 \
    evaluation.num_workers=10 \
    training.epochs=4 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 5.2 条件B: lr↓
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44 \
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
    evaluation.num_matches=20 \
    evaluation.num_workers=10 \
    training.epochs=4 \
    training.lr=0.0002 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 5.3 条件C: lr↓↓
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44 \
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
    evaluation.num_matches=20 \
    evaluation.num_workers=10 \
    training.epochs=4 \
    training.lr=0.0001 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 5.4 条件D: clip↓
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --seeds 42,43,44 \
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
    evaluation.num_matches=20 \
    evaluation.num_workers=10 \
    training.epochs=4 \
    training.clip_epsilon=0.1 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

---

## 6. 成功判定

各条件の batch について以下を確認する。

- `success_count == 3/3`
- `aggregate.eval_mode == "rotation"`
- 各 run の `runs[].eval_diff` が存在する
  - これは `eval_before -> eval` の差分が計算済みであることを意味する
- 必要なら詳細確認として、各 `run_dir` の `eval/` 配下と `run.log` を参照する

---

## 7. 比較の見方

### 7.1 最優先で見るもの
各条件について、各 seed の `runs[].eval_diff` を見る。

特に見る指標:
- `Δavg_rank`
- `Δavg_score`
- `Δwin_rate`
- `Δdeal_in_rate`

### 7.2 良い条件の基準
まずは以下を満たす条件を候補とする。

- `Δavg_rank <= 0` に近い、または負
- `Δavg_score >= 0` に近い、または正
- `Δwin_rate >= 0`
- `Δdeal_in_rate <= 0`

### 7.3 次に見るもの
after の最終指標:
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

---

## 8. self-play 統計で追加確認すること

各 run の `summary.json.phase_stats.selfplay` から、少なくとも以下を確認する。

- `policy_wins`
- `policy_deal_ins`
- `policy_draws`
- `tsumo_count`
- `ron_count`
- `ryukyoku_count`

必要なら `selfplay/worker_*/round_results.jsonl` を見て、
- 和了が減っているのか
- 放銃が増えているのか
- 流局寄りになっているのか
を確認する。

### 見たい変化
- PPO後に `policy_wins` が減っていないか
- `policy_deal_ins` が増えていないか
- `policy_draws` が極端に増えていないか

---

## 9. 所要時間の目安

Codex 見積もり:
- 3 seed × 4条件
- `evaluation.num_matches=20`

でおおむね **1〜1.5時間**

---

## 10. Runbook5 の結論ルール

### 10.1 明確な勝ち条件
ある条件が以下を満たすなら有望候補:
- baseline より delta が明らかに良い
- after の最終指標も悪くない
- self-play 統計でも和了/放銃の悪化が見えない

### 10.2 次に進む条件
有望候補が 1〜2 個見つかったら、その条件だけ

- seed を 5 本へ拡張
- `evaluation.num_matches=50`

で再確認する。

### 10.3 どれもダメな場合
次段の候補:
- `training.epochs=2`
- `training.entropy_coef` を下げる
- `training.value_loss_coef` を調整する

---

## 11. まだやらないこと

この Runbook ではまだやらない。

- imitation 量の再探索
- モデル大型化
- encoder 変更
- gamma / gae_lambda の探索
- 10 seed 本番比較

---

## 12. メモ

- 今回は「最終性能」よりまず「PPOが壊さないか」を見る
- そのため after の平均だけでなく delta を主に見る
- self-play 統計も合わせて確認する
- 良い候補が見えたら、次に seed と eval を増やして確証を取る

---