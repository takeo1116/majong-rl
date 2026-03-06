# experiments/exp_007/runbook.md（Runbook 7）

最終更新: 2026-03-06  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **`training.lr=0.0001` が「本当に良い更新」なのか、それとも「壊れるのが遅いだけ」なのかを切り分ける**

---

## 0. この実験の意図（Codex向けに明記）

Runbook 5 / 6 で分かったこと:

- E（軽量 imitation→PPO）は A（PPO直学習）より良い
- ただし PPO 段は依然として `eval_before -> eval` を悪化させがち
- `training.lr=0.0001` は、既定値より **悪化幅を小さくした**
- しかし、ここで未解決なのは次の問いである

> `training.lr=0.0001` は  
> **「良い更新方向だから改善しやすい」のか**  
> それとも  
> **「単に更新が遅いだけで、壊れるのが遅延している」のか**

この Runbook 7 は、この問いを切り分けるために行う。  
そのため、**low-lr を固定したまま epochs を動かす**。

### この実験で見たいこと
- `lr=0.0001` で epochs を増やしたとき、
  - 改善が維持される / さらに良くなる  
    → low-lr は本当に有望
  - 結局悪化する  
    → low-lr は単に壊れるのが遅いだけ
- さらに、entropy を少し下げると、imitation の良さを保ちやすくなるかも確認する

---

## 1. 基本方針

### 1.1 ベース設定
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`

### 1.2 固定条件
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
- `evaluation.num_matches=20`
- `evaluation.num_workers=10`
- `training.lr=0.0001`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

### 1.3 worker に関する注記
rotation eval では `evaluation.num_workers=10` を指定しても、実装上の実効 worker 数は 8 になる。  
ただし全条件で同じ設定を使うため比較の公平性は保たれる。

---

## 2. 比較条件

### 条件A: low-lr + epochs=2
- `training.epochs=2`

### 条件B: low-lr + epochs=4（基準）
- `training.epochs=4`

### 条件C: low-lr + epochs=8
- `training.epochs=8`

### 条件D: low-lr + epochs=4 + entropy低下
- `training.epochs=4`
- `training.entropy_coef=0.005`

---

## 3. この4条件を選ぶ理由

### 条件A / B / C
low-lr のまま epochs を動かして、
- 改善の最適点があるのか
- 単に更新が遅いだけなのか
を切り分ける。

### 条件D
もし imitation の良い初期方策を entropy が崩しているなら、
entropy を弱めることで保持しやすくなる可能性がある。  
ただし、まず本命は epochs 比較であり、D は補助線である。

---

## 4. 実験予算

### 4.1 小規模探索
- seeds: `42,43,44`
- `evaluation.num_matches=20`

理由:
- まず傾向を見る段階だから
- 4条件でも回しやすい
- 良い条件だけ次段で 5 seed / 50 matches に拡張する

### 4.2 次段
有望条件が見つかったら、
- seeds: `42,43,44,45,46`
- `evaluation.num_matches=50`
で確証を取る。

---

## 5. 実行コマンド

### 5.1 条件A: low-lr + epochs=2
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
    training.epochs=2 \
    training.lr=0.0001 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 5.2 条件B: low-lr + epochs=4（基準）
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

### 5.3 条件C: low-lr + epochs=8
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
    training.epochs=8 \
    training.lr=0.0001 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 5.4 条件D: low-lr + epochs=4 + entropy低下
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
    training.entropy_coef=0.005 \
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
各条件について、各 seed の `runs[].eval_diff` を確認する。

主に見る指標:
- `Δavg_rank`
- `Δavg_score`
- `Δwin_rate`
- `Δdeal_in_rate`

### 7.2 この実験で特に知りたいこと
#### ケース1
epochs を増やしても改善が維持 / 向上する  
→ `lr=0.0001` は「良い更新」の可能性が高い

#### ケース2
epochs を増やすと結局悪化する  
→ `lr=0.0001` は「壊れるのが遅いだけ」の可能性が高い

#### ケース3
epochs に最適点がある  
→ `lr=0.0001` は正しいが、更新量に最適点がある

### 7.3 次に見るもの
after の最終指標:
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

---

## 8. self-play 統計で確認すること

各 run の `summary.json.phase_stats.selfplay` から以下を確認する。

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
を補助的に確認する。

---

## 9. 判定ルール

### 9.1 最有力条件
次段（5 seed / 50 matches）に進める条件は、少なくとも以下を満たすものとする。

- `Δavg_rank` が最も小さい（できれば負）
- `Δavg_score` が最も大きい（できれば正）
- `Δdeal_in_rate` が最も小さい
- after 指標も大きく悪くない

### 9.2 追加確認が必要な場合
もし A/B/C/D が拮抗するなら、
- seeds を 5 本へ
- `evaluation.num_matches=50`
で再比較する

### 9.3 どれも悪化する場合
その場合は次に
- `training.value_loss_coef`
- `training.entropy_coef` の追加段階
- あるいは reward / advantage 周り
の検討に進む

---

## 10. 所要時間の目安

- 3 seed × 4 条件
- rotation eval 20

で、Runbook 5 と同程度の規模感を想定する。

---

## 11. メモ

- この実験の意図は「新しいベスト条件探し」ではなく、  
  **low-lr が本当に良いのか、遅いだけなのかを切り分けること**
- そのため、after の最終指標だけでなく、  
  **delta（before→after）** を主評価対象とする
- self-play 統計も合わせて確認し、何が改善/悪化したかを読む

---