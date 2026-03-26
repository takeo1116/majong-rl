# experiments/exp_008/runbook.md（Runbook 8）

最終更新: 2026-03-06  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **`training.lr=0.0001` を固定したとき、`training.epochs=2` と `training.epochs=4` のどちらを次の標準候補にするべきかを決める**

---

## 0. この実験の意図

Runbook 7 で分かったこと:

- `training.lr=0.0001` は、少なくとも既定値より良い方向だった
- ただし `training.epochs=8` では明確に悪化し、過更新寄りの挙動が見えた
- `training.epochs=2` と `training.epochs=4` はどちらも改善方向で、良い帯にある
- まだ不明なのは、
  - **より保守的な `epochs=2` が本当に良いのか**
  - **`epochs=4` でもまだ十分に有効なのか**
  という点である

この Runbook 8 は、  
**low-lr の実用的な最適域の中で、次の標準設定として 2 と 4 のどちらを採用するかを決める**  
ための実験である。

### この実験で切り分けたいこと
- `epochs=2` は、壊れにくい代わりに更新が足りないのか
- `epochs=4` は、まだ有効な更新域なのか
- あるいは、両者がほぼ同等で、実務上はより保守的な `epochs=2` を選ぶべきなのか

### 判定の考え方
この実験では、最終 after 指標だけでなく、  
**`eval_before -> eval` の delta を主評価対象**とする。

理由:
- 今の課題は「PPO 段が imitation の良さを壊していないか」であるため
- まずは **壊さない** ことが重要であり、その次に **上積みできる** かを見るべきだから

---

## 1. 比較対象

### 条件A
- `training.epochs=2`

### 条件B
- `training.epochs=4`

差分はこの 1 項目だけに限定する。  
それ以外の条件は完全に固定し、**1要因比較**にする。

---

## 2. 共通条件

### 2.1 ベース config
- `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`

### 2.2 seed
- `42,43,44,45,46`

### 2.3 imitation / self-play / eval / device
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
- `training.lr=0.0001`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

### 2.4 rotation eval に関する注記
`evaluation.num_workers=10` を指定しても、rotation 評価では実装上の実効 worker 数は 8 になる。  
ただし両条件で同一設定を使うため、比較の公平性は保たれる。

---

## 3. この条件設定の意味

この実験では、以下を固定している。

- imitation の質と量
- self-play の量
- PPO の learning rate
- evaluation の厳しさ
- seed 群

つまり、今回の結果はできるだけ素直に

> **同じ初期方策・同じデータ収集条件・同じ学習率のもとで、  
> 更新回数 2 と 4 のどちらが良いか**

を表すようにしている。

---

## 4. 実行コマンド

### 4.1 条件A: epochs=2
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
    training.epochs=2 \
    training.lr=0.0001 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 4.2 条件B: epochs=4
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

## 5. 成功判定

各条件の batch について以下を確認する。

- `success_count == 5/5`
- `aggregate.eval_mode == "rotation"`
- 各 run の `runs[].eval_diff` が存在する
  - これは `eval_before -> eval` の差分が計算済みであることを意味する
- 必要なら詳細確認として、各 `run_dir` の `eval/` 配下と `run.log` を参照する

---

## 6. 主な評価項目

### 6.1 最優先（delta）
各条件について、まず以下の平均差分を見る。

- `Δavg_rank`
- `Δavg_score`
- `Δwin_rate`
- `Δdeal_in_rate`

この実験では、**最終 after 指標より先に delta を見る**。

### 6.2 次点（after）
そのうえで、最終性能も確認する。

- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 6.3 補助（self-play 統計）
各 run の `summary.json.phase_stats.selfplay` から以下を確認する。

- `policy_wins`
- `policy_deal_ins`
- `policy_draws`
- `tsumo_count`
- `ron_count`
- `ryukyoku_count`

必要なら `selfplay/worker_*/round_results.jsonl` も見る。

---

## 7. どう結果を読むか

### 7.1 条件A（epochs=2）が勝つ場合
これは、

- より少ない更新回数の方が imitation 後の方策を保ちやすい
- まだ PPO は過更新寄り
- 今後の標準はより保守的にした方がよい

ことを意味する。

### 7.2 条件B（epochs=4）が勝つ場合
これは、

- `lr=0.0001` なら 4 回程度まではまだ有効な更新領域
- 更新量を減らしすぎる必要はない
- 次の探索も `epochs=4` ベースでよい

ことを意味する。

### 7.3 A/B がほぼ同等の場合
この場合は、実務上は **epochs=2 を優先候補**にするのが自然である。

理由:
- より壊れにくい
- learner 時間も短い
- 今の段階では「強くする」より「安定に改善する」を優先したいため

---

## 8. 判定ルール

### 8.1 まず見る順序
1. `Δavg_rank`
2. `Δavg_score`
3. `Δdeal_in_rate`
4. `Δwin_rate`
5. after 指標
6. self-play 統計

### 8.2 標準候補にする条件
以下を満たす方を優先する。

- `Δavg_rank` がより小さい（できれば負）
- `Δavg_score` がより大きい（できれば正）
- `Δdeal_in_rate` がより小さい
- after 指標が大きく劣らない

### 8.3 差が曖昧な場合
- self-play 統計も見て
  - `policy_wins`
  - `policy_deal_ins`
  - `policy_draws`
  のどれが効いているかを確認する
- それでも差が小さければ、**epochs=2 を暫定標準**にして先へ進んでよい

---

## 9. レポートに必ず含める項目

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

## 10. 次のアクション

### 10.1 epochs=2 が有力なら
- `lr=0.0001, epochs=2` を新 baseline 候補にする
- 次は
  - `value_loss_coef`
  - `batch_size`
あたりを見る

### 10.2 epochs=4 が有力なら
- `lr=0.0001, epochs=4` を新 baseline 候補にする
- 次は
  - `value_loss_coef`
  - `batch_size`
  - 必要なら entropy 再調整
を見る

### 10.3 どちらもまだ悪化するなら
- PPO 段の改善はまだ十分ではないので、
  - value 側
  - reward / advantage 周り
  - self-play 統計を見た原因分解
へ進む

---

## 11. メモ

- この実験は、次の長い探索に入る前の「標準PPO設定の決着戦」である
- 今回は after だけでなく、**before→after の delta を主に見る**
- ここで標準が決まると、その後の比較（モデル容量、特徴量、報酬設計など）がかなり安定する

---