# experiments/exp_009/runbook.md（Runbook 9）

最終更新: 2026-03-06  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **`lr=0.0001, epochs=4` を固定したまま `value_loss_coef` を探索し、shared trunk を通じて PPO がより安定する値を探す**

---

## 0. この実験の位置づけ

ここまでの実験で分かったこと:

1. **軽量 warm start（E）は PPO 直学習より良い**
2. ただし PPO 段は imitation 後の方策を壊しがちだった
3. learning rate を下げると悪化幅が減った
4. `training.lr=0.0001` は有望だった
5. low-lr のまま epochs を振ると
   - `epochs=8` は悪い
   - `epochs=2` と `epochs=4` は良い帯
6. 5 seed / rotation eval 50 の比較で、**`epochs=4` が `epochs=2` より良かった**
7. よって、現時点の PPO baseline 候補は
   - **`training.lr=0.0001`**
   - **`training.epochs=4`**
   である

Runbook 9 では、この baseline を固定し、次の候補として  
**`value_loss_coef`** を探索する。

---

## 1. この実験の意図

### 1.1 何を知りたいのか
今のモデルは、policy/value が **shared trunk** を完全共有している。

構造の要点:
- encoder: `FlatFeatureEncoder`
- model: `MLPPolicyValueModel`
- trunk: `Linear -> ReLU -> Linear -> ReLU`
- その後で
  - `policy_head`
  - `value_head(round_delta)`
  に分岐する

このため、`value_loss_coef` を変えると、
- value head の fitting 強度
だけでなく、
- **shared trunk に流れる value 側勾配の強さ**
も変わる。

つまりこの実験は、単なる「value をどれだけ当てるか」ではなく、

> **value の重みを変えたとき、shared trunk の表現が安定し、  
> policy 学習が改善するか**
> それとも
> **value に引っ張られすぎて policy が悪くなるか**

を調べる実験である。

### 1.2 なぜ今これを見るのか
- `lr=0.0001, epochs=4` で PPO はかなり改善した
- それでも `eval_before -> eval` の delta はまだ平均で少し悪化寄り
- 次の候補としては
  - policy 更新量そのものをさらにいじる
より前に、
  - **value の比重を見直す**
のが自然

理由:
- trunk 共有構造では value 側の重みが policy 側の表現学習にも影響する
- したがって、`value_loss_coef` はかなり意味のあるノブである

---

## 2. この実験で切り分けたい仮説

### 仮説A
`value_loss_coef` が小さすぎると、value 推定が弱くなり、
- return/advantage 推定が不安定
- PPO 更新が noisy
になって policy を悪化させる

### 仮説B
`value_loss_coef` が大きすぎると、value fitting が強すぎて
- trunk が value 寄りに引っ張られる
- policy に必要な表現が圧迫される
- PPO 更新が悪化する

### 仮説C
今の `0.5` 近辺、またはその前後に
**実用的な最適域** がある

---

## 3. 基本方針

### 3.1 固定するもの
今回の実験では、`value_loss_coef` 以外は固定し、**1要因比較**にする。

固定:
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
- `evaluation.num_matches=20`
- `evaluation.num_workers=10`
- `training.lr=0.0001`
- `training.epochs=4`
- `training.batch_size=256`（既定のまま固定）
- `training.gamma=0.99`（固定）
- `training.gae_lambda=0.95`（固定）
- `training.entropy_coef=0.01`（固定）
- `training.clip_epsilon=0.2`（固定）
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

### 3.2 なぜ `batch_size`, `gamma`, `gae_lambda` も固定するのか
Codex 確認の通り、
- `batch_size` は実質ミニバッチサイズ
- `gamma`, `gae_lambda` は target/advantage 計算に効く
ので、これらが動くと `value_loss_coef` の解釈が崩れるため。

### 3.3 評価予算
今回はまず **小規模探索** とする。

- seeds: `42,43,44`
- `evaluation.num_matches=20`

理由:
- まず傾向を見る段階
- 良い候補だけ次段で 5 seed / 50 matches に拡張するため

---

## 4. 比較条件

### 条件A: 低め
- `training.value_loss_coef=0.25`

### 条件B: baseline
- `training.value_loss_coef=0.5`

### 条件C: 高め
- `training.value_loss_coef=1.0`

---

## 5. この3条件を選ぶ理由

- `0.5` は現 baseline 候補
- `0.25` は value を軽くして、policy 側の比重を相対的に上げる方向
- `1.0` は value を強くして、trunk の安定化が効くかを見る方向

まずはこの3点で十分。  
ここで傾向が見えたら、次段で必要に応じて
- `0.125`
- `0.75`
などを足せばよい。

---

## 6. 実行コマンド

### 6.1 条件A: value_loss_coef=0.25
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

### 6.2 条件B: value_loss_coef=0.5（baseline）
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
    training.lr=0.0001 \
    training.epochs=4 \
    training.value_loss_coef=0.5 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 6.3 条件C: value_loss_coef=1.0
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
    training.lr=0.0001 \
    training.epochs=4 \
    training.value_loss_coef=1.0 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

---

## 7. 成功判定

各条件の batch について以下を確認する。

- `success_count == 3/3`
- `aggregate.eval_mode == "rotation"`
- 各 run の `runs[].eval_diff` が存在する
  - これは `eval_before -> eval` の差分が計算済みであることを意味する
- 必要なら詳細確認として、各 `run_dir` の `eval/` 配下と `run.log` を参照する

---

## 8. 主な評価項目

### 8.1 最優先（delta）
この実験でも主評価は `eval_before -> eval` の差分。

見る指標:
- `Δavg_rank`
- `Δavg_score`
- `Δwin_rate`
- `Δdeal_in_rate`

まずは、
- 悪化を減らせるか
- 改善方向に押せるか
を確認する。

### 8.2 次点（after）
そのうえで、最終性能を見る。

- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 8.3 補助（self-play 統計）
各 run の `summary.json.phase_stats.selfplay` から以下も確認する。

- `policy_wins`
- `policy_deal_ins`
- `policy_draws`
- `tsumo_count`
- `ron_count`
- `ryukyoku_count`

必要なら `selfplay/worker_*/round_results.jsonl` を見て補助解釈する。

---

## 9. 結果の読み方

### 9.1 条件A（0.25）が良い場合
- value 側の重みが強すぎた可能性
- trunk が value fitting に引っ張られすぎていた可能性
- policy 寄りにした方が良い

### 9.2 条件B（0.5）が良い場合
- 現 baseline のバランスが妥当
- 次は別のノブを見る段階

### 9.3 条件C（1.0）が良い場合
- value をしっかり当てることで advantage が安定し、
  trunk の表現も改善している可能性
- value の弱さがボトルネックだった可能性

---

## 10. 判定ルール

### 10.1 まず見る順序
1. `Δavg_rank`
2. `Δavg_score`
3. `Δdeal_in_rate`
4. `Δwin_rate`
5. after 指標
6. self-play 統計

### 10.2 次段（5 seed / 50 matches）へ進める候補
以下を満たす条件を有望候補とする。

- `Δavg_rank` が最も小さい（できれば負）
- `Δavg_score` が最も大きい（できれば正）
- `Δdeal_in_rate` が最も小さい
- after 指標が大きく悪化していない

### 10.3 差が小さい場合
差が小さいなら、勝者候補を 1〜2 条件に絞って

- seeds: `42,43,44,45,46`
- `evaluation.num_matches=50`

で再確認する。

---

## 11. レポートに必ず含める項目

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

## 12. 注意点

### 12.1 checkpoint 単体では条件復元できない
Codex 確認の通り、checkpoint 本体には
- config
- encoder 情報
- value head 名
などは保存されていない。

再利用時は必ず
- `config.yaml`
- `summary.json`
- `artifacts_manifest.json`
とセットで扱うこと。

### 12.2 これは「value head 単体」の実験ではない
shared trunk 構造のため、`value_loss_coef` は
- value head
- shared trunk
に効き、policy 品質にも間接影響する。

したがってこの実験は、
**shared trunk を通じた trunk 学習のバランス実験**
として読むのが正しい。

### 12.3 今回は still small-scale
- seeds=3
- eval=20
なので、これはあくまで探索段階である。
良い候補が見えたら、次段で 5 seed / 50 matches に拡張する。

---

## 13. 次のアクション

### 13.1 明確な勝者が出た場合
- その条件を baseline 候補として固定
- 次は `batch_size` を探索する

### 13.2 差が曖昧な場合
- 1〜2条件に絞って 5 seed / eval 50 へ拡張

### 13.3 どれも改善しない場合
- value_loss だけでは十分でない
- 次は
  - `batch_size`
  - `gamma` / `gae_lambda`
  - reward / advantage 解釈
を検討する

---

## 14. メモ

- この実験の本質は「value を当てること」ではなく、
  **value の重みを通じて PPO 全体の安定性を改善できるか**
  を見ることにある
- 今の構造では policy/value が trunk を共有しているので、
  `value_loss_coef` の意味は大きい
- ここで baseline がまた1段固まれば、次のモデル比較や特徴量比較の土台がより安定する

---