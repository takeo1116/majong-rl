# experiments/exp_012/runbook.md（Runbook 12）

最終更新: 2026-03-07  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **暫定 baseline（`lr=0.0001, epochs=4, value_loss_coef=0.25, clip_epsilon=0.2`）を固定し、`batch_size` を本番条件で比較して、PPO の更新安定性をさらに改善できるか確認する**

---

## 0. この実験の位置づけ

ここまでの実験で分かったこと:

1. 軽量 warm start は PPO 直学習より良い
2. ただし PPO 段は imitation 後の良さを still 少し壊す
3. `training.lr=0.0001` は有望
4. `training.epochs=4` は `2` より良く、`8` は悪い
5. `training.value_loss_coef=1.0` は悪く、`0.25` と `0.5` は僅差
6. 再利用比較の主評価順に従うと、`value_loss_coef=0.25` を暫定採用するのが妥当
7. `clip_epsilon=0.3` は悪く、`0.1` と `0.2` は僅差
8. `clip_epsilon=0.1` を正式採用するほどの差はまだなく、**現時点では `0.2` 維持が最も安定**

したがって、現時点の暫定 baseline は

- `training.lr=0.0001`
- `training.epochs=4`
- `training.value_loss_coef=0.25`
- `training.clip_epsilon=0.2`

である。

Runbook 12 では、この baseline を固定し、次の learner ノブとして  
**`training.batch_size`** を比較する。

---

## 1. この実験の意図

### 1.1 何を知りたいのか
ここまでの流れで、PPO が壊れやすい方向はかなり絞れてきた。

- 学習率が高すぎるのは悪い
- epochs が多すぎるのは悪い
- value を重くしすぎるのは悪い
- clip を広げすぎるのは悪い

つまり、今の主要課題は

> **更新の平均的な方向はかなり良くなったが、更新のばらつきやノイズがまだ残っている可能性**

である。

`batch_size` は、PPO 更新における **勾配推定のノイズ量と更新安定性** に効く主要ノブなので、  
今の段階で最も自然に見るべき候補である。

### 1.2 何を切り分けたいか
- `batch_size` が小さいと更新が noisy すぎて、still 少し壊しているのか
- `batch_size` が大きいと更新が安定し、delta が改善するのか
- 逆に大きすぎると更新が鈍くなり、改善量が落ちるのか

### 1.3 なぜ今これを見るのか
- `lr`
- `epochs`
- `value_loss_coef`
- `clip_epsilon`

の主要ノブは、少なくとも実用域が見えてきた
- 次は **更新の質（安定性）** をもう一段整えるフェーズ
- `batch_size` はその観点で最も素直に効く

### 1.4 なぜ今夜これをやるのか
今回は、寝ている間に回せる前提なので、

- 3条件
- 5 seeds
- rotation eval 50
- reuse 前提

の **判断力の高い本番寄り比較** を最初から回す。

---

## 2. 比較したい仮説

### 仮説A
`batch_size=128` のように小さくすると、
- 更新が noisy になりやすい
- そのノイズが PPO の悪化要因になる可能性がある
- 一方で、場合によっては改善の勢いが出る可能性もある

### 仮説B
`batch_size=256` は、今の baseline として十分妥当な中間点である

### 仮説C
`batch_size=512` のように大きくすると、
- 勾配推定が安定し
- `eval_before -> eval` の悪化が減る
可能性がある  
ただし、大きすぎると更新が鈍くなり、改善しにくくなる可能性もある

---

## 3. 実験方針

### 3.1 比較対象
- A: `training.batch_size=128`
- B: `training.batch_size=256`（baseline）
- C: `training.batch_size=512`

### 3.2 固定するもの
今回の比較では、`batch_size` 以外は固定し、**1要因比較**にする。

固定:
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
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
- `training.epochs=4`
- `training.value_loss_coef=0.25`
- `training.gamma=0.99`
- `training.gae_lambda=0.95`
- `training.entropy_coef=0.01`
- `training.clip_epsilon=0.2`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

### 3.3 実行方式
今回は **再利用前提** で行う。

各 seed ごとに、
1. 参照元 run を 1 本作る
2. その run の
   - `imitation`
   - `selfplay`
   - `eval_before`
   を再利用して
3. learner/eval だけを条件 A/B/C に分岐して実行する

これにより、
- learner 条件差だけを比較しやすくする
- 実行時間を削減する
- 同一 self-play データ上で learner を比較できる

---

## 4. 再利用運用上の重要事項

### 4.1 `--reuse-from` は `--seeds` と併用不可
現行実装では `--reuse-from` は **単一 run 専用**。  
したがって本 runbook も、**seed ごとに単発実行**する。

### 4.2 整合性チェックは存在確認中心
現行実装は、再利用時に成果物の存在は見るが、  
設定値一致を厳密には強制しない。

したがって、**この Runbook で固定と明示した条件を人間側で崩さないこと** が前提である。

### 4.3 `eval_diff` の確認
再利用 run では、
- `eval/eval_diff.json` が存在する
- 主要4指標の delta が `null` でない
ことを必ず確認する。

---

## 5. 参照元 run の役割

各 seed ごとに 1 本、参照元 run を作る。  
参照元 run は、後続の A/B/C 比較に対して

- imitation checkpoint
- selfplay shard
- eval_before 情報

を提供する。

### 5.1 参照元 run の phase
- `experiment.phases='["imitation","selfplay","learner","eval"]'`

注意:
- 参照元 run の learner/eval 自体は主比較ではない
- 再利用成立に必要な成果物を揃えるために full run を作る

---

## 6. 実行コマンド

以下、seed=42 の完全形を示す。  
seed=43〜46 は、`experiment.global_seed` と `runs/<REF_RUN_DIR_SEEDxx>` の部分だけ変えて同様に実行する。

---

## 6.1 seed=42

### 6.1.1 参照元 run（seed=42）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    experiment.phases='["imitation","selfplay","learner","eval"]' \
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

### 6.1.2 条件A 再利用 run（seed=42, batch_size=128）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --reuse-from runs/<REF_RUN_DIR_SEED42> \
  --reuse-phases imitation,selfplay,eval_before \
  --override \
    experiment.global_seed=42 \
    experiment.phases='["learner","eval"]' \
    training.lr=0.0001 \
    training.epochs=4 \
    training.value_loss_coef=0.25 \
    training.batch_size=128 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 6.1.3 条件B 再利用 run（seed=42, batch_size=256）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --reuse-from runs/<REF_RUN_DIR_SEED42> \
  --reuse-phases imitation,selfplay,eval_before \
  --override \
    experiment.global_seed=42 \
    experiment.phases='["learner","eval"]' \
    training.lr=0.0001 \
    training.epochs=4 \
    training.value_loss_coef=0.25 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 6.1.4 条件C 再利用 run（seed=42, batch_size=512）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --reuse-from runs/<REF_RUN_DIR_SEED42> \
  --reuse-phases imitation,selfplay,eval_before \
  --override \
    experiment.global_seed=42 \
    experiment.phases='["learner","eval"]' \
    training.lr=0.0001 \
    training.epochs=4 \
    training.value_loss_coef=0.25 \
    training.batch_size=512 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

---

## 6.2 seed=43

### 6.2.1 参照元 run（seed=43）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=43 \
    experiment.phases='["imitation","selfplay","learner","eval"]' \
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

### 6.2.2 条件A 再利用 run（seed=43, batch_size=128）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --reuse-from runs/<REF_RUN_DIR_SEED43> \
  --reuse-phases imitation,selfplay,eval_before \
  --override \
    experiment.global_seed=43 \
    experiment.phases='["learner","eval"]' \
    training.lr=0.0001 \
    training.epochs=4 \
    training.value_loss_coef=0.25 \
    training.batch_size=128 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 6.2.3 条件B 再利用 run（seed=43, batch_size=256）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --reuse-from runs/<REF_RUN_DIR_SEED43> \
  --reuse-phases imitation,selfplay,eval_before \
  --override \
    experiment.global_seed=43 \
    experiment.phases='["learner","eval"]' \
    training.lr=0.0001 \
    training.epochs=4 \
    training.value_loss_coef=0.25 \
    training.batch_size=256 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 6.2.4 条件C 再利用 run（seed=43, batch_size=512）
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --reuse-from runs/<REF_RUN_DIR_SEED43> \
  --reuse-phases imitation,selfplay,eval_before \
  --override \
    experiment.global_seed=43 \
    experiment.phases='["learner","eval"]' \
    training.lr=0.0001 \
    training.epochs=4 \
    training.value_loss_coef=0.25 \
    training.batch_size=512 \
    training.gamma=0.99 \
    training.gae_lambda=0.95 \
    training.entropy_coef=0.01 \
    training.clip_epsilon=0.2 \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

---

## 6.3 seed=44〜46
上と同じ構成で、`experiment.global_seed` と参照元 run_dir を `44,45,46` に変えて実行する。  
ローカルドライバを使う場合は、seed ごとに
- ref
- A
- B
- C
の 4 run を自動実行し、`run_map.json` を更新する構成を推奨する。

---

## 7. 成功判定

### 7.1 参照元 run
各 seed の参照元 run について、以下を確認する。

- `summary.json.success == true`
- `artifacts_manifest.json` が存在する
- `checkpoint_imitation.pt` が存在する
- `selfplay/` が存在する
- `eval_before` 再利用に必要な情報が揃っている

### 7.2 再利用 run
A/B/C 全 run について、以下を確認する。

- `summary.json.success == true`
- `summary.json.reuse_info.ref_run_dir` が正しい参照元を指している
- `summary.json.reuse_info.reused_phases` に
  - `imitation`
  - `selfplay`
  - `eval_before`
  が入っている
- `phase_action` では
  - `imitation`
  - `selfplay`
  - `eval_before`
  が `reused` または `skipped` になっていることを確認する
- `phase_timing.learner` と `phase_timing.eval` が存在する
- `eval/eval_diff.json` が存在する
- `eval_diff` の主要4指標
  - `avg_rank`
  - `avg_score`
  - `win_rate`
  - `deal_in_rate`
  の delta がすべて非 `null` である

---

## 8. 集計方法

現行実装では `--reuse-from` と `--seeds` を併用できないため、  
条件 A/B/C それぞれについて **5 本の単発 run を集約**する。

### 条件A
- `batch_size=128`
- 5 seed の mean ± std を集計

### 条件B
- `batch_size=256`
- 5 seed の mean ± std を集計

### 条件C
- `batch_size=512`
- 5 seed の mean ± std を集計

---

## 9. 主な評価項目

### 9.1 最優先（delta）
今回も主評価は `eval_before -> eval` の差分。

見る順序:
1. `Δavg_rank`
2. `Δavg_score`
3. `Δdeal_in_rate`
4. `Δwin_rate`

### 9.2 次点（after）
そのうえで、最終到達点を見る。

- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 9.3 補助（再利用成立確認）
- `reuse_info`
- `phase_action`
- `run_map.json`
- `eval_diff` 欠落がないこと

### 9.4 補助（self-play 統計）
self-play は固定再利用なので、本来大差はないはず。  
差が出たら再利用不整合を疑う。

見る項目:
- `policy_wins`
- `policy_deal_ins`
- `policy_draws`
- `tsumo_count`
- `ron_count`
- `ryukyoku_count`
- `num_rounds`

---

## 10. 結果の読み方

### 10.1 条件A（128）が勝つ場合
- まだ更新が鈍く、より小さい batch の方が改善しやすい
- 現 baseline は少し安定寄りすぎる可能性
- 次 baseline 候補は `batch_size=128`

### 10.2 条件B（256）が勝つ場合
- 現在の batch_size は妥当
- 他ノブを見に行く段階
- 次 baseline 候補は `batch_size=256`

### 10.3 条件C（512）が勝つ場合
- より安定した更新が必要だった
- まだ勾配ノイズが悪化要因だった可能性が高い
- 次 baseline 候補は `batch_size=512`

---

## 11. 判定ルール

### 11.1 最優先
事前に定めた順で判定する。

1. `Δavg_rank`
2. `Δavg_score`
3. `Δdeal_in_rate`
4. `Δwin_rate`

### 11.2 baseline 採用条件
- `Δavg_rank` が最も小さい（できれば負）
- `Δavg_score` が最も大きい（できれば正）
- `Δdeal_in_rate` が最も小さい
- after 指標が大きく悪化していない

### 11.3 差が僅差の場合
差が小さい場合は、
- after 指標
- 実行時間
- 運用の安定性
も補助的に見る

ただし、まずは主評価順を優先する。

---

## 12. レポートに必ず含める項目

- 参照元 run_dir 一覧（seed ごと）
- 条件A/B/C の run_dir 一覧
- 各 run の success
- `reuse_info` / `phase_action` 確認結果
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
- `eval_diff` の 4 指標 delta が非 `null` だったこと
- self-play 統計（確認用）
- 所要時間
  - ref run
  - reuse run
  - 全体
- 可能なら通常版相当と比べた時間短縮の感触

---

## 13. この実験の副次目的

この runbook には、`batch_size` 比較に加えて

> **再利用前提の learner 比較が、数時間規模の本番実験でも安定運用できるか**

を確認する副次目的がある。

確認したい点:
- ローカルドライバ運用で、寝ている間や外出中でも最後まで流せるか
- `run_map.json` / `summary.json` / `report.md` の連携が崩れないか
- 実験時間に対して再利用の恩恵が十分大きいか

---

## 14. 次のアクション

### 14.1 128 が勝った場合
- baseline を
  - `lr=0.0001`
  - `epochs=4`
  - `value_loss_coef=0.25`
  - `clip_epsilon=0.2`
  - `batch_size=128`
に更新
- 次は `clip_epsilon=0.1 vs 0.2` 再確認、または `gamma/gae_lambda` を見る

### 14.2 256 が勝った場合
- baseline は `batch_size=256` を維持
- 次は `gamma/gae_lambda` か、必要なら `clip_epsilon=0.1 vs 0.2` 再確認

### 14.3 512 が勝った場合
- baseline 更新候補
- まだ勾配ノイズが主因だったと解釈する
- 次は `gamma/gae_lambda` か reward / advantage 側を確認する

### 14.4 どれも僅差の場合
- 主評価順に従って暫定採用
- 以後の runbook で cross-check する

---

## 15. メモ

- 今回は「数時間かかってもよい」前提なので、最初から判断力の高い条件で回す
- learner ノブ比較は、今のフェーズでは再利用の価値が高い
- この runbook は、今後の本番寄り reuse 比較の基本形としても使える

---