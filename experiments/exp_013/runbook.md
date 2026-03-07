# experiments/exp_013/runbook.md（Runbook 13）

最終更新: 2026-03-07  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **暫定 baseline（`lr=0.0001, epochs=4, value_loss_coef=0.25, clip_epsilon=0.2, batch_size=256`）を固定し、`gae_lambda` を本番条件で比較して、advantage 推定のノイズを減らせるか確認する**

---

## 0. この実験の位置づけ

ここまでの実験で、learner 側の主要ノブはかなり整理できている。

現時点の暫定 baseline は以下。

- `training.lr=0.0001`
- `training.epochs=4`
- `training.value_loss_coef=0.25`
- `training.clip_epsilon=0.2`
- `training.batch_size=256`

また、これまでに見えた傾向はかなり一貫している。

- 学習率が高すぎると悪い
- epochs が多すぎると悪い
- value を重くしすぎると悪い
- clip を広げすぎると悪い
- batch を小さくしすぎても大きくしすぎても悪い

つまり、**更新器そのものの強さ・安定性** はかなり詰まってきた。  
それでもなお、PPO は `eval_before -> eval` で **still 少し壊す**。

したがって次は、更新器そのものではなく、

> **advantage / return 側の作り方**

を見る段階に入る。

Runbook 13 では、その第一候補として **`training.gae_lambda`** を比較する。

---

## 1. この実験の意図

### 1.1 `gae_lambda` とは何か
`gae_lambda` は、PPO が advantage を作るときに、

- どれくらい短期寄りにするか
- どれくらい長期寄りにするか

を決めるノブである。

- 低めの `gae_lambda`
  - 直近寄り
  - advantage が比較的安定しやすい
  - ただし先の結果を十分反映しにくい可能性
- 高めの `gae_lambda`
  - 長期寄り
  - 先の結果まで反映しやすい
  - ただし variance が増え、noisy になりやすい可能性

### 1.2 今なぜこれを見るのか
今の主要課題は

> **PPO の更新方向そのものはかなり良くなったが、advantage がまだ少し noisy で、結果として方策を壊している可能性**

である。

この場合、`gae_lambda` を調整すると

- advantage のノイズが減る
- imitation 後方策の良さを壊しにくくなる
- `eval_before -> eval` の悪化が縮まる

可能性がある。

### 1.3 何を切り分けたいか
- `gae_lambda=0.95` は長期寄りすぎて、今のタスクでは noisy なのか
- 逆に、もっと高い方がうまく credit assignment できるのか
- あるいは現状 0.95 が妥当で、問題は別の場所にあるのか

---

## 2. 比較したい仮説

### 仮説A
`gae_lambda=0.90` に下げると、
- advantage が短期寄りになって安定し
- PPO の悪化が減る
可能性がある

### 仮説B
`gae_lambda=0.95`（現行）はすでに妥当な中間点であり、
- これ以上振っても改善しない
可能性がある

### 仮説C
`gae_lambda=0.98` に上げると、
- もっと先の結果まで反映され
- policy 改善に必要な credit が伝わりやすくなる
可能性がある  
ただし、今までの流れからは noisy になって悪化する可能性もある

---

## 3. 実験方針

### 3.1 比較対象
- A: `training.gae_lambda=0.90`
- B: `training.gae_lambda=0.95`（baseline）
- C: `training.gae_lambda=0.98`

### 3.2 固定するもの
今回の比較では、`gae_lambda` 以外は固定し、**1要因比較**にする。

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
- `training.batch_size=256`
- `training.gamma=0.99`
- `training.entropy_coef=0.01`
- `training.clip_epsilon=0.2`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

### 3.3 実行方式
今回も **再利用前提** で行う。

各 seed ごとに、
1. 参照元 run を 1 本作る
2. その run の
   - `imitation`
   - `selfplay`
   - `eval_before`
   を再利用して
3. learner/eval だけを A/B/C に分岐して実行する

これにより、
- learner 条件差だけを比較しやすくする
- 実行時間を削減する
- 同一 self-play データ上で learner を比較できる

---

## 4. 実行規模

今回は、前回の `batch_size` 実験と同等以上の判断力を持たせる。

- `5 seeds`
- `rotation eval`
- `evaluation.num_matches=50`
- `reuse` 前提

これは、寝ている間に流す前提の **本番寄り比較** とする。

---

## 5. 再利用運用上の重要事項

### 5.1 `--reuse-from` は `--seeds` と併用不可
現行実装では `--reuse-from` は **単一 run 専用**。  
したがって本 runbook も、**seed ごとに単発実行**する。

### 5.2 整合性チェックは存在確認中心
現行実装は、再利用時に成果物の存在は見るが、  
設定値一致を厳密には強制しない。

したがって、**この Runbook で固定と明示した条件を崩さないこと** が前提である。

### 5.3 `eval_diff` の確認
再利用 run では、
- `eval/eval_diff.json` が存在する
- 主要4指標の delta が `null` でない
ことを必ず確認する。

---

## 6. 参照元 run の役割

各 seed ごとに 1 本、参照元 run を作る。  
参照元 run は、後続の A/B/C 比較に対して

- imitation checkpoint
- selfplay shard
- eval_before 情報

を提供する。

### 6.1 参照元 run の phase
- `experiment.phases='["imitation","selfplay","learner","eval"]'`

注意:
- 参照元 run の learner/eval 自体は主比較ではない
- 再利用成立に必要な成果物を揃えるために full run を作る

---

## 7. 実行コマンド

以下、seed=42 の完全形を示す。  
seed=43〜46 は、`experiment.global_seed` と `runs/<REF_RUN_DIR_SEEDxx>` の部分だけ変えて同様に実行する。

---

## 7.1 seed=42

### 7.1.1 参照元 run（seed=42）
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

### 7.1.2 条件A 再利用 run（seed=42, gae_lambda=0.90）
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
    training.gae_lambda=0.90 \
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

### 7.1.3 条件B 再利用 run（seed=42, gae_lambda=0.95）
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

### 7.1.4 条件C 再利用 run（seed=42, gae_lambda=0.98）
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
    training.gae_lambda=0.98 \
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

## 7.2 seed=43

### 7.2.1 参照元 run（seed=43）
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

### 7.2.2 条件A 再利用 run（seed=43, gae_lambda=0.90）
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
    training.gae_lambda=0.90 \
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

### 7.2.3 条件B 再利用 run（seed=43, gae_lambda=0.95）
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

### 7.2.4 条件C 再利用 run（seed=43, gae_lambda=0.98）
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
    training.gae_lambda=0.98 \
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

## 7.3 seed=44〜46
上と同じ構成で、`experiment.global_seed` と参照元 run_dir を `44,45,46` に変えて実行する。  
ローカルドライバを使う場合は、seed ごとに
- ref
- A
- B
- C
の 4 run を自動実行し、`run_map.json` を更新する構成を推奨する。

---

## 8. 成功判定

### 8.1 参照元 run
各 seed の参照元 run について、以下を確認する。

- `summary.json.success == true`
- `artifacts_manifest.json` が存在する
- `checkpoint_imitation.pt` が存在する
- `selfplay/` が存在する
- `eval_before` 再利用に必要な情報が揃っている

### 8.2 再利用 run
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

## 9. 集計方法

現行実装では `--reuse-from` と `--seeds` を併用できないため、  
条件 A/B/C それぞれについて **5 本の単発 run を集約**する。

### 条件A
- `gae_lambda=0.90`
- 5 seed の mean ± std を集計

### 条件B
- `gae_lambda=0.95`
- 5 seed の mean ± std を集計

### 条件C
- `gae_lambda=0.98`
- 5 seed の mean ± std を集計

---

## 10. 主な評価項目

### 10.1 最優先（delta）
今回も主評価は `eval_before -> eval` の差分。

見る順序:
1. `Δavg_rank`
2. `Δavg_score`
3. `Δdeal_in_rate`
4. `Δwin_rate`

### 10.2 次点（after）
そのうえで、最終到達点を見る。

- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 10.3 補助（再利用成立確認）
- `reuse_info`
- `phase_action`
- `run_map.json`
- `eval_diff` 欠落がないこと

### 10.4 補助（self-play 統計）
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

## 11. 結果の読み方

### 11.1 条件A（0.90）が勝つ場合
- 現在の advantage は長期寄りすぎて noisy
- 少し短期寄りにした方が PPO が安定する
- 次 baseline 候補は `gae_lambda=0.90`

### 11.2 条件B（0.95）が勝つ場合
- 現在の `gae_lambda` は妥当
- 問題は別の場所にある可能性が高い
- 次 baseline 候補は `gae_lambda=0.95`

### 11.3 条件C（0.98）が勝つ場合
- もっと先の結果まで見た方が advantage がうまく働く
- 今は少し短期寄りすぎた可能性
- 次 baseline 候補は `gae_lambda=0.98`

---

## 12. 判定ルール

### 12.1 最優先
事前に定めた順で判定する。

1. `Δavg_rank`
2. `Δavg_score`
3. `Δdeal_in_rate`
4. `Δwin_rate`

### 12.2 baseline 採用条件
- `Δavg_rank` が最も小さい（できれば負）
- `Δavg_score` が最も大きい（できれば正）
- `Δdeal_in_rate` が最も小さい
- after 指標が大きく悪化していない

### 12.3 差が僅差の場合
差が小さい場合は、
- after 指標
- 実行時間
- 運用の安定性
も補助的に見る

ただし、まずは主評価順を優先する。

---

## 13. レポートに必ず含める項目

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

## 14. この実験の副次目的

この runbook には、`gae_lambda` 比較に加えて

> **target / advantage 側のノブに対しても、再利用前提の learner 比較が安定して回せるか**

を確認する副次目的がある。

確認したい点:
- ローカルドライバ運用で、寝ている間でも最後まで流せるか
- `run_map.json` / `summary.json` / `report.md` の連携が崩れないか
- 実験時間に対して再利用の恩恵が十分大きいか

---

## 15. 次のアクション

### 15.1 0.90 が勝った場合
- baseline を
  - `lr=0.0001`
  - `epochs=4`
  - `value_loss_coef=0.25`
  - `clip_epsilon=0.2`
  - `batch_size=256`
  - `gae_lambda=0.90`
に更新
- 次は `gamma` か、特徴量強化（シャンテン特徴量）を検討する

### 15.2 0.95 が勝った場合
- baseline は `gae_lambda=0.95` を維持
- 次は `gamma` か、特徴量強化（シャンテン特徴量）を検討する

### 15.3 0.98 が勝った場合
- baseline 更新候補
- 長期 credit assignment が効いている可能性がある
- 次は `gamma` か reward / return 側の解釈を進める

### 15.4 どれも僅差の場合
- 主評価順に従って暫定採用
- 以後の runbook で cross-check する
- あるいは learner ノブ追求をいったん切り上げ、特徴量強化へ進む

---

## 16. メモ

- 今回は、更新器のノブではなく **advantage の作り方** に踏み込む実験である
- ここで改善が出るなら、PPO の “still 少し壊す” の原因は target 側にあった可能性が高い
- ここでも決め手が出ないなら、次は
  - `gamma`
  - reward / return 設計
  - シャンテン特徴量
を本格候補として扱う

---