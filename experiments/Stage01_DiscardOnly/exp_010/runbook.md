# experiments/exp_010/runbook.md（Runbook 10）

最終更新: 2026-03-07  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **`training.value_loss_coef=0.25` と `0.5` を、phase 再利用ありで厳密に比較し、次の baseline を確定する**

---

## 0. この実験の位置づけ

Runbook 9 で分かったこと:

- `training.lr=0.0001`
- `training.epochs=4`

を固定した上で `value_loss_coef` を比較したところ、

- `value_loss_coef=1.0` は明確に悪い
- `0.25` と `0.5` はかなり近い
- 主評価優先順（`Δavg_rank` → `Δavg_score` → `Δdeal_in_rate`）では **`0.5` が僅差で優位**
- ただし
  - seed=3
  - eval=20
なので、最終判断にはまだ弱い

したがって Runbook 10 では、

- `0.25`
- `0.5`

の **2条件だけ** に絞り、
さらに **imitation / selfplay / eval_before を固定再利用**して、  
learner 条件差だけをより純粋に比較する。

---

## 1. この実験の意図

### 1.1 何を切り分けたいか
今回知りたいのは、

> `value_loss_coef=0.25` と `0.5` の差が、  
> learner の更新則そのものの差として見ても残るか

である。

Runbook 9 では、各条件ごとに
- imitation
- selfplay
- eval_before
までやり直していたため、比較対象としては十分だったが、  
learner 比較としてはまだノイズが混ざりうる。

Runbook 10 では、**各 seed ごとに共通の参照元 run を1本作り**、そこから

- learner + eval（`value_loss_coef=0.25`）
- learner + eval（`value_loss_coef=0.5`）

だけを分岐させる。

これにより、各 seed 内では

- 同じ imitation checkpoint
- 同じ selfplay shard
- 同じ eval_before

を共有したまま、**learner 条件だけを比較**できる。

### 1.2 なぜ再利用を使うのか
今の実験フェーズでは、知りたい差は
- PPO learner の loss バランス
だけである。

このとき
- imitation
- selfplay
- eval_before
を毎回やり直すのは、
- 時間がかかる
- learner 比較としては不要なノイズが増える

ので、再利用機構を使う価値が高い。

### 1.3 この実験で確認したいこと
- `0.25` と `0.5` のどちらが `eval_before -> eval` で良いか
- after の最終指標でも差が残るか
- 差が小さい場合でも、再利用比較ではどちらを baseline に採るべきか

---

## 2. 重要な運用上の注意

### 2.1 `--reuse-from` は `--seeds` と併用不可
現行実装では、

- `--reuse-from`
- `--resume-run`

は **単一 run 専用** であり、`--seeds` とは併用できない。

したがって Runbook 10 は、
**seed ごとに参照元 run を1本ずつ作り、seed ごとに再利用 run を2本ずつ回す**
構成とする。

### 2.2 再利用時の自動整合チェックは限定的
現行実装の再利用は、
- 成果物の存在チェック
中心であり、
- selfplay 条件
- seed
- eval 条件
- learner 条件
の厳密一致を自動で保証しない。

したがって、Runbook 10 では **何を固定し、何だけを変えてよいか** を明示し、それを人間側で守る。

### 2.3 `eval_diff` の確認
再利用時は、`runs[].eval_diff` が存在するだけでなく、  
**主要4指標の delta が `null` でないこと** を確認すること。

---

## 3. 実験の全体構成

各 seed について、次の 3 本を作る。

### 3.1 参照元 run
目的:
- imitation
- selfplay
- eval_before を含む基準 run を作る

### 3.2 比較条件A
参照元 run を再利用して、
- learner
- eval
だけを回す  
条件:
- `training.value_loss_coef=0.25`

### 3.3 比較条件B
参照元 run を再利用して、
- learner
- eval
だけを回す  
条件:
- `training.value_loss_coef=0.5`

各 seed について A/B を比較し、最後に 5 seed 分を集約して判断する。

---

## 4. 比較対象

### 条件A
- `training.value_loss_coef=0.25`

### 条件B
- `training.value_loss_coef=0.5`

差分はこの 1 項目のみ。  
それ以外の learner / selfplay / eval 条件は固定する。

---

## 5. 共通条件

### 5.1 ベース config
- `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`

### 5.2 seeds
- `42,43,44,45,46`

### 5.3 imitation / selfplay / eval / learner 固定
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
- `training.batch_size=256`
- `training.gamma=0.99`
- `training.gae_lambda=0.95`
- `training.entropy_coef=0.01`
- `training.clip_epsilon=0.2`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

### 5.4 rotation eval に関する注記
`evaluation.num_workers=10` を指定しても、rotation 評価では実装上の実効 worker 数は 8 になる。  
ただし全条件で同じなので比較の公平性は保たれる。

---

## 6. 参照元 run の作り方

### 6.1 参照元 run に必要なもの
再利用で必要なのは実質的に以下。

- imitation 後 checkpoint
- selfplay shard
- eval_before 情報

現行実装上、`eval_before` の再利用を安定して使うためには、  
**参照元 run は `eval` まで含めて完了している方が安全**。

### 6.2 参照元 run の phase
- `experiment.phases='["imitation","selfplay","learner","eval"]'`

注意:
- ここでの learner/eval の結果自体は主目的ではない
- 参照可能な成果物一式を整えるために full run を1本作る

---

## 7. 実行コマンド

以下、各 seed ごとに参照元 run を1本作り、その後 A/B を回す。  
`<REF_RUN_DIR_SEED42>` のような部分は、実際に生成された run_dir に置き換える。

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

### 7.1.2 条件A 再利用 run（seed=42, value_loss_coef=0.25）
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

### 7.1.3 条件B 再利用 run（seed=42, value_loss_coef=0.5）
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
    training.value_loss_coef=0.5 \
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

### 7.2.2 条件A 再利用 run（seed=43, value_loss_coef=0.25）
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

### 7.2.3 条件B 再利用 run（seed=43, value_loss_coef=0.5）
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
    training.value_loss_coef=0.5 \
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

## 7.3 seed=44〜46
上と同じ構成で、`experiment.global_seed` と参照元 run_dir だけ `44,45,46` に変えて実行する。

---

## 8. 成功判定

各 seed の各 run について以下を確認する。

### 8.1 参照元 run
- `summary.json.success == true`
- `artifacts_manifest.json` が存在
- `reuse` ではなく通常 full run になっている
- `checkpoint_imitation.pt`
- `selfplay/`
- `eval_before` 相当の情報
が揃っている

### 8.2 再利用 run
- `summary.json.success == true`
- `summary.json.reuse_info.ref_run_dir` が参照元を指している
- `summary.json.reuse_info.reused_phases` に
  - `imitation`
  - `selfplay`
  - `eval_before`
  が入っている
- `phase_action` が `reused/skipped` になっている
- `eval/eval_diff.json` が存在する
- 主要4指標の delta が `null` でない

---

## 9. 集計方法

現行実装では `--reuse-from` と `--seeds` を併用できないため、  
A/B それぞれについて **5 本の単発 run を手で集計**する。

### 9.1 条件A
各 seed の再利用 run 5 本を集めて、
- after 指標
- delta
を mean ± std でまとめる

### 9.2 条件B
同様に 5 本を集める

### 9.3 比較の軸
この実験では、まず delta を見る。

主評価順:
1. `Δavg_rank`
2. `Δavg_score`
3. `Δdeal_in_rate`
4. `Δwin_rate`

その次に after 指標を見る。

---

## 10. どう結果を読むか

### 10.1 条件A（0.25）が勝つ場合
- value の重みを軽くした方が trunk が policy に都合よく働く
- 現 baseline の `0.5` はやや重すぎる可能性
- 次 baseline 候補は `0.25`

### 10.2 条件B（0.5）が勝つ場合
- 現 baseline のバランスがより妥当
- `0.25` まで下げる必要はない
- 次 baseline 候補は `0.5`

### 10.3 差がほぼない場合
- 実務上は `0.5` 維持でもよい
- ただし after / delta / self-play 統計のどれが近いかを明記する
- 差がごく小さいなら、変更コストを考えて `0.5` 維持を選びやすい

---

## 11. self-play 統計の扱い

今回の A/B 差は learner ノブ由来であり、再利用 run では self-play は固定される。  
したがって `summary.json.phase_stats.selfplay` は基本的に同一になるはずである。

### 11.1 確認目的
- 参照元 run が本当に固定されているかの確認
- A/B の差が self-play 分布差ではないことの確認

### 11.2 参考項目
- `policy_wins`
- `policy_deal_ins`
- `policy_draws`
- `tsumo_count`
- `ron_count`
- `ryukyoku_count`
- `num_rounds`

---

## 12. レポートに必ず含める項目

- 参照元 run_dir 一覧（seedごと）
- 条件A 再利用 run_dir 一覧
- 条件B 再利用 run_dir 一覧
- 各 run の success
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
- 主要4指標 delta が `null` でなかったか
- `reuse_info` の確認結果
- self-play 統計（確認用）
- 所要時間
  - 参照元 run の時間
  - 再利用 run の learner/eval 時間
  - 通常版と比べてどれくらい短縮されたか

---

## 13. この実験の副次目的

Runbook 10 には、`value_loss_coef` 比較そのものに加えて、  
**再利用機構が実験運用として十分実用になるか確認する** という副次目的もある。

特に見たいこと:
- 参照元 run の作成 → 再利用 run の流れが素直か
- summary / manifest / log で参照関係が追いやすいか
- 体感として時間短縮が十分大きいか

---

## 14. 次のアクション

### 14.1 `0.25` が勝った場合
- 次 baseline を
  - `lr=0.0001`
  - `epochs=4`
  - `value_loss_coef=0.25`
に更新
- 次は `batch_size` を見る

### 14.2 `0.5` が勝った場合
- 次 baseline を
  - `lr=0.0001`
  - `epochs=4`
  - `value_loss_coef=0.5`
に維持
- 次は `batch_size` を見る

### 14.3 差が小さい場合
- baseline は `0.5` 維持寄り
- ただしレポートで「なぜ差が小さいと見なしたか」を明記する
- その上で次のノブへ進む

---

## 15. メモ

- この Runbook は、今後の再利用版 Runbook の雛形にもなる
- 今のフェーズでは
  - imitation
  - selfplay
  - eval_before
を固定して learner 比較する価値が高い
- 将来の multi-cycle / async actor-learner では再利用価値は相対的に下がるが、
  今はかなり有効

---