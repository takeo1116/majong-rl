# experiments/exp_017/runbook.md（Runbook 17）

最終更新: 2026-03-08  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **`shanten_hint` の効き方を、教師再現・極小過学習・strict vs tie-aware・warm start + PPO までを一連で診断し、問題が「特徴量」なのか「学習目標」なのか「downstream interaction」なのかを切り分ける**

---

## 0. この実験の位置づけ

ここまでの結果は以下の通り。

### 0.1 既存 baseline
現時点の learner baseline は以下。

- `training.lr=0.0001`
- `training.epochs=4`
- `training.value_loss_coef=0.25`
- `training.clip_epsilon=0.2`
- `training.batch_size=256`
- `training.gae_lambda=0.95`

### 0.2 `shanten_hint` の結果
- imitation-only 相当（exp_014）では、`shanten_hint=on` は
  - `avg_rank` を小幅改善
  - `avg_score` を小幅改善
  - ただし `win_rate` は微悪化
- warm start + PPO（exp_015）では、`shanten_hint=on` は
  - delta でも after でも総合的に悪化
  - 現形では採用見送り

### 0.3 教師再現度診断（exp_016）
- `shanten_hint=on` で
  - `teacher_top1_match_rate`
  - `teacher_best_set_hit_rate`
  は小幅ながら一貫して改善
- よって
  - 特徴量が全く使われていない
  - encoder から model に渡っていない
  - imitation が全然学べていない
  という線は薄くなった
- 一方で、最終性能にはつながっていないため、
  **PPO interaction / shortcut 性 / strict top-1 学習目標の限界**
  が本命論点となった

### 0.4 今回の新要素
- discard imitation に
  - `strict_top1`
  - `tie_aware_best_set`
  の loss mode 切替が入った
- teacher 指標と loss mode が run / batch 成果物から追えるようになった

したがって次は、これらを**別々の小実験ではなく、一連の診断 runbook**としてまとめて回すのが合理的である。

---

## 1. この runbook の目的

この runbook は、次の大きな問いに答えるためのもの。

> **`shanten_hint` はなぜ imitation では少し効くのに、warm start + PPO では悪化するのか？**

これを以下の順で切り分ける。

### Part 1
`shanten_hint` on/off で、教師再現度にどの程度差があるか再確認する

### Part 2
極小データなら、`shanten_hint=on` で teacher を十分に覚え切れるか確認する  
→ ここで弱いなら、学習系の健全性を疑う

### Part 3
通常 imitation-only 相当で、`strict_top1` と `tie_aware_best_set` を比較する  
→ strict top-1 CE の限界があるか見る

### Part 4
strict と tie-aware を warm start + PPO で再比較する  
→ 学習目標変更が downstream でも効くか確認する

---

## 2. 実験全体の方針

### 2.1 この runbook は 4 部構成
- **Part 1**: `shanten_hint` on/off の教師再現率再確認
- **Part 2**: 極小データ過学習 sanity check
- **Part 3**: imitation-only 相当で strict vs tie-aware 比較
- **Part 4**: strict vs tie-aware を warm start + PPO で比較

### 2.2 実行順
Runbook 内では以下の順で進める。

1. Part 1
2. Part 2
3. Part 3
4. Part 4

### 2.3 実行コスト最適化
- Part 3 の `A3` は **Part 1 の `B1` を再利用**する
- よって Part 3 で新規実行するのは `B3` のみ
- Part 4 は strict / tie-aware の両条件を実行し、PPO まで含めた比較を行う

---

## 3. 共通の評価軸

### 3.1 教師再現度
- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`

### 3.2 imitation-only の after 指標
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 3.3 PPO を含む場合の主評価
- `Δavg_rank`
- `Δavg_score`
- `Δdeal_in_rate`
- `Δwin_rate`

### 3.4 補助
- imitation loss
- total_steps
- 実行時間
- `encoder_features.shanten_hint`
- `encoder_features.input_dim`
- `phase_stats.imitation.imitation_loss_mode`

---

## 4. Part 1: `shanten_hint` on/off の教師再現率再確認

### 4.1 目的
- `shanten_hint=on` が教師再現度を本当に押し上げるか再確認する
- tie-aware 比較の土台として最新実装下で再確認する

### 4.2 条件
- A1: `shanten_hint=off`, `imitation_loss_mode=strict_top1`
- B1: `shanten_hint=on`, `imitation_loss_mode=strict_top1`

### 4.3 実行方式
imitation-only 相当の最短経路を使う。

- `experiment.phases='["imitation","selfplay","eval"]'`
- `selfplay.num_matches=0`

### 4.4 共通条件
- seeds: `42,43,44,45,46`
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

### 4.5 コマンド

#### A1: shanten hint off / strict
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
    training.imitation_loss_mode=strict_top1 \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

#### B1: shanten hint on / strict
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
    training.imitation_loss_mode=strict_top1 \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 4.6 主判定
- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`

### 4.7 読み方
- B1 が teacher 指標で改善するなら、`shanten_hint` は教師再現に効いている
- 改善しないなら、上流の健全性を再度疑う

---

## 5. Part 2: 極小データ過学習 sanity check

### 5.1 目的
- `shanten_hint=on` かつ小データで、teacher をどこまで覚え切れるか見る
- strict / tie-aware のどちらが teacher 構造に合っているか、最小条件で確認する

### 5.2 重要な注意
この Part は、**実行環境に合わせて「極小条件」に落とす**。  
runbook としては最小案を示すが、ローカルドライバで seed や局数をさらに絞ってもよい。

### 5.3 条件
- A2: `shanten_hint=on`, `strict_top1`
- B2: `shanten_hint=on`, `tie_aware_best_set`

### 5.4 実行方針
「極小データ」を作るため、以下のように極端に小さくする。

- seed: `42`
- `selfplay.imitation_matches=4`
- `training.imitation_epochs=40`
- `selfplay.num_matches=0`
- `evaluation.num_matches=20`

目的は性能評価ではなく、**覚え切れるか** の確認。

### 5.5 コマンド

#### A2: shanten hint on / strict / tiny overfit
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    experiment.phases='["imitation","selfplay","eval"]' \
    imitation.num_workers=10 \
    selfplay.imitation_matches=4 \
    training.imitation_epochs=40 \
    selfplay.num_matches=0 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    feature_encoder.shanten_hint='{"enabled":true}' \
    training.imitation_loss_mode=strict_top1 \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=20 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

#### B2: shanten hint on / tie-aware / tiny overfit
```bash
python -m mahjong_rl.cli \
  --config configs/stage1_full_flat_mlp_imitation_then_ppo.yaml \
  --base-dir runs \
  --override \
    experiment.global_seed=42 \
    experiment.phases='["imitation","selfplay","eval"]' \
    imitation.num_workers=10 \
    selfplay.imitation_matches=4 \
    training.imitation_epochs=40 \
    selfplay.num_matches=0 \
    selfplay.num_workers=10 \
    selfplay.policy_ratio=1.0 \
    selfplay.save_baseline_actions=false \
    feature_encoder.shanten_hint='{"enabled":true}' \
    training.imitation_loss_mode=tie_aware_best_set \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=20 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 5.6 主判定
- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`
- imitation loss

### 5.7 読み方
- strict / tie-aware のどちらでも十分覚え切れないなら、学習系上流をまだ疑う
- tie-aware の方が極小条件で明確に teacher 再現を押し上げるなら、学習目標が合っていない可能性が高い

---

## 6. Part 3: imitation-only で strict vs tie-aware 比較

### 6.1 目的
- `shanten_hint=on` を固定し、strict top-1 CE の限界が本当にあるかを見る
- 通常 imitation-only 相当条件で、学習目標の違いが teacher 再現率と after 指標にどう出るか確認する

### 6.2 条件
- A3: `shanten_hint=on`, `strict_top1`  
  ※ **新規実行は行わず、Part 1 の B1 をそのまま流用する**
- B3: `shanten_hint=on`, `tie_aware_best_set`

### 6.3 共通条件
- seeds: `42,43,44,45,46`
- `experiment.phases='["imitation","selfplay","eval"]'`
- `imitation.num_workers=10`
- `selfplay.imitation_matches=25`
- `training.imitation_epochs=4`
- `selfplay.num_matches=0`
- `selfplay.num_workers=10`
- `selfplay.policy_ratio=1.0`
- `selfplay.save_baseline_actions=false`
- `feature_encoder.shanten_hint='{"enabled":true}'`
- `evaluation.mode=rotation`
- `evaluation.rotation_seats='[0,1,2,3]'`
- `evaluation.num_matches=50`
- `evaluation.num_workers=10`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

### 6.4 実行方針
- **A3 は Part 1 の B1 を再利用**する
- この Part で新規に実行するのは **B3 のみ**

### 6.5 コマンド

#### A3: shanten hint on / strict
- **Part 1 の B1 をそのまま使用する（再実行しない）**

#### B3: shanten hint on / tie-aware
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
    training.imitation_loss_mode=tie_aware_best_set \
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 6.6 主判定
主役:
- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`

次点:
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 6.7 読み方
- tie-aware が teacher 指標と after 指標の両方で良ければ、strict の限界がかなり怪しい
- tie-aware が teacher 指標だけ改善し、after が変わらなければ、学習目標の改善は局所的
- tie-aware でもダメなら、`shanten_hint` 自体の downstream 価値が低い可能性が高い

---

## 7. Part 4: strict vs tie-aware を warm start + PPO で再比較

### 7.1 目的
- imitation-only での差が、warm start + PPO を通しても維持されるか確認する
- strict top-1 と tie-aware の違いが downstream でも意味を持つかを見る

### 7.2 実行条件
Part 4 では、`shanten_hint=on` を固定し、
- `strict_top1`
- `tie_aware_best_set`
の **両条件** を warm start + PPO で比較する。

固定 baseline:
- `training.lr=0.0001`
- `training.epochs=4`
- `training.value_loss_coef=0.25`
- `training.clip_epsilon=0.2`
- `training.batch_size=256`
- `training.gamma=0.99`
- `training.gae_lambda=0.95`

固定:
- seeds: `42,43,44,45,46`
- `imitation.num_workers=10`
- `selfplay.imitation_matches=25`
- `training.imitation_epochs=4`
- `selfplay.num_matches=200`
- `selfplay.num_workers=10`
- `selfplay.policy_ratio=1.0`
- `selfplay.save_baseline_actions=false`
- `feature_encoder.shanten_hint='{"enabled":true}'`
- `evaluation.mode=rotation`
- `evaluation.rotation_seats='[0,1,2,3]'`
- `evaluation.num_matches=50`
- `evaluation.num_workers=10`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

### 7.3 比較条件
- A4: `strict_top1`
- B4: `tie_aware_best_set`

### 7.4 コマンド

#### A4: shanten hint on / strict / PPO
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
    feature_encoder.shanten_hint='{"enabled":true}' \
    training.imitation_loss_mode=strict_top1 \
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

#### B4: shanten hint on / tie-aware / PPO
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
    feature_encoder.shanten_hint='{"enabled":true}' \
    training.imitation_loss_mode=tie_aware_best_set \
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

### 7.5 主判定
主評価:
1. `Δavg_rank`
2. `Δavg_score`
3. `Δdeal_in_rate`
4. `Δwin_rate`

次点:
- after 指標
- teacher 指標
- imitation loss

### 7.6 読み方
- tie-aware が PPO 後でも改善するなら、strict top-1 学習目標がボトルネックだった可能性が高い
- imitation-only では良くても PPO でまた悪いなら、なお downstream interaction が問題

---

## 8. 成功判定

### 8.1 共通
- `summary.json.success == true`
- `aggregate.eval_mode == "rotation"`（batch）
- `summary.encoder_features.shanten_hint` が意図通り
- `summary.encoder_features.input_dim` が記録されている
- `summary.phase_stats.imitation.imitation_loss_mode` または同等キーで mode が追える
- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`
が成果物から読める

### 8.2 Part 1〜3
- `summary.phase_status.imitation == "success"`
- `summary.phase_status.selfplay == "success"`
- `summary.phase_status.eval == "success"`
- `selfplay.num_matches=0` は正常
- `teacher_best_set_hit_rate >= teacher_top1_match_rate`
  が run 単位で概ね成り立つ
- 指標が `null` / `NaN` でない

### 8.3 Part 4
- `eval_before/eval_rotation.json`
- `eval/eval_rotation.json`
- `eval/eval_diff.json`
が存在
- `eval_diff` の主要4指標 delta が非 `null`

---

## 9. レポートに必ず含める項目

### 9.1 Part 1
- batch_dir（A1/B1）
- success
- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`
- after 指標
- imitation loss
- 実行時間

### 9.2 Part 2
- run_dir（A2/B2）
- success
- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`
- imitation loss
- after 指標
- 「極小条件で覚え切れたか」の所見

### 9.3 Part 3
- Part 1-B1 を A3 として再利用したことの明記
- batch_dir（B3）
- strict vs tie-aware の比較表
- teacher 指標
- after 指標
- imitation loss
- 実行時間
- strict top-1 の限界が疑われるかの所見

### 9.4 Part 4
- batch_dir（A4/B4）
- delta 比較表
- after 指標
- teacher 指標
- 実行時間
- final interpretation

---

## 10. 最終的な分岐判断

### ケースA
- tie-aware が Part 3 でも Part 4 でも良い  
→ 次 baseline 候補  
→ 「strict top-1 の限界」がかなり濃い

### ケースB
- tie-aware は Part 3 では良いが Part 4 では悪い  
→ 学習目標改善だけでは足りず、downstream interaction が問題

### ケースC
- tie-aware でもほとんど改善しない  
→ `shanten_hint` 自体の downstream 価値が低い、または別表現が必要

### ケースD
- 極小過学習ですら再現が弱い  
→ まだ上流の健全性や設定を疑うべき

---

## 11. 実行時間を有効に使うためのメモ

- Part 1 と Part 3-B は batch でそのまま回せる
- Part 3-A は Part 1-B1 を再利用する
- Part 2 は軽いので、間に差し込んでもよい
- Part 4 は strict / tie-aware 両方を実行する
- ローカルドライバを使う場合は、
  - run ごとに `part` / `role` / `loss_mode` を持たせる
  - Part 3-A は `source_run` として Part 1-B1 を参照する
のがよい

---

## 12. この runbook の意味

この runbook は単なるハイパラ探索ではない。  
目的は、

> **`shanten_hint` がダメなのか、strict imitation 目標がダメなのか、PPO downstream がダメなのか**

を、1本の論理線で切り分けることである。

実行後には、
- 上流健全性
- 学習目標
- downstream interaction
のどこが本命かをかなり明確にすることを狙う。

---