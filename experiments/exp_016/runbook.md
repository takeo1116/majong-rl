# experiments/exp_016/runbook.md（Runbook 16）

最終更新: 2026-03-08  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **shanten_hint on/off で imitation 教師再現度（top-1 一致率 / tie許容包含率）を比較し、特徴量が本当に教師方策再現に効いているかを診断する**

---

## 0. この実験の位置づけ

Runbook 14 では、`shanten_hint` を imitation-only 相当で比較した。

結果:
- `shanten_hint=on` は
  - `avg_rank` を小幅改善
  - `avg_score` を小幅改善
- 一方で
  - `win_rate` は小幅悪化
  - `deal_in_rate` はほぼ同等で微悪化

Runbook 15 では、同じ特徴量を warm start + PPO 条件で比較した。

結果:
- `shanten_hint=on` は
  - `eval_before -> eval` の delta でも
  - after 指標でも
  総合的に `off` より悪かった
- したがって、**現形の `shanten_hint` は採用見送り**となった

しかし、ここで未解決なのは次の問いである。

> **`shanten_hint=on` で、モデルはそもそも教師方策を十分再現できているのか？**

もし再現率が低いなら、
- 学習設定
- 特徴量受け渡し
- 教師ラベル整合
- action mask / legal candidate
- より上流の設計 / 実装問題

まで疑う価値がある。

一方、再現率が高いのに PPO 後で悪いなら、
- 教師方策の限界
- shortcut 的作用
- 最終性能に効かない局所ヒューリスティクス

が本命になる。

Runbook 16 は、これを切り分けるための **診断実験** である。

---

## 1. この実験の意図

### 1.1 何を知りたいのか
今回の主質問は次の2つ。

1. `shanten_hint=on` は、imitation における **教師 top-1 action** の再現率を本当に改善するか
2. `shanten_hint=on` は、tie 許容の **教師最良候補集合** への包含率を改善するか

### 1.2 なぜこの診断が重要か
今回の `shanten_hint` は、かなり強い特徴量である。  
あなたの過去経験では、

> **「そのフラグが立っている牌を切る」ことを学習するだけで、かなりルールベースに近い強さを再現できる**

という前提がある。

したがって、もし今回 `shanten_hint=on` にもかかわらず教師再現率が低いなら、

- 特徴量は入ったがモデルが使えていない
- imitation 学習設定が不十分
- 実装上のどこかに不自然さがある

といった **健全性問題** を疑うべきである。

### 1.3 今回は性能比較ではなく診断
今回は対局成績そのものよりも、

- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`

を主役にする。

対局成績や imitation loss は、あくまで補助観測とする。

---

## 2. 比較したい仮説

### 仮説A
`shanten_hint=on` では
- top-1 一致率
- best-set hit rate

の両方が明確に上がる  
→ 特徴量は教師再現には効いている  
→ それでも PPO で悪いなら、特徴量の downstream 効果や shortcut 性が問題

### 仮説B
`shanten_hint=on` でも top-1 / best-set hit がほとんど上がらない  
→ 特徴量の使われ方、学習設定、モデル容量、あるいはパイプライン上流を疑う

### 仮説C
`best_set_hit_rate` は上がるが `top1_match_rate` はあまり上がらない  
→ モデルは良い候補群には入れているが、strict teacher 再現は弱い  
→ tie 構造や学習目標の扱いが論点になる

---

## 3. 実験方針

### 3.1 比較対象
- A: `shanten_hint=off`
- B: `shanten_hint=on`

### 3.2 実行方式
今回も imitation-only 相当の最短経路を使う。

- `experiment.phases='["imitation","selfplay","eval"]'`
- `selfplay.num_matches=0`

これにより、
- imitation は実行される
- PPO learner は実行されない
- `eval` は imitation 後モデルの評価になる

### 3.3 なぜ imitation-only 相当か
今回知りたいのは **教師再現度** であり、  
PPO を挟むと解釈が混ざるため。

まずは
- `shanten_hint`
- imitation
- 教師再現率

の関係だけを切り出す。

---

## 4. 実験条件

### 4.1 共通条件
- config: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml`
- seeds: `42,43,44,45,46`
- `experiment.phases='["imitation","selfplay","eval"]'`
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

### 4.2 比較条件
- A: `feature_encoder.shanten_hint='{"enabled":false}'`
- B: `feature_encoder.shanten_hint='{"enabled":true}'`

### 4.3 shanten hint 指定の注意
CLI override では section.key 形式のみなので、JSON 値で渡す。

- OFF: `feature_encoder.shanten_hint='{"enabled":false}'`
- ON: `feature_encoder.shanten_hint='{"enabled":true}'`

---

## 5. 実行コマンド

### 5.1 条件A: shanten hint off
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
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

### 5.2 条件B: shanten hint on
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
    evaluation.mode=rotation \
    evaluation.rotation_seats='[0,1,2,3]' \
    evaluation.num_matches=50 \
    evaluation.num_workers=10 \
    training.device=cuda \
    selfplay.inference_device=cpu \
    evaluation.inference_device=cpu
```

---

## 6. 成功判定

### 6.1 batch 単位
- `success_count == 5/5`
- `aggregate.eval_mode == "rotation"`

### 6.2 各 run の最低確認
- `summary.json.success == true`
- `summary.phase_status.imitation == "success"`
- `summary.phase_status.selfplay == "success"`
- `summary.phase_status.eval == "success"`
- `summary.encoder_features.shanten_hint` が意図通り
- `summary.encoder_features.input_dim` が記録されている
- `summary.phase_stats.eval.eval_mode == "rotation"`
- `eval/eval_rotation.json` が存在する

### 6.3 教師再現度指標
少なくとも run 成果物から以下が読めること。

- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`

### 6.4 指標の健全性
- `teacher_best_set_hit_rate >= teacher_top1_match_rate`
  が run 単位で概ね成り立つこと
- 指標が `null` / `NaN` でないこと

### 6.5 selfplay.num_matches=0 の扱い
- `selfplay` phase 自体は走る
- `total_matches=0` は正常
- ここでの `selfplay` は imitation-only 相当経路を成立させるための最小経路である

---

## 7. 集計方法

今回は性能比較ではなく診断なので、**教師再現度指標を主役** にする。

### 7.1 主確認先
- `summary.json`
- `batch_summary.json.runs[*]`
- 可能なら `batch_summary.json.aggregate`

### 7.2 主な集計指標
主診断:
- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`

補助:
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`
- imitation loss
- imitation total_steps

---

## 8. 主な評価項目

### 8.1 最優先（教師再現度）
見る順序:
1. `teacher_top1_match_rate`
2. `teacher_best_set_hit_rate`

### 8.2 次点（補助観測）
- imitation loss
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 8.3 追跡情報
- `summary.encoder_features.shanten_hint`
- `summary.encoder_features.input_dim`
- `notes.md` の `shanten_hint=on/off, input_dim=...`

### 8.4 実行時間
- imitation
- eval
- total

---

## 9. 結果の読み方

### 9.1 条件B（on）で top-1 / best-set が明確に改善する場合
- `shanten_hint` は教師再現には効いている
- それでも exp_015 で悪いなら、問題は
  - 教師方策の限界
  - shortcut 性
  - PPO との相性
に寄る

### 9.2 条件B（on）で best-set は改善するが top-1 は弱い場合
- モデルは良い候補群には入れている
- strict teacher 再現はまだ弱い
- tie 構造や学習目標の設計が論点になる

### 9.3 条件B（on）でも再現率があまり改善しない場合
- 特徴量が十分使われていない
- imitation 設定やモデル容量が不十分
- 特徴量受け渡し / 教師ラベル整合 / mask の問題
- 場合によっては実装バグ寄りの健全性問題
を疑う

---

## 10. 判定ルール

### 10.1 重要判定
この runbook の主目的は、性能優劣ではなく **切り分け** である。

### 10.2 次の分岐
#### ケースA
- `teacher_top1_match_rate` 上昇
- `teacher_best_set_hit_rate` 上昇

→ `shanten_hint` は教師再現には効いている  
→ 特徴量そのものより、teacher quality / PPO interaction を考える

#### ケースB
- top-1 は低い
- best-set も大きく伸びない

→ 学習系の健全性を疑う  
→ 次は
- 小規模過学習 sanity check
- imitation 設定見直し
- ラベル整合確認
を優先する

#### ケースC
- best-set は高い
- top-1 は低い

→ strict teacher 学習の設計や tie の扱いを再点検する

---

## 11. レポートに必ず含める項目

- batch_dir（A/B）
- 成功率
- `aggregate.eval_mode`
- `encoder_features.shanten_hint`
- `encoder_features.input_dim`
- 教師再現度指標（mean ± std）
  - `teacher_top1_match_rate`
  - `teacher_best_set_hit_rate`
- 可能なら imitation loss
- 補助として after 指標
  - `avg_rank`
  - `avg_score`
  - `win_rate`
  - `deal_in_rate`
- 実行時間
  - imitation
  - eval
  - total
- `best_set_hit_rate >= top1_match_rate` の確認結果
- 診断解釈
- 次アクション

---

## 12. この実験の副次目的

この runbook には、教師再現度診断に加えて、

> **新規 imitation 診断指標が runbook/report 運用に自然に乗るか**

を確認する副次目的がある。

確認したい点:
- `summary.json` で run 単位に追えるか
- `batch_summary.json` で横比較できるか
- report に自然に載せられるか
- on/off 比較と同時に読めるか

---

## 13. 次のアクション

### 13.1 `shanten_hint=on` で再現率が高い場合
- 現形特徴量は教師再現には効いている
- 次は
  - teacher quality の限界
  - shortcut 性
  - PPO interaction
を議論する
- 必要なら別形式（弱い scalar / auxiliary）を検討する

### 13.2 `shanten_hint=on` で再現率が低い場合
- 学習系の健全性確認を優先
- 次は
  - 小規模過学習 sanity check
  - imitation パラメータ見直し
  - ラベル / mask / feature 受け渡し確認
に進む

### 13.3 best-set は高いが top-1 が弱い場合
- tie を含む教師構造に対して strict top-1 CE が十分でない可能性
- 必要なら教師表現や loss の扱いを次段で見直す

---

## 14. メモ

- 今回は性能比較ではなく、**学習系の健全性と特徴量の本質的効き方を診断する実験**
- ここで再現率が十分高ければ、「効いているのに勝てない」という次の議論に進める
- ここで再現率が低ければ、性能比較を続ける前に上流を疑うべき

---