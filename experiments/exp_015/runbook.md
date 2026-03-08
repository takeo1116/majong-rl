# experiments/exp_015/runbook.md（Runbook 15）

最終更新: 2026-03-07  
対象: Stage 1 / FullObservation / FlatFeatureEncoder / MLPPolicyValueModel  
目的: **shanten hint on/off を、同一 learner baseline の warm start + PPO 条件で比較し、imitation-only で見えた小幅改善が PPO 後にも残るか確認する**

---

## 0. この実験の位置づけ

Runbook 14 では、`shanten_hint` を **imitation-only 相当** で比較した。

結果の要点:
- `shanten_hint=on` は
  - `avg_rank` を小幅改善
  - `avg_score` を小幅改善
- 一方で
  - `win_rate` は小幅悪化
  - `deal_in_rate` はほぼ同等で微悪化
- したがって、
  - **強い採用根拠が確定するほどではない**
  - ただし **切るには惜しく、次段に進む価値がある**

ここで未解決なのは、

> **シャンテン補助特徴が PPO 後にも効くのか**
> それとも
> **imitation-only では少し効いても、PPO 段では差が消える / shortcut 依存で悪化するのか**

である。

Runbook 15 では、現時点の learner baseline を固定し、  
**warm start + PPO 条件で shanten hint on/off を比較**する。

---

## 1. この実験の意図

### 1.1 何を知りたいのか
今回の主質問は次の1つに尽きる。

> **shanten hint は、imitation-only の小幅改善を PPO 後にもつなげられるか？**

もしつなげられるなら、
- これは単なる imitation 補助ではなく
- **実際に RL まで含めた初期方策改善の有効特徴**
と言いやすくなる。

逆に、PPO を通すと差が消えたり悪化したりするなら、
- shortcut 的である
- 現在の入れ方は強すぎる / 情報の質が足りない
可能性が高くなる。

### 1.2 今なぜこれを見るのか
- learner ノブはかなり整理できた
- imitation-only では shanten hint の小幅プラスが見えた
- 次に必要なのは
  - 「PPO を挟んでも有効か」
  - 「PPO が壊す方向をむしろ悪化させないか」
の確認

### 1.3 今回の比較は何を固定するか
この runbook では、**learner baseline は固定** する。

固定 baseline:
- `training.lr=0.0001`
- `training.epochs=4`
- `training.value_loss_coef=0.25`
- `training.clip_epsilon=0.2`
- `training.batch_size=256`
- `training.gae_lambda=0.95`

そのうえで、差分は **`shanten_hint` の on/off のみ** にする。

---

## 2. 比較したい仮説

### 仮説A
`shanten_hint=on` は imitation 後の初期方策を少し良くし、その改善が PPO 後にも残る  
→ `eval_before -> eval` の delta が良くなる、あるいは after 指標が改善する

### 仮説B
`shanten_hint=on` は imitation-only では少し効くが、PPO を通すと差が消える  
→ 補助特徴としては弱い、または learner の方が支配的

### 仮説C
`shanten_hint=on` は shortcut 的に働き、PPO 後にはむしろ悪化する  
→ 現形での採用は見送るべき

---

## 3. 実験方針

### 3.1 比較対象
- A: `shanten_hint=off`
- B: `shanten_hint=on`

### 3.2 固定するもの
今回の比較では、`shanten_hint` 以外は固定し、**1要因比較**にする。

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
- `training.gae_lambda=0.95`
- `training.entropy_coef=0.01`
- `training.clip_epsilon=0.2`
- `training.device=cuda`
- `selfplay.inference_device=cpu`
- `evaluation.inference_device=cpu`

### 3.3 shanten hint の指定
CLI override は section.key 形式のみなので、JSON 値で渡す。

- OFF: `feature_encoder.shanten_hint='{"enabled":false}'`
- ON: `feature_encoder.shanten_hint='{"enabled":true}'`

### 3.4 実行方式
今回は **通常の multi-seed batch 実行** でよい。  
理由:
- 比較したい差は特徴量 on/off であり、参照元 run の seed ごと固定までやらなくても十分解釈可能
- 前回の imitation-only 比較も batch で自然に回せた
- 実行時間も十分許容範囲

---

## 4. 実験規模

今回は、imitation-only ではなく **warm start + PPO** の本命比較なので、最初から本番寄り条件で行う。

- `5 seeds`
- `rotation eval`
- `evaluation.num_matches=50`

---

## 5. 実行コマンド

### 5.1 条件A: shanten hint off
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
    feature_encoder.shanten_hint='{"enabled":false}' \
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

### 5.2 条件B: shanten hint on
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

---

## 6. 成功判定

### 6.1 batch 単位
- `success_count == 5/5`
- `aggregate.eval_mode == "rotation"`

### 6.2 各 run の最低確認
- `summary.json.success == true`
- `summary.encoder_features.shanten_hint` が意図通り
- `summary.encoder_features.input_dim` が記録されている
- `summary.phase_stats.eval.eval_mode == "rotation"`
- `eval/eval_rotation.json` が存在する
- `eval_before` と `eval` の両方が記録されている
- `runs[*].eval_diff` が計算できる（batch summary 側）
  - または run 単位で `eval/eval_diff.json` が存在する

### 6.3 追跡性
- `config.yaml` で shanten hint on/off が確認できる
- `notes.md` に `shanten_hint=on/off, input_dim=...` が残る

---

## 7. 集計方法

今回は PPO を含むため、主評価は again **`eval_before -> eval` の差分** に戻す。

### 7.1 主確認先
- `batch_summary.json.aggregate`
- `batch_summary.json.runs[*].eval_diff`
- run 単位では
  - `summary.json`
  - `eval/eval_rotation.json`
  - `eval/eval_diff.json`

### 7.2 主な集計指標
差分:
- `Δavg_rank`
- `Δavg_score`
- `Δwin_rate`
- `Δdeal_in_rate`

after:
- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

---

## 8. 主な評価項目

### 8.1 最優先（delta）
今回の主評価は `eval_before -> eval` の差分。

見る順序:
1. `Δavg_rank`
2. `Δavg_score`
3. `Δdeal_in_rate`
4. `Δwin_rate`

### 8.2 次点（after）
そのうえで、最終到達点を見る。

- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 8.3 補助確認
- `summary.encoder_features.shanten_hint`
- `summary.encoder_features.input_dim`
- imitation loss（可能なら）
- 実行時間
- 既存 baseline より大きく時間悪化していないか

---

## 9. 結果の読み方

### 9.1 条件B（shanten hint on）が良い場合
- シャンテン補助特徴は imitation-only の補助に留まらず、PPO 後まで価値がある
- 現在の flat + MLP 構成では有力な補助特徴と考えてよい
- 次 baseline 候補として採用を検討できる

### 9.2 条件A/B がほぼ同等の場合
- imitation-only の小幅改善はあるが、PPO を通すと差が縮む
- 補助特徴としての価値は限定的
- 本採用は慎重に判断する

### 9.3 条件B（shanten hint on）が悪い場合
- shortcut 的に働いている可能性がある
- 現形での採用は見送る
- 入れ方を弱める / auxiliary 化する余地を考える

---

## 10. 判定ルール

### 10.1 採用判断
`shanten_hint=on` を有望とみなす条件:

- `Δavg_rank` が改善
- `Δavg_score` が改善
- `Δdeal_in_rate` が悪化しすぎない
- after 指標も大きく悪化していない

### 10.2 本採用候補に進める条件
- delta で優位
- after でも同等以上
- 実行時間増が許容範囲
- `win_rate` / `deal_in_rate` の副作用が小さい

### 10.3 差が僅差の場合
差が僅差なら、
- `Δavg_rank`
- `Δavg_score`
を優先し、
- after 指標
- 実行時間
を補助的に見る。

---

## 11. レポートに必ず含める項目

- batch_dir（A/B）
- 成功率
- `aggregate.eval_mode`
- `encoder_features.shanten_hint`
- `encoder_features.input_dim`
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
- 可能なら imitation loss
- 実行時間
  - imitation
  - selfplay
  - learner
  - eval
  - total
- shanten hint on/off の比較解釈
- 次アクション

---

## 12. この実験の副次目的

この runbook には、shanten hint の有効性確認に加えて、

> **新特徴量 on/off が PPO を含む通常 batch 実験でも自然に追跡・比較できるか**

を確認する副次目的がある。

確認したい点:
- `summary.json` / `config.yaml` / `notes.md` で on/off が追えるか
- report で比較しやすいか
- 入力次元変化が運用を壊さないか

---

## 13. 次のアクション

### 13.1 shanten hint on が良い場合
- 暫定 baseline への採用を検討
- 次は
  - gamma
  - 追加特徴量
  - モデル側改善
のどれに進むか判断する

### 13.2 shanten hint on が微妙な場合
- 現形では採用保留
- 必要なら
  - 入れ方を弱める
  - scalar 化
  - auxiliary target 化
を検討する

### 13.3 shanten hint on が悪い場合
- 現形での採用は見送る
- gamma など残ノブに戻る
- 追加特徴量はまだ広げない

---

## 14. メモ

- 今回は imitation-only ではなく、**実際の warm start + PPO 条件での有効性確認**
- ここで勝てれば、shanten hint はかなり有力な特徴量候補になる
- ここで負けるなら、今の入れ方は shortcut 的または効果限定と判断しやすい

---