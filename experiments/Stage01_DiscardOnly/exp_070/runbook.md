# Experiment Runbook: exp_070

作成日: 2026-03-23  
参照: `experiments/exp_068/report.md`, `experiments/exp_069/report.md`

## 1. 背景

`exp_068` で、post-fix 後の current best baseline 候補は

- `gamma=0.75`
- `gae_lambda=0.3`
- `clip_epsilon=0.15`
- `policy_anchor.coef=0.75`
- `training.rule_mix.policy_ratio=0.10`
- `value_loss_coef=0.25`
- `reward.shaping.shanten_delta.scale=0.003`

であることが見えている。特に B `anchor075_ratio010` は 3 seeds で

- `final_score mean = 2348.58`
- `cycle 20-29 mean = 2494.93`
- `drawdown mean = 882.33`

となり、`cycle 20-29` 平均では imitation 直後を上回った。

ここまでで、現行特徴量のままでも plateau 保持はかなりできることが分かった。次は、FullObservation 下で追加した防御特徴

- `opponent_current_shanten`
- `opponent_tenpai_flag`
- `danger_mask`

が、さらに plateau を押し上げるかを確認する。

## 2. 問い

新しい防御特徴のうち、どの成分が実際に効くのかを切り分ける。

具体的には次を確認する。

1. 相手の危険度文脈 (`opponent_current_shanten`, `opponent_tenpai_flag`) だけで改善するか
2. 牌別危険情報 (`danger_mask`) だけで改善するか
3. 文脈 + danger mask の併用が最良か

## 3. 基準条件

基準は `exp_068` の B `anchor075_ratio010` とする。

- seed: `42`
- `gamma=0.75`
- `gae_lambda=0.3`
- `clip_epsilon=0.15`
- `policy_anchor.coef=0.75`
- `training.rule_mix.policy_ratio=0.10`
- `value_loss_coef=0.25`
- `reward.shaping.shanten_delta.scale=0.003`
- `policy_direct_hints.sources = ["shanten_hint", "discard_ukeire_hint"]`

補足:

- 今回はまず `seed=42` の 1 seed pilot で切る
- `REF` の新規再実行は行わず、比較基準は `exp_068/report.md` の B `seed 42` を参照する
- 新特徴量の速度オーバーヘッドは軽いローカル計測で encoder 約 `+18.6%` だったため、pilot 実行コストとしては許容範囲とみなす

## 4. 比較条件

### REF `anchor075_ratio010` (参照のみ)

`exp_068` の B `seed 42` を参照する。

- 特徴量追加なし
- `policy_direct_hints.sources = ["shanten_hint", "discard_ukeire_hint"]`

### A `context_only`

追加:

- `feature_encoder.opponent_current_shanten.enabled = true`
- `feature_encoder.opponent_tenpai_flag.enabled = true`

据え置き:

- `feature_encoder.danger_mask.enabled = false`
- `policy_direct_hints.sources = ["shanten_hint", "discard_ukeire_hint"]`

狙い:

- trunk に入る「相手がどれだけ危険か」の文脈だけで改善するかを見る

### B `danger_only`

追加:

- `feature_encoder.danger_mask.enabled = true`

据え置き:

- `feature_encoder.opponent_current_shanten.enabled = false`
- `feature_encoder.opponent_tenpai_flag.enabled = false`

変更:

- `policy_direct_hints.sources = [
    "shanten_hint",
    "discard_ukeire_hint",
    "danger_mask_shimo",
    "danger_mask_toimen",
    "danger_mask_kamicha"
  ]`

狙い:

- danger mask だけで、牌別の守備判断が直接改善するかを見る

### C `context_plus_danger`

追加:

- `feature_encoder.opponent_current_shanten.enabled = true`
- `feature_encoder.opponent_tenpai_flag.enabled = true`
- `feature_encoder.danger_mask.enabled = true`

変更:

- `policy_direct_hints.sources = [
    "shanten_hint",
    "discard_ukeire_hint",
    "danger_mask_shimo",
    "danger_mask_toimen",
    "danger_mask_kamicha"
  ]`

狙い:

- 文脈 + danger mask を同時に入れたときが最良かを見る

## 5. 実行順

1. A `context_only`
2. B `danger_only`
3. C `context_plus_danger`

理由:

- まず trunk 文脈だけの寄与を見たい
- 次に direct branch だけの寄与を見る
- 最後に併用で相乗効果があるかを確認する

## 6. 評価指標

主に次を見る。

1. `cycle 20-29 mean avg_score`
2. `final avg_score`
3. `best -> final drawdown`
4. `avg_rank`
5. `win_rate`
6. `deal_in_rate`
   - 注: 純粋な放銃率ではなく、相手得点を伴う失点イベント率として解釈する
7. `clip_fraction`
8. `ratio_std`
9. `teacher_best_set_hit_rate_after`
10. `improve / same / worsen advantage`

## 7. 成功判定

pilot では次のどれかを満たせば当たり候補とみなす。

- `cycle 20-29 mean` が REF を上回る
- `final avg_score` が REF を上回る
- `drawdown` が REF より小さい
- `deal_in_rate` が改善しつつ `win_rate` が大きく悪化しない

逆に、

- `deal_in_rate` だけ改善して `win_rate` が大きく落ちる
- `improve / same / worsen` の向きが不自然化する

場合は、防御に寄りすぎか設計ミスの可能性を疑う。

## 8. 解釈ルール

### A が良い場合

- trunk に入る危険度文脈が効いている
- danger mask まで直接与えなくても改善余地がある

### B が良い場合

- 牌別危険情報を直接 logits 近くに入れるのが本命
- danger mask の寄与が大きい

### C だけ良い場合

- 「誰が危険か」という文脈と、「どの牌が危険か」という局所情報の両方が必要
- 今回の防御特徴設計は bundle で使う価値がある

### どれも悪い場合

- 現行 best baseline に対して新特徴量はまだ不要
- もしくは direct hint の入れ方 / trunk バランスを見直す

## 9. 次アクション

### 当たり条件が出た場合

- その条件だけ `seed=43,44` を追加し、`exp_068` B の 3-seed baseline と比較する

### どれも横並びの場合

- 特徴量追加は一旦保留し、現行 best baseline の近傍ハイパラ探索に戻る

### signal が崩れた場合

- 特徴量設計よりも model への入れ方を再検討する
- 特に `danger_mask` の direct branch source 構成を見直す
