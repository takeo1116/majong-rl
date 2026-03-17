# Bug Report: Full Observation Auxiliary Features Used Player 0 Hand (exp_059_bugfix)

作成日: 2026-03-17  
起点実験: `experiments/exp_059_bugfix`  
関連修正: CQ-0208

## 1. 概要

`observation_mode=full` の `FlatFeatureEncoder` で、

- `shanten_hint`
- `discard_ukeire_hint`
- `current_shanten`
- `shape_hint`

が、実際の行動者 `current_player` の手牌ではなく、**常に player 0 の手牌**から生成されていた。

その結果、`current_player != 0` の局面では補助特徴が別プレイヤーの手牌に対応するものになっており、特に `shanten_hint` / `discard_ukeire_hint` を強く使う imitation 実験や `policy_direct_hints` 系実験の解釈を大きく歪めていた。

## 2. 発見経緯

`exp_059` の長尺 imitation (`1000 matches x 50 chunks`) 後に、

- `teacher_best_set_hit_rate ≈ 0.94`
- それでも `avg_score ≈ -6959`

という結果をどう解釈するかを議論する中で、まず `best_set` miss の内訳を後解析した。

その後、`teacher_best_mask` と `shanten_hint` / `discard_ukeire_hint` の整合を spot check すると、同一 shard で大規模な不整合が観測された。

そこから生成経路をソースで追ったところ、

1. `FullObservation.hands` は絶対座席順 `[0..3]`
2. 実際の行動者は `obs.current_player`
3. しかし `FlatFeatureEncoder._encode_full()` は補助特徴用の手牌として `obs.hands[0]` を固定使用

となっていることが分かった。

## 3. 事象の詳細

### 3.1 期待仕様

`observation_mode=full` でも、打牌候補に依存する補助特徴は**現在の行動者の手牌**から計算されるべきである。

具体的には以下が `obs.current_player` 基準でなければならない。

- `shanten_hint`
- `discard_ukeire_hint`
- `current_shanten`
- `shape_hint`

### 3.2 実際

修正前の `FlatFeatureEncoder._encode_full()` では、4家手牌を列挙するループの中で `p == 0` の手牌だけを補助特徴用に保持していた。

そのため:

- `current_player == 0` の局面ではたまたま正しい
- `current_player != 0` の局面では誤った手牌から補助特徴を生成

となっていた。

full 観測では `current_player` が 0 固定ではないため、実質的に多数の局面で補助特徴が誤っていた。

## 4. 根本原因

根本原因は、`full` 観測の補助特徴生成で

- `FullObservation.hands` の絶対座席順
- `obs.current_player` による行動者指定

の区別を encoder 側で正しく扱っていなかったこと。

`partial` 観測では手牌は常に自家情報なので問題は出ないが、`full` 観測では `hands[0]` を「現在の行動者」と見なすことはできない。

## 5. 影響範囲

### 5.1 直接影響

`observation_mode=full` かつ以下のいずれかを有効化している run に影響する。

- `feature_encoder.shanten_hint.enabled=true`
- `feature_encoder.discard_ukeire_hint.enabled=true`
- `feature_encoder.current_shanten.enabled=true`
- `feature_encoder.shape_hint.enabled=true`

特に `policy_direct_hints` 新モデルは `shanten_hint` / `discard_ukeire_hint` を policy logits 直前で強く使うため、影響が大きい。

### 5.2 実験影響

少なくとも以下の解釈は修正前結果をそのまま信じられない。

- `exp_058` の imitation-only A/B 比較
- `exp_059` の multi-chunk imitation ceiling 比較
- `shanten_hint` / `discard_ukeire_hint` に依存する teacher 指標の解釈

また、`full` 観測の補助特徴を利用した過去実験全般で、性能差や ceiling の解釈に再評価が必要な可能性がある。

## 6. 修正内容（CQ-0208）

以下を修正した。

- `python/mahjong_rl/encoders/flat_encoder.py`
  - `_encode_full()` で補助特徴用手牌を `p == 0` ではなく `p == obs.current_player` から取得
  - 変数名も `hand_counts_p0` から `hand_counts_current` に整理
- `tests/python/test_encoders.py`
  - `FullObservation` で `current_player != 0` のケースを直接検証するテストを追加
  - `shanten_hint`
  - `discard_ukeire_hint`
  - `current_shanten`
  - `shape_hint`
    が `hands[current_player]` から計算されることを固定
- `docs/RL_SPEC.md`
  - `full` 観測の補助特徴は `current_player` 基準であることを明記

## 7. 修正後の期待挙動

- `observation_mode=full` でも、補助特徴は常に行動者の手牌に対応する
- `teacher_best_mask` と `shanten_hint` / `discard_ukeire_hint` の整合が回復する
- `policy_direct_hints` の効果を正しく評価できる
- imitation ceiling / teacher 指標の解釈が、ようやく仕様どおりにできる

## 8. 再発防止

- `full` 観測で「絶対座席順」と「現在の行動者」を混同しないテストを恒常化する
- encoder の補助特徴は、どのプレイヤーの手牌を参照しているかを unit test で直接固定する
- 後解析スクリプト（`scripts/local/diagnose_best_set_miss_breakdown.py`, `scripts/local/audit_teacher_best_mask_consistency.py`）を、特徴量と教師信号の整合確認に継続利用する

## 9. 今後の扱い

- 本修正前の `exp_058` / `exp_059` は、**補助特徴が壊れた条件での結果**として扱う
- 同条件を修正後に再取得し、
  - imitation-only A/B
  - multi-chunk imitation ceiling
  - 必要なら rule-only mixed PPO
  を再評価する
- 特に `policy_direct_hints` 新モデルは、修正後にどこまで ceiling を押し上げるかを改めて確認する
