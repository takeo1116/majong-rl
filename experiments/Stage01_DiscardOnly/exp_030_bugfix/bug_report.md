# Bug Report: Reward Scale Not Applied to Stage1Env (exp_030)

作成日: 2026-03-11  
起点実験: `experiments/exp_030`  
関連修正: CQ-0162

## 1. 概要

`reward.point_delta_scale=0.0001` を設定しているにもかかわらず、実際の self-play / eval / imitation で `Stage1Env` に reward config が渡っておらず、C++ デフォルト値 `point_delta_scale=1.0` で報酬が計算されていた。

その結果、`point_delta_reward` が「点数差そのもの（raw points）」に近いスケールで learner に流入し、想定していた報酬単位（1/10000）と大きく乖離していた。

## 2. 発見経緯

exp_030 の新診断（`shanten_diag.reward / point_delta_reward / shanten_delta_reward / delta_t`）確認中に、以下の不自然値を検出した。

- `improve.point_delta_reward.mean ≈ -39.67`
- `worsen.point_delta_reward.mean ≈ -8.26`

`point_delta_scale=0.0001` が有効なら 1 step 平均がこの絶対値になるのは不自然で、
「1 step あたり数万点規模の損失」を示す値になっていたため、報酬単位の不整合を疑って調査を開始。

調査は `experiments/exp_030/reward_unit_checklist.md` に沿って実施。

## 3. 事象の詳細

### 3.1 期待仕様

- 設定: `configs/stage1_full_flat_mlp_imitation_then_ppo.yaml` で `reward.point_delta_scale=0.0001`
- 期待: `point_delta_reward = (実点数差) * 0.0001`

### 3.2 実際

- `Stage1Env` 生成時に `RewardPolicyConfig` が未注入の経路が存在
- 未注入時は C++ 側デフォルト `point_delta_scale=1.0` が適用
- 結果として `point_delta_reward` は raw points スケールで記録・学習

## 4. 根本原因

根本原因は「reward config の経路未接続」。

修正前は、以下で `Stage1Env` に reward config が渡らない経路があった。

- self-play 経路
- imitation warm start（self-play worker と同じ env 構築経路）
- evaluation worker 経路

このため、設定ファイル上では `0.0001` を持っていても実行時に反映されず、結果的に `1.0` が使われていた。

## 5. 影響範囲

## 5.1 直接影響

- Stage1 の報酬値スケールが想定より 10000 倍大きい状態で学習される
- `point_delta_reward` を使った診断値（`reward`, `delta_t`, `return`, `advantage`）の解釈が歪む
- reward 成分分解で「逆転がどこで起きているか」の定量比較が不正確になる

## 5.2 実験影響

- `point_delta_scale=0.0001` を前提に評価していた過去の Stage1 実験は、実際には raw points スケールで動いていた可能性が高い
- 少なくとも exp_030 の初回観測値はこの不具合を反映している

注: 相対比較（同一バグ条件間の比較）は一部有効でも、
「想定仕様に対する絶対値解釈」は再評価が必要。

## 6. 修正内容（CQ-0162）

以下の経路で `RewardPolicyConfig(point_delta_scale=...)` を明示的に渡すように修正。

- `python/mahjong_rl/selfplay_worker.py`
  - `config["reward"]` から `RewardPolicyConfig` を構築し `Stage1Env` に注入
- `python/mahjong_rl/evaluator.py`
  - `EvaluationRunner` が受け取った reward config を `Stage1Env` に注入
- `python/mahjong_rl/runner.py`
  - eval worker 起動時に reward config を転送

追加テスト:

- `tests/python/test_selfplay.py::TestRewardScale`
- `tests/python/test_runner.py::TestRewardScaleE2E`

## 7. 修正後の期待挙動

- `point_delta_scale=0.0001` 設定時:
  - 例: `-1000` 点イベントは `point_delta_reward=-0.1`
- self-play / imitation / eval で同一 reward policy が適用される
- `shanten_diag` の reward 成分と `delta_t` が仕様スケールで解釈可能になる

## 8. 再発防止

- reward unit チェックを runbook の確認項目に恒常化する
  - 1 step の `point_delta_reward` 絶対値の sanity check
  - `reward.point_delta_scale` の config 実注入確認
- 診断系追加時は「config値の伝播先（env constructor）まで」統合テストで固定する

## 9. 今後の扱い

- 本修正前に取得した実験結果は、報酬単位の前提を明記したうえで参照する
- 単位整合後の baseline を再取得して、主要な比較軸（`reward`, `delta_t`, `advantage`, `eval_diff`）を更新する
