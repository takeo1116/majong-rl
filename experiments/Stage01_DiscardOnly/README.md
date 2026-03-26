# Stage01 DiscardOnly

## 概要

`Stage01_DiscardOnly` は、このプロジェクトで最初に立ち上げた学習段階である。
学習対象は **自摸直後の打牌のみ** で、副露は行わない。

- 学習対象: 打牌 (`Discard`)
- 自動処理: `Ron`, `Tsumo`, `Riichi`, `Skip` 系の非打牌判断
- 主目的:
  - imitation / PPO 基盤の立ち上げ
  - 観測・特徴量・reward・advantage・worker 周りの不具合洗い出し
  - FullObservation 下での学習可能性の確認

この段階は、最終目標そのものではなく、**後続ステージのための学習基盤を作る段階**として運用した。

## 何をやったか

Stage01 では、大きく次の 4 つを進めた。

1. imitation + PPO の基礎パイプライン構築
2. reward / target / same-player semantics の不具合修正
3. PPO hyperparameter と training regime の探索
4. 防御寄り特徴量の追加による headroom の検証

途中では複数回の bugfix フェーズを挟み、特に

- trajectory / advantage semantics
- worker / evaluation の安定性
- feature / direct-hint 周りの整合性

を重点的に修正した。

## Stage01 で分かったこと

### 1. PPO 自体は壊れていなかった

停滞していた時期には、

- PPO / GAE / target の問題なのか
- モデルや特徴量の headroom が足りないのか

が切り分けきれていなかった。

しかし Stage01 後半の結果から、**学習余地のある特徴量を与えれば PPO は imitation を安定して上回れる** ことが分かった。

### 2. corrected semantics 後の強い baseline は作れた

bugfix 後の hyperparameter 探索では、

- `gamma=0.75`
- `gae_lambda=0.3`
- `value_loss_coef=0.25`
- `policy_anchor.coef=0.75`
- `training.rule_mix.policy_ratio=0.10`

の組み合わせが、Stage01 の post-fix baseline として最も安定していた。

特に `exp_068` と `exp_069` で、

- imitation 直後を後半 plateau 平均で上回れること
- `policy_ratio` を上げすぎると score plateau が下がること

が確認できた。

### 3. 防御特徴量は非常に大きな headroom を持っていた

Stage01 の最終フェーズでは、

- `opponent_current_shanten`
- `opponent_tenpai_flag`
- `danger_mask`

を追加して比較した。

結果はかなり明確で、**`danger_mask` が大当たり**だった。

- `context_only` は弱い
- `danger_only` は大幅改善
- `context_plus_danger` が最良

という構図で、FullObservation 下では防御由来の改善余地が非常に大きいと分かった。

## 現時点の Stage01 best

現時点の Stage01 current best は、`exp_070` の

- `C context_plus_danger`

である。

3 seeds 集計の要約:

- `final avg_score mean = 6518.75`
- `cycle 20-29 mean = 6073.86`
- `drawdown mean = 133.25`

この結果により、Stage01 については

- PPO が imitation を超えられること
- corrected semantics と current diagnostics が概ね健全であること
- FullObservation では防御特徴量が非常に効くこと

を確認できた。

## Stage01 の位置づけ

Stage01 はこれ以上の主戦場ではない。
今後は次の用途に使う。

- regression harness
- FullObservation 上限比較
- feature / PPO / diagnostics の sanity check

つまり Stage01 は「完了して捨てる段階」ではなく、**後続ステージのための基準系**として残す。

## 次ステージへの引き継ぎ

Stage01 の結果から、次は `Stage02a CallUnlock` に進む。

主な理由は次のとおり。

- Stage01 の停滞は、学習器そのものの限界ではなかった
- 打牌のみルールでさらに細かく詰めるより、行動空間を広げる方が情報価値が高い
- 最終目標は「完全な麻雀ルールで強いエージェント」であり、discard-only は中間段階に過ぎない

Stage02a では、まず

- `Chi`
- `Pon`
- `Daiminkan`
- `Skip`

を学習対象に加え、`DiscardPolicy` と `CallPolicy` を分けた構造へ進む。

## 参照先

Stage01 全体の流れを見るときは、次を優先して読めば十分である。

- `experiments/Stage01_DiscardOnly/exp_065_bugfix/report.md`
- `experiments/Stage01_DiscardOnly/exp_068/report.md`
- `experiments/Stage01_DiscardOnly/exp_069/report.md`
- `experiments/Stage01_DiscardOnly/exp_070/report.md`

補足:

- `deal_in_rate` は純粋なロン放銃率ではなく、現在の evaluator 実装上は「相手得点を伴う失点イベント率」に近い
- 各 `exp_xxx` の詳細条件は個別 `runbook.md` / `report.md` を参照
