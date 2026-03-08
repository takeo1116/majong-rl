# RL_SPEC.md

この文書は、このプロジェクトにおける **学習基盤・評価基盤・実験基盤の長期実装仕様** を定義する。  
日常の実験判断や優先順位は [PROJECT.md](/home/takeo1116/Git/majong-rl/docs/PROJECT.md) を正とし、個別実験の条件と結果は `experiments/exp_xxx/runbook.md` / `report.md` を正とする。  
本書はそれらの土台になる **RL システムの設計基準** を扱う。

ゲームルールそのものは `GAME_RULE.md`、ゲームエンジン仕様は `GAME_SPEC.md`、研究上の方針は `RL_RULE.md` を参照する。

---

## 1. この文書の役割

この文書の用途は次の3つに限定する。

1. RL 基盤を変更する CQ の設計基準にする  
2. 現在実装済みの学習・評価・実験基盤の境界を明確にする  
3. 将来拡張時に、どこまでが現仕様でどこからが構想かを分けて記録する  

この文書は「今すぐ実験でどう動くか」を毎回説明する文書ではない。  
その役割は `PROJECT.md` と各 runbook/report が担う。

---

## 2. 文書の読み方

本書では内容を次の2層に分ける。

- **現仕様**
  - 現在のコードベースで成立している実装仕様
  - レビュー時に「実装とズレていないか」を確認する対象
- **将来案**
  - まだ未実装だが、今後の設計余地として維持したい方向
  - 現実装への適用を前提に読まない

実装修正を伴う議論では、まず現仕様を優先し、将来案は必要なときだけ参照する。

---

## 3. スコープ

### 3.1 この文書が扱うもの

- observation / feature / model の境界
- self-play / imitation / learner / evaluation の責務
- shard / checkpoint / summary / batch_summary の成果物仕様
- 実験設定と run 成果物の最低限の追跡仕様
- 現在の Stage 1 学習基盤

### 3.2 この文書が主対象にしないもの

- ゲームルールの細部
- 実験優先順位そのもの
- 個別のハイパラ採否
- GUI / MLOps / クラウド運用

---

## 4. 現在地の要約

2026-03-08 時点の RL 基盤は次の段階にある。

- Stage 1（DiscardOnly）の学習基盤は成立
- imitation / self-play / PPO / evaluation / batch 集約 / phase 再利用 / resume は実装済み
- `shanten_hint`、`teacher_best_mask`、`tie_aware_best_set`、`ppo_diag` などの診断補助は実装済み
- 現在の主要課題は **PPO が imitation 初期方策を平均的に悪化させること**

したがって本書は、学習基盤の「立ち上げ仕様」ではなく、  
**診断と拡張に耐える基盤仕様** として維持する。

---

## 5. システム全体構成

学習システムは次の層に分ける。

1. **Game Engine Layer (C++)**
   - 対局進行
   - 合法手列挙
   - 観測生成
   - 精算・reward 計算
2. **Binding Layer**
   - Python から C++ の環境を呼び出す境界
3. **Training Layer (Python)**
   - FeatureEncoder
   - Model
   - Imitation / PPO learner
   - shard 読み込み
4. **Self-Play / Evaluation Layer**
   - self-play worker
   - baseline 対戦
   - rotation evaluation
   - 集計
5. **Experiment Management Layer**
   - YAML config
   - run directory
   - checkpoint / metrics / notes / summary
   - batch_summary / batch_table

責務の原則:

- ゲーム進行・精算は C++
- 学習・集計・実験運用は Python
- observation / feature / model / action selection を密結合にしない

---

## 6. 現仕様: 学習対象

### 6.1 Stage 1 の正式対象

現在の正式対象は **DiscardOnly** である。

- 学習対象行動: 自摸直後の打牌のみ
- 副露なし
- ロン / ツモ和了は自動
- テンパイ時自動立直
- 九種九牌なし
- 通常流局あり

### 6.2 Policy 抽象

将来的には policy を分割可能とする。

- `DiscardPolicy`
- `CallPolicy`
- `RiichiPolicy`
- `WinPolicy`

ただし現仕様で学習しているのは `DiscardPolicy` のみ。

---

## 7. 現仕様: Observation / Feature / Model

### 7.1 原則

Observation、Feature 表現、Model は分離する。

- **Observation**: エンジン由来のモデル非依存情報
- **FeatureEncoder**: Observation をモデル入力へ変換
- **Model**: 特徴量から policy / value を出力
- **ActionSelector**: legal mask を適用して行動を選ぶ

### 7.2 Observation モード

現仕様で扱う observation mode:

- `full`
- `partial`

運用上の主系は `full`。  
`partial` は比較・将来拡張のために維持する。

### 7.3 FeatureEncoder

現仕様で利用可能:

- `FlatFeatureEncoder`
- `ChannelTensorEncoder`

ただし現在の主力実験は `FlatFeatureEncoder`。

#### 7.3.1 FlatFeatureEncoder のオプション特徴量

`feature_encoder.shanten_hint.enabled=true` で `delta_shanten_sign[34]` を追加できる。

現仕様:

- 各打牌候補について、打牌後シャンテン変化の符号を補助特徴として持つ
- 既定は `false`
- feature off 時は既存入力次元を維持する

現運用上の注意:

- これはあくまで補助特徴であり、教師方策そのものを直接埋め込むものではない
- 比較実験では on/off を config 上で明示的に追跡する

### 7.4 Model

現仕様での主系:

- `MLPPolicyValueModel`
- FullObservation + FlatFeatureEncoder + MLP

現在の標準的な value head は:

- `round_delta`

複数 value head を許容する設計だが、日常実験の主系はまだ単一 head 前提に近い。

---

## 8. 現仕様: Action / legal mask

Stage 1 では policy は **34 種の打牌ロジット** を出す。  
legal mask により、実行可能な打牌へ射影する。

現仕様:

- action space は牌種ベース
- 実際の `Action` との対応付けは env / selector 側が行う
- legal mask は feature と分離して扱う

---

## 9. 現仕様: reward

### 9.1 現在の主系 reward

現在の主系 config は次である。

- `reward.type = "point_delta"`
- `reward.point_delta_scale = 0.0001`

意味:

- 点数変化に比例した reward を返す
- 学習では scale を掛けた値を用いる

### 9.2 Stage 1 での実際の挙動

運用上の理解としては次の通り。

- 局中の大半の打牌 step の reward は 0
- 和了 / 放銃 / 流局精算 / リーチ棒支払い / 半荘終了時など、点数変化が起きた step に非ゼロ reward が乗る
- learner はその系列 reward から GAE を計算して advantage / return を作る

したがって現 reward は、
**点数変化ベースの sparse な即時報酬**
として扱うのが正確である。

### 9.3 現仕様での reward 関連の既知課題

- sparse
- tail が強い
- 打牌ごとの credit assignment が粗い

これは現在の `PROJECT.md` にある主要診断課題の一つであり、  
reward 設計変更を伴う CQ ではこの節を起点に議論する。

### 9.4 将来案

将来的には以下を許容する。

- `final_rank`
- `combined`
- shaped / intermediate reward
- reward normalization / clipping の明示仕様

ただし現仕様では主系ではない。

---

## 10. 現仕様: 学習データと shard

### 10.1 基本単位

学習データの基本単位は **step sample**。

1 サンプルは少なくとも次を持つ。

- observation / encoded feature に必要な情報
- legal mask
- chosen action
- reward
- log_prob
- value
- terminated flag
- round / episode / worker / seed などの追跡情報

### 10.2 shard 方針

現仕様:

- self-play データは file-based shard で保存
- 保存形式は Parquet
- learner は shard 群を読んで学習する

### 10.3 後方互換

shard は列追加に耐える方針を取る。  
既存 reader / learner を壊さない追加を優先する。

### 10.4 現在追加済みの補助列

現在の主な追加済み情報:

- `teacher_best_mask`
- self-play round end 系の補助出力（別ログ / stats）

これらは診断・模倣学習改善のための列であり、既存列の意味は変えない。

---

## 11. 現仕様: Self-Play

### 11.1 役割

Self-Play Worker は対局を生成し、学習サンプルと統計を保存する。

### 11.2 対戦相手構成

現運用では設定可能だが、主系では

- 学習中 policy
- ルールベース baseline

の混合を用いる。

### 11.3 Stage 1 baseline

現行 baseline の設計意図:

- シャンテン数最小
- 同点なら受け入れ最大
- テンパイ時自動立直

この baseline は imitation 教師と self-play 対戦相手の両方に関与するため、  
変更時は模倣学習と self-play 分布の双方に影響する。

### 11.4 self-play の可観測性

現仕様で run 成果物から確認可能:

- `policy_wins`
- `policy_deal_ins`
- `policy_draws`
- `tsumo_count`
- `ron_count`
- `ryukyoku_count`
- `num_rounds`
- `round_results.jsonl`

これらは PPO 改善診断の基礎材料として扱う。

---

## 12. 現仕様: Learner

### 12.1 役割

Learner は shard を読み込み、model を更新し、training metrics / checkpoint / summary を出力する。

### 12.2 現在の主系アルゴリズム

- imitation warm start
- PPO

### 12.3 warm start

現運用では、ルールベース打牌の軽い imitation を先に行い、その後 self-play + PPO に入る構成を標準とする。

### 12.4 現在の既知課題

現在の実験上の主要課題:

- imitation 改善は見える
- しかし PPO 後に平均悪化が起きやすい

このため learner は、単なる更新器ではなく **診断対象** でもある。

### 12.5 現仕様で成果物に残る learner 情報

- imitation metrics
- PPO loss 系
- `ppo_diag`

`ppo_diag` の主なカテゴリ:

- `advantage_*`
- `return_*`
- `old_value_*`
- `new_value_*`
- `value_error_*`
- `ratio_*`
- `clip_fraction`

出力先:

- `metrics/train_metrics.json`
- `summary.json.phase_stats.learner.ppo_diag`
- `batch_summary.json.runs[].learner_diag`
- `batch_summary.json.aggregate.learner_diag`

### 12.6 現仕様での解釈上の注意

- reuse 実験では self-play が共通なので `value_error` など一部統計は条件間で同じになりうる
- learner 診断統計は「改善そのもの」ではなく「悪化理由の候補切り分け」に使う

---

## 13. 現仕様: imitation 診断

### 13.1 教師再現指標

imitation フェーズ完了後、少なくとも次を出力する。

- `teacher_top1_match_rate`
- `teacher_best_set_hit_rate`

### 13.2 best-set の定義

教師最良候補集合は、RuleBasedBaseline の評価規則

1. シャンテン最小  
2. 受け入れ最大  

で同率最良となる全合法手の集合。

### 13.3 imitation loss mode

現仕様で利用可能な mode:

| 値 | 意味 |
|---|---|
| `strict_top1` | 教師 action への cross-entropy |
| `tie_aware_best_set` | `-log(sum_{a in best_set} pi(a))` |

### 13.4 現仕様上の原則

- strict 経路は既定として維持する
- tie-aware は追加モードとして扱う
- shard に必要な補助情報がなければ fail-fast する

---

## 14. 現仕様: 評価

### 14.1 主指標

評価の主指標は次。

- `avg_rank`
- `avg_score`
- `win_rate`
- `deal_in_rate`

### 14.2 評価モード

現仕様で特に重要なのは `rotation`。

rotation では:

- aggregate 結果
- seat 別結果

を保存する。

### 14.3 `eval_before -> eval`

現在の PPO 比較では、after 指標よりも

- `eval_before`
- `eval`
- `eval_diff`

の前後差分が主評価になることが多い。

したがって、reuse を含む運用でも `eval_before` の復元と `eval_diff` の生成は後方互換を壊さず維持する必要がある。

---

## 15. 現仕様: 実験設定

### 15.1 正本

人間可読な正本は YAML。

主要カテゴリ:

- `experiment`
- `feature_encoder`
- `model`
- `reward`
- `selfplay`
- `training`
- `evaluation`

### 15.2 運用上の原則

- 個別実験の具体条件は runbook に書く
- YAML と CLI override の組み合わせで条件を固定する
- 実験比較で重要な差分は `summary` / `notes` / `batch_summary` から追跡可能であることを重視する

---

## 16. 現仕様: 実験成果物

### 16.1 run directory

run ごとに少なくとも次を持つ。

- `config.yaml`
- `summary.json`
- `notes.md`
- `metrics/`
- `selfplay/`
- `eval/`
- `checkpoints/`

### 16.2 追跡すべき成果物

現在の主な分析対象:

- `summary.json`
- `metrics/train_metrics.json`
- `eval/eval_rotation.json`
- `eval/eval_diff.json`
- `batch_summary.json`

### 16.3 phase 再利用

現仕様では phase 成果物再利用をサポートする。

主な再利用パターン:

- imitation checkpoint 再利用
- `imitation,selfplay,eval_before` 再利用
- phase 単位 resume

設計原則:

- 通常 run を壊さない
- 再利用時は整合性チェックを行う
- summary / manifest から再利用元を追える

---

## 17. 現仕様: batch 集約

batch 実行時は、run ごとの主要成果物を集約する。

少なくとも:

- per-run metrics
- aggregate mean/std

を扱えること。

現時点で特に重要なのは:

- imitation metrics
- learner diagnostic metrics
- eval aggregate metrics

batch 集約は「詳細全部を持つ」よりも、
**runbook / report で横並び比較できる最小十分なキーを固定する**
ことを優先する。

---

## 18. 現在の既知課題

2026-03-08 時点の主要な既知課題は次の通り。

1. PPO が imitation 初期方策を平均的に悪化させる  
2. reward が sparse で tail が強い  
3. learner が見ている target / advantage / value の質が十分良いか未確定  
4. 更新強度の調整だけでは問題が解けない可能性が高い  

この節は `PROJECT.md` の判断と整合すること。  
詳細な優先順位は `PROJECT.md` に置き、本書では「基盤として意識すべき技術課題」に留める。

---

## 19. テスト要件

### 19.1 最低限必要な層

- unit
- integration
- runner / batch 集約
- reproducibility

### 19.2 診断機能追加時の原則

診断メトリクスや補助列を追加するときは、少なくとも次を満たす。

- 既存 run / shard / summary / batch を壊さない
- 追加のみで入れる
- NaN / 欠落 / 型不整合を検知する
- run 単位と batch 単位の最低限の読取り経路をテストする

### 19.3 実行レーン

テストは、

- 変更に直接関係する targeted test
- 必要時のみ smoke / broader integration

を原則とする。  
実験速度を不必要に損なう一律実行は避ける。

---

## 20. CHANGE_QUEUE 運用との関係

RL 基盤の仕様変更は `CHANGE_QUEUE.md` で管理する。

推奨 Type:

- `RL`
- `Training`
- `Experiment`
- `Eval`

レビュー運用上の原則:

- レビュー NG 時は旧 CQ を削除し、新規 CQ を起票する
- 仕様変更と実験解釈を混ぜない
- 本書に関わる変更は、必要なら本書も同期更新する

---

## 21. 将来案

以下は保持したい将来拡張の方向であり、現仕様ではない。

- PartialObservation 主系化
- Full → Partial 蒸留
- CNN / token / transformer 系 encoder
- 複数 policy の本格分離
- `final_rank` や `combined` の本格運用
- shaped reward / curriculum
- search policy / MCTS 接続
- async actor-learner / distributed learner
- ONNX Runtime を使った高速推論運用

将来案を実装対象に昇格させるときは、まず CQ と `PROJECT.md` で優先順位を確定し、その後本書の現仕様へ移す。

---

## 22. 一文要約

**RL_SPEC.md は、現在の Stage 1 学習基盤を壊さず拡張するための長期設計基準であり、日常の実験判断は PROJECT.md、個別条件は runbook/report を正とする。**
