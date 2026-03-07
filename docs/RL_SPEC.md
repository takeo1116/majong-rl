# RL_SPEC.md - 学習システム実装仕様

この文書は、このプロジェクトにおける **強化学習・探索・実験基盤の実装仕様** を定義する。  
学習方針・研究方針は `RL_RULE.md` に定義し、この文書はそれをどのように実装するかを定義する。

**学習システム実装と最終的に一致していなければならない正本はこの `RL_SPEC.md` とする。**  
`RL_RULE.md` と矛盾がある場合は、両者を更新して整合させること。

ゲームルールそのものは `GAME_RULE.md`、ゲームエンジンの実装仕様は `GAME_SPEC.md` を参照する。

---

## 1. 目的

本システムは、日本式リーチ麻雀ゲームエンジンの上で動作する **学習・探索・評価・実験管理基盤** を提供する。  
主目的は以下のとおりである。

1. Stage 1 から Stage 3 まで段階的に学習可能な構造を提供する
2. FullObservation / PartialObservation の両実験を比較可能にする
3. self-play、評価対戦、探索、記録、再現を一貫して扱えるようにする
4. Python による学習と、将来の C++ 高速推論を両立できる構造を提供する
5. モデル、特徴量、報酬、実験設定を複数保持し、比較可能な研究基盤とする

---

## 2. スコープ

## 2.1 初期実装対象
初期実装では、少なくとも以下を対象とする。

- Stage 1 (DiscardOnly) の学習基盤
- Python + PyTorch による learner
- C++ ゲームエンジンの Python 呼び出し
- FullObservation / PartialObservation 両対応の Observation 入口
- Observation と FeatureEncoder の分離
- `DiscardPolicy` 中心の policy-value 学習
- replay 用データ保存
- shard file による self-play データ永続化
- 実験設定 YAML
- 実験ディレクトリ管理
- checkpoint / metrics / notes 管理
- 評価対戦基盤
- 将来の ONNX export を妨げないモデル設計

## 2.2 初期実装の非対象
初期段階では以下を必須としない。

- 完全行動学習
- 分散 learner
- 高度な league training
- 本格 MCTS 併用学習
- ONNX Runtime による本番推論実装
- GUI ベースの実験管理
- 大規模 MLOps 基盤
- クラウド依存のジョブ管理

## 2.3 並列化の初回スコープと非対象

初回の並列化実装は **単機 multi-process** に限定する。以下は初回スコープ外とする。

- learner の分散化（複数 GPU / 複数ノード）
- 非同期 actor-learner パイプライン
- ノード間通信管理
- 障害復旧・再実行制御
- 高度な動的負荷分散

ただし、将来的に複数サーバへの拡張を妨げない設計（プロセス間で共有メモリに依存しない、file-based model 受け渡し等）を維持する。

---

## 3. 参照文書

- `GAME_RULE.md`
- `GAME_SPEC.md`
- `RL_RULE.md`
- `CHANGE_QUEUE.md`

---

## 4. システム全体構成

学習システムは少なくとも以下の層に分ける。

1. **Game Engine Layer (C++)**
   - 対局進行
   - 合法手列挙
   - 観測生成
   - 状態複製
   - determinization 補助

2. **Binding Layer**
   - Python から C++ Environment を呼ぶための境界
   - pybind11 等を想定

3. **Training Layer (Python)**
   - FeatureEncoder
   - Model
   - Policy/Value 学習
   - replay / shard 読み込み
   - optimizer / scheduler

4. **Self-Play / Evaluation Layer**
   - self-play worker
   - baseline 対戦
   - checkpoint 評価
   - metrics 集計

5. **Experiment Management Layer**
   - YAML 設定
   - run ディレクトリ
   - checkpoint 保存
   - notes / metrics / eval 結果管理

---

## 5. 学習対象の抽象化

## 5.1 Policy 分割

行動種別の性質が異なるため、policy は分割可能な設計とする。

最低限、以下の抽象単位を想定する。

- `DiscardPolicy`
- `CallPolicy`
- `RiichiPolicy`
- `WinPolicy`
- 必要に応じて `AbortiveDrawPolicy`

## 5.2 初期実装対象
- 初期実装で学習対象とするのは `DiscardPolicy` のみ
- 他の policy は未実装、または固定ルール / ルールベース処理とする

## 5.3 Action 抽象
学習側では、エンジンの `Action` をそのまま扱うのではなく、  
各 policy に対応した **学習対象 action 空間** を持てるようにする。

例:
- `DiscardPolicyAction`
- 将来の `CallPolicyAction`

---

## 6. Observation / FeatureEncoder / Model の分離

## 6.1 原則
Observation と Feature 表現と Model を密結合にしてはならない。  
最低限、以下の責務分離を持つ。

- **Observation**: エンジン由来のモデル非依存情報
- **FeatureEncoder**: Observation をモデル入力へ変換
- **Model**: 入力特徴量から policy/value を推論
- **ActionSelector**: policy 出力と legal mask から行動選択

## 6.2 Observation
Observation は `GAME_SPEC.md` に定義された `PartialObservation` / `FullObservation` を利用する。  
学習システムは両方を受け取れること。

## 6.3 FeatureEncoder
FeatureEncoder は差し替え可能にする。
最低限、以下を想定する。

- `FlatFeatureEncoder`
- `ChannelTensorEncoder`

将来的には以下を追加可能とする。

- `TokenSequenceEncoder`
- `HybridFeatureEncoder`

### 6.3.1 FlatFeatureEncoder のオプション特徴量

#### delta_shanten_sign (shanten_hint)

`feature_encoder.shanten_hint.enabled: true` で有効化。既定は `false`。
有効時、特徴ベクトル末尾に `delta_shanten_sign[34]` を追加する（+34次元）。

- 各打牌候補 t (0〜33) について、手牌から t を1枚除いた場合のシャンテン数変化を符号化する
- `base = compute_shanten(手牌)`, `after = compute_shanten(手牌 - t)`, `delta = base - after`
- `0.0`: 維持（最適打牌候補）または手牌に存在しない牌種
- `-1.0`: 悪化（シャンテン数が増加する打牌）
- `+1.0`: 定義上は改善だが、`shanten(n枚) <= shanten(n-1枚)` の数学的性質により **現行の discard 評価では発生しない**

この「+1 非発生」は shanten 関数の単調性に由来する。
将来 draw 評価やツモ牌選択など異なる文脈で同関数を流用する場合は再検討が必要。

## 6.4 Model
Model は FeatureEncoder 出力に依存してよいが、Observation に直接依存しないこと。

---

## 7. Stage 1 の正式学習対象

Stage 1 では以下を固定する。

- 学習対象行動: 自摸直後の打牌のみ
- 副露なし
- ロン / ツモ和了は自動
- テンパイ時自動立直
- 九種九牌なし
- 通常流局あり

学習システムは、この制約付き対局を扱えること。

---

## 8. Observation モード

## 8.1 対応モード
少なくとも以下の Observation モードをサポートする。

- `full`
- `partial`

## 8.2 推奨開始点
推奨開始点は `full` とする。  
ただし `partial` 実験も同一システム上で切り替え可能であること。

## 8.3 FullObservation の利用目的
- 学習基盤の立ち上げ
- 上限比較
- teacher 学習
- 蒸留

## 8.4 PartialObservation の利用目的
- 実戦想定学習
- 不完全情報下の評価
- 最終 deployment 対応

---

## 9. Feature 表現仕様

## 9.1 基本方針
Feature 表現は model ごとに差し替え可能とする。  
ただし、最低限の共通仕様を持つ。

## 9.2 初期対応形式
### 9.2.1 Flat Feature
- フラットな固定長数値ベクトル
- 手牌、河、副露、点数、局情報などを平坦化する
- MLP 系向け

### 9.2.2 Channel Tensor
- チャネル分割した固定テンソル
- 牌種情報、河、ドラ、局情報等をチャネル化する
- CNN 系向け

## 9.3 将来形式
- TokenSequence
- Hybrid

## 9.4 legal mask
Feature とは別に、合法打牌マスクを扱えること。  
Stage 1 では 34 種打牌ロジットに対する legal mask を標準とする。

---

## 10. モデル仕様

## 10.1 初期 policy head
Stage 1 の初期推奨実装は以下とする。

- **34 種牌ロジット出力**
- **legal mask による実行可能打牌への射影**

ただし将来の action scoring へ拡張可能な抽象化を持つこと。

## 10.2 初期 value head
value head は最初から持つ。  
複数 value head を追加可能な設計とする。

初期推奨:
- 局点差 value
- 半荘最終収支 value

将来的に追加可能:
- 半荘順位 value

## 10.3 モデル候補
初期候補として以下を想定する。

- `MLPPolicyValueModel`
- `CNNPolicyValueModel`

## 10.4 ONNX 配慮
モデルは将来 ONNX export 可能であることを推奨する。  
極端に特殊な演算や export 困難な構造へ依存しすぎないこと。

---

## 11. ActionSelector

## 11.1 責務
ActionSelector は、policy 出力と legal mask から実際の行動を決定する。

## 11.2 Stage 1
Stage 1 では少なくとも以下を扱う。

- argmax 選択
- サンプリング選択
- 探索ノイズ付き選択（任意）

## 11.3 将来拡張
将来は以下も扱えるようにする。

- MCTS 由来の policy 分布
- temperature 制御
- evaluation / training 切り替え

---

## 12. 学習データ仕様

## 12.1 基本単位
学習データの基本単位は **ステップサンプル** とする。  
ただし、局・半荘境界情報を保持する。

## 12.2 1 サンプルに含める情報
少なくとも以下を持つ。

- observation
- observation_mode
- feature_encoder_name（必要なら）
- legal action mask
- chosen action
- policy target（必要なら）
- immediate reward
- cumulative / target reward
- terminal flags
- actor_id
- episode_id
- round_id
- step_id
- model_version
- generation

## 12.3 補助情報
必要に応じて以下も持てる。

- baseline/opponent type
- experiment_id
- run_id
- timestamp
- seed
- notes / tags

---

## 13. replay buffer

## 13.1 役割
replay buffer は、self-play で生成された学習サンプルを蓄積し、学習に再利用するための層である。

## 13.2 初期方針
初期標準は **file-based shard を replay 的に利用する方式** とする。  
メモリ内 ring buffer を主要前提にはしない。

## 13.3 データ利用方針
- shard ファイル群から必要なサンプルを読み込む
- シャッフルしてミニバッチを作る
- recent data を優先する戦略を持てるようにする

## 13.4 将来拡張
必要なら、ローカル検証用にメモリ buffer を追加してよい。

---

## 14. shard file / 永続化仕様

## 14.1 基本方針
self-play データは shard file 単位で保存する。  
将来の複数 CPU / GPU サーバ運用を見据え、共有ストレージ前提で扱えること。

## 14.2 保存形式
保存形式は抽象化する。  
ただし初期推奨実装は **Parquet** とする。

## 14.3 shard の粒度
- ステップ単位サンプルをまとめて shard に格納する
- 1 ファイル 1 局や 1 半荘に限定しない
- shard サイズは設定可能とする

## 14.4 必須メタデータ
各 shard には少なくとも以下を持たせる。

- `experiment_id`
- `run_id`
- `worker_id`
- `shard_id`
- `model_version`
- `generation`
- `timestamp`
- `episode_id`
- `round_id`
- `step_id`

## 14.5 スキーマ方針
スキーマは後方互換を意識する。  
列追加に比較的耐えられる構造を推奨する。

---

## 15. Self-Play Worker 仕様

## 15.1 役割
Self-Play Worker は対局を生成し、学習サンプルを shard file として保存する。

## 15.2 入力
- game engine interface
- policy / baseline
- experiment config
- self-play config
- current model version

## 15.3 出力
- shard files
- worker logs
- 生成統計

## 15.4 対戦相手構成
self-play 相手構成は設定可能とする。  
推奨初期値:

- 学習中ポリシー: 0.5
- ルールベースベースライン: 0.5

## 15.5 Stage 1 のルールベースベースライン
最低要件:
- シャンテン数最小打牌

推奨目標:
- シャンテン数最小
- 同点なら受け入れ最大
- テンパイ時自動立直

## 15.6 対戦モード
少なくとも以下を許可する。

- policy vs policy
- policy vs baseline
- baseline mix
- checkpoint evaluation mode（将来）

---

## 16. Learner 仕様

## 16.1 役割
Learner は shard data を読み込み、policy-value model を学習し、checkpoint を生成する。

## 16.2 入力
- experiment config
- model config
- reward config
- training config
- shard data source

## 16.3 出力
- checkpoints
- optimizer states（必要なら）
- training metrics
- evaluation summary
- notes 更新情報

## 16.4 学習アルゴリズム
アルゴリズムは固定しすぎない。  
推奨は **PPO 系** とする。  
初期の軽い imitation / supervised warm start も許可する。

## 16.5 warm start
Stage 1 では、ルールベース打牌の軽い模倣を少し行った後に self-play へ入る方法を推奨する。

---

## 17. 評価システム

## 17.1 役割
評価システムは、checkpoint や実験設定ごとの性能比較を行う。

## 17.2 主指標
必ず集計すべき主指標:

- 平均順位
- 平均半荘収支
- 和了率
- 放銃率

## 17.3 補助指標
必要に応じて以下を集計する。

- 平均局収支
- テンパイ率
- 流局率
- policy entropy
- value loss
- self-play 生成速度
- プレイアウト速度
- 学習スループット

## 17.4 比較対象
- ルールベースベースライン
- 過去 checkpoint
- Full / Partial
- 同一設定の別 seed

---

## 18. 実験設定仕様

## 18.1 正本
実験設定の人間可読な正本は YAML とする。

## 18.2 初期運用
初期運用は **1 実験 1 YAML** を推奨する。  
将来的には base + override へ拡張してよい。

## 18.3 設定カテゴリ
少なくとも以下を設定可能にする。

- experiment
- observation
- feature_encoder
- model
- reward
- selfplay
- training
- evaluation
- export

## 18.4 memo
YAML 側に memo フィールドを持ってよい。  
実行時に notes へ転写可能であることを推奨する。

---

## 19. 実験ディレクトリ仕様

## 19.1 基本形式
実験成果物は run 単位で保存する。  
推奨ディレクトリ名:

- `runs/<date>_<name>_<id>/`

## 19.2 必須構造
各 run ディレクトリは少なくとも以下を持てるようにする。

- `config.yaml`
- `notes.md`
- `checkpoints/`
- `selfplay/`
- `eval/`

## 19.3 notes
- `notes.md` は実験ごとに持つ
- checkpoint 比較ログは別管理でよい
- YAML memo や主要設定を `notes.md` に転写できることを推奨する

---

## 20. Python / C++ 境界

## 20.1 基本方針
- ゲームエンジンは C++
- learner は Python
- 境界は pybind11 等を前提とする

## 20.2 Python から使う最小 API
少なくとも以下を Python 側から利用可能にすることを想定する。

- `reset()`
- `step()`
- `get_legal_actions()`
- `make_observation()`
- 状態複製（必要なら）
- self-play / evaluation に必要な補助 API

## 20.3 責務
- 対局進行・合法手・精算: C++
- 学習・optimizer・dataset 管理: Python

---

## 21. Full / Partial 学習と蒸留

## 21.1 位置づけ
FullObservation と PartialObservation は、同一基盤で比較可能であること。

## 21.2 推奨開始点
推奨開始点は FullObservation とする。

## 21.3 蒸留
Stage 2 候補として、**FullObservation teacher → PartialObservation student** を明示的にサポート可能な設計にする。

### 蒸留実験で必要な前提
- Observation と FeatureEncoder の分離
- teacher/student の別モデル保持
- 同一局面から Full / Partial を生成可能
- 実験設定で teacher/student を識別できること

---

## 22. モデル交換形式

## 22.1 初期
初期学習は PyTorch モデルを標準とする。

## 22.2 将来
将来的な C++ 高速推論の第一候補は **ONNX Runtime** とする。

## 22.3 設計上の推奨
- モデルは ONNX export 可能であることを推奨する
- ONNX 変換困難な演算への依存を避けすぎなくてよいが、最終的な export を常に意識する
- 必要なら試作段階で他方式を使ってもよいが、正式仕様上の第一候補は ONNX Runtime とする

---

## 23. 探索基盤との接続

## 23.1 役割
探索は RL システムの一部として扱えるようにするが、初期から MCTS を必須にしない。

## 23.2 初期探索
初期探索基盤では少なくとも以下を扱えることを想定する。

- プレイアウト速度計測
- 候補打牌比較
- 同一局面再評価の安定性確認

## 23.3 将来
- MCTS
- policy prior 利用
- value 利用
- search policy distillation
へ拡張可能であること

---

## 24. テスト要件

## 24.1 unit テスト
- FeatureEncoder
- legal mask 整合
- dataset row 変換
- reward 計算
- baseline policy
- config 読み込み

## 24.2 integration テスト
- Python から Environment を呼べる
- self-play worker が shard を書ける
- learner が shard を読める
- checkpoint 保存 / 読込ができる
- Full / Partial 切り替えが機能する

## 24.3 replay / reproducibility テスト
- 同一 seed + 同一 config で再現性がある
- shard メタデータが正しい
- experiment/run 情報が追跡可能

---

## 25. CHANGE_QUEUE 運用

RL 側の変更も `CHANGE_QUEUE.md` に統合して管理する。  
必要に応じて以下の Type を使ってよい。

- `RL`
- `Training`
- `Experiment`
- `Eval`

ゲームエンジン側と同様に、実装後にレビューし、レビュー完了した CQ は queue から削除する。

---

## 26. TODO

以下は将来拡張または追加検討項目とする。

- PartialObservation 本命運用時の特徴量最適化
- Full → Partial 蒸留の具体方式
- PPO 以外の学習アルゴリズム比較
- MCTS 併用学習
- 分散 learner
- ONNX Runtime 実推論導入
- 高速専用バイナリ保存形式
- token / transformer 系 encoder
- policy 分割の本格実装
- league training
- curriculum 学習

---