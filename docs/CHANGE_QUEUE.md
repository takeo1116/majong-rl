# CHANGE_QUEUE.md

このファイルは **未反映の変更要求だけを置く作業キュー** である。  
履歴台帳ではなく、いま実装・レビューすべき項目だけを管理する。

- `Status: [Proposed]`: 未実装
- `Status: [Implemented]`: 実装済み、レビュー待ち

## 運用ルール

- 新しい CQ は末尾に追記する
- 項目順は並べ替えない
- 実装者は CQ を削除しない
- レビュー完了した CQ はレビュアーが削除する
- Claude Code が編集してよいのは原則として以下のみ
  - `Status` の更新
  - `実装メモ` への短い追記
  - 明確な誤字修正
- 仕様の議論や設計メモは、このファイルに長く書かず `PROJECT.md` / `GAME_SPEC.md` / `RL_SPEC.md` / `reference/stage2/` に置く
- `PROJECT.md` / `GAME_RULE.md` / `GAME_SPEC.md` / `RL_SPEC.md` / 実装は最終的に整合していなければならない

## テンプレート

### CQ-XXXX
- Status: [Proposed]
- Type: Rule | Engine | RL | Training | Eval | Test | Docs | IO
- Priority: High | Medium | Low
- Title: ここに短い変更タイトルを書く

#### 背景
なぜこの変更が必要かを書く。  
既存仕様や既存実装との関係があれば簡潔に書く。

#### 要求内容
実装してほしい変更内容を具体的に書く。  
必要なら箇条書きで列挙する。

#### 関連文書
- PROJECT.md: 段階判断や優先順位が関係する場合に書く
- GAME_RULE.md: 該当セクションがあれば書く
- GAME_SPEC.md: 該当セクションがあれば書く
- RL_SPEC.md: 該当セクションがあれば書く
- その他: 任意

#### 受け入れ条件
- 変更後に満たしてほしい条件を書く
- テストで確認可能な形が望ましい

#### 実装メモ
必要なら補足を書く。

---

## 変更要求一覧

### CQ-0272
- Status: [Proposed]
- Type: RL
- Priority: High
- Title: tile_presence_flags を feature_encoder config で on/off 切替可能にする

#### 背景
`CQ-0270` で追加した `tile_presence_flags` は現在 always-on で encoder に含まれている。  
`exp_016` では `yakuflags` あり条件の policy が `exp_015` baseline より悪化した一方で、特徴量アイデア自体を棄却するには早い。  
次の `exp_017` では

- `yakuflags なし`
- `yakuflags あり`
- trunk 幅そのまま / 拡張

を同一コードベースで比較したい。

#### 要求内容
`FlatFeatureEncoder` の `tile_presence_flags` を config flag 化する。

- `feature_encoder.tile_presence_flags.enabled: true/false` を追加する
- `false` のとき
  - `tile_presence_flags` を encode 出力に含めない
  - `metadata.feature_ranges` にも出さない
  - output dim も旧 `exp_015` 相当に戻る
- `true` のときは現行 `CQ-0270` と同じ動作にする
- full / partial 両 path で self 基準を維持する
- runner / tests / model input dim 計算がこの flag に追従するようにする

#### 関連文書
- RL_SPEC.md: feature_encoder / Stage2a 実験条件に関係
- その他: `experiments/Stage02_CallUnlock/exp_015/report.md`
- その他: `experiments/Stage02_CallUnlock/exp_016/runbook.md`

#### 受け入れ条件
- `tile_presence_flags=false` で `exp_015` 相当の observation dim になる
- `tile_presence_flags=true` で現行 `exp_016` 相当の observation dim になる
- `feature_ranges` に `tile_presence_flags` が出るのは enabled 時のみ
- full / partial encode が両設定で壊れない
- Stage2a config validation / model build / learner tests が通る
- `exp_017` で `yakuflags` on/off の条件を config override だけで切り替えられる

#### 実装メモ
`exp_017` では

- baseline (`tile_presence_flags=false`)
- yakuflags (`tile_presence_flags=true`)

を trunk 幅変更あり/なしで比較する予定。

---
