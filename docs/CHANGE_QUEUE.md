# CHANGE_QUEUE.md

このファイルは **未反映の変更要求キュー** である。  
ここに記載された `Status: [Proposed]` の項目が、実装対象となる。

- `Status: [Proposed]` : 未実装
- `Status: [Implemented]` : 実装済み

## 運用ルール

- レビュー完了した項目は **レビュアーが削除する**
- 項目順は **並べ替えない**
- 実装完了後は、該当項目の `Status` を `[Implemented]` に更新する
- Claude Code が編集してよいのは原則として以下のみ
  - `Status` の更新
  - `実装メモ` への短い追記
  - 明確な誤字修正
- 実装者は CQ を削除しない（削除はレビュアーのみ）
- 仕様未確定事項はこのファイルで議論せず、対応する `GAME_SPEC.md` / `RL_SPEC.md` 側で管理する
- `GAME_RULE.md` / `GAME_SPEC.md` / `RL_RULE.md` / `RL_SPEC.md` / 実装は最終的に整合していなければならない

## テンプレート

以下のテンプレートをコピーして追記すること。

### CQ-XXXX
- Status: [Proposed]
- Type: Rule | Engine | RL | Test | Docs
- Priority: High | Medium | Low
- Title: ここに短い変更タイトルを書く

#### 背景
なぜこの変更が必要かを書く。  
既存仕様や既存実装との関係があれば簡潔に書く。

#### 要求内容
実装してほしい変更内容を具体的に書く。  
必要なら箇条書きで列挙する。

#### 関連文書
- GAME_RULE.md: 該当セクションがあれば書く
- GAME_SPEC.md: 該当セクションがあれば書く
- RL_RULE.md: 該当セクションがあれば書く
- RL_SPEC.md: 該当セクションがあれば書く
- その他: 任意

#### 受け入れ条件
- 変更後に満たしてほしい条件を書く
- テストで確認可能な形が望ましい

#### 実装メモ
- `_selfplay_worker_fn` の stats.json に `match_index_start`, `match_index_end`, `first_match_seed`, `last_match_seed` を追加
- `match_seeds` が渡された場合（parallel 経路）のみ記録される

---

## 変更要求一覧
