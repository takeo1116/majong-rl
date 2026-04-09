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

### CQ-0273
- Status: [Implemented]
- Type: RL
- Priority: High
- Title: tile_presence_flags を semantic/value trunk 限定入力にする

#### 背景
`CQ-0270` で追加した self tile-presence flags

- `self_has_honor`
- `self_has_terminal`
- `self_has_simple`
- `self_has_man`
- `self_has_pin`
- `self_has_sou`

は、`exp_016` では shared encoder input への常時追加としては baseline 更新に失敗した。  
一方 `exp_017` では、

- `yakuflags on + narrow` は悪い
- `yakuflags on + wide` では policy と diagnostics がかなり回復
- 特に `Tanyao` の `mean_p / hit@0.2` は大きく改善

という結果になり、特徴量アイデア自体は有望だが、**raw feature を policy trunk まで直接流しているのが重い**可能性が高くなった。

現状の Stage2a では、

- `discard_trunk`
- `optional_trunk`
- `value_trunk`

が分かれており、`terminal / yaku / value` は value 側でまとまっている。  
このため、次は tile_presence_flags を **semantic/value 側には入れるが、discard/optional の raw policy 入力には直接入れない** 条件を試したい。

#### 要求内容
Stage2a で `tile_presence_flags` を semantic/value trunk 限定で使えるようにする。

具体的には:

- encoder は従来どおり `tile_presence_flags` を出してよい
- ただし model 側で
  - `value_trunk` には tile_presence_flags を含む full feature を入れる
  - `discard_trunk` と `optional_trunk` には tile_presence_flags を除いた feature を入れる
- semantic summary 経由の影響は従来どおり許す
  - つまり policy は raw flag を見ないが、semantic summary 経由では影響を受けうる

切替は config でできるようにする。

推奨:

- `model.semantic_aux.tile_presence_flags_semantic_only: true/false`
  - `false`: 現行どおり raw で全 trunk に入る
  - `true`: value/semantic 側のみ raw 入力に残し、discard/optional からは除外

#### 関連文書
- RL_SPEC.md
- `experiments/Stage02_CallUnlock/exp_016/report.md`
- `experiments/Stage02_CallUnlock/exp_017/report.md`
- `reference/stage2/stage2a_semantic_aux_trunk_design.md`

#### 受け入れ条件
- `tile_presence_flags_semantic_only=false` で現行 `CQ-0270/0272` と同一挙動になる
- `tile_presence_flags_semantic_only=true` のとき:
  - `value_trunk` 入力には `tile_presence_flags` が残る
  - `discard_trunk` / `optional_trunk` の raw 入力からは `tile_presence_flags` が除外される
  - semantic summary の経路は壊れない
- full / partial とも feature range の意味は変えない
- config summary / notes / model feature dump から mode が確認できる
- model smoke / runner / learner の既存テストが通る

#### 実装メモ
- 今回の狙いは「特徴量を削除すること」ではなく、「raw policy trunk への直接流入を止めること」
- `exp_018` ではこの mode を narrow / wide で比較する想定

---
