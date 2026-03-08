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

### CQ-0129
- Status: [Implemented]
- Type: RL | Performance | Test
- Priority: Medium
- Title: baseline 打牌選択の fast path を追加し、best-set 導入後の余分な計算コストを抑える

#### 背景
CQ-0128 で定義整合のため `select_discard()` が `select_discard_with_best_set()` に委譲される構造になった。  
これにより、best-set が不要な経路でも常に同率候補収集（2パス）が走る可能性がある。

現状は正しさ優先で問題ないが、self-play / evaluation は baseline 呼び出し回数が多いため、CPU 時間の増加要因になりうる。  
診断用途の出力互換を維持したまま、不要時は従来相当の軽量経路に戻す改善を行いたい。

#### 要求内容
- `RuleBasedBaseline` に「best-set 不要時の fast path」を追加する。
  - `save_baseline_actions=false` など、teacher_best_mask を保存しない経路では追加コストを避ける。
  - `save_baseline_actions=true` の経路は現状通り best-set を計算・保存する。
- strict top-1 と best-set の定義整合は維持する。
  - 評価規則の重複実装による将来ドリフトを避ける構造を維持すること。
- 既存の診断指標（`teacher_top1_match_rate`, `teacher_best_set_hit_rate`, `teacher_best_set_status`）出力は壊さない。

#### 関連文書
- RL_SPEC.md: imitation / baseline 教師データ / 診断指標
- その他: `python/mahjong_rl/baseline/rule_based.py`, `python/mahjong_rl/selfplay_worker.py`, `tests/python/test_baseline.py`, `tests/python/test_runner.py`

#### 受け入れ条件
- best-set 保存が不要な経路で、不要な best-set 計算が行われないことをテストで確認できる。
- best-set 保存が必要な経路では、従来どおり `teacher_best_mask` が保存される。
- CQ-0128 で追加した summary/notes の可観測性を壊さない。
- 既存の smoke/core/full テスト運用を壊さない。

#### 実装メモ
- 仕様変更ではなく性能改善。学習アルゴリズム・教師定義は変更しない。
- `_find_best_score()` を共通内部メソッドとして抽出し、評価規則（シャンテン最小→受け入れ最大）を一本化
- `select_discard()`: `_find_best_score()` + 最初の一致候補を返す（1パスのみ、best_mask 未生成）
- `select_discard_with_best_set()`: `_find_best_score()` + 全同率候補を収集（2パス目）
- `selfplay_worker._baseline_step()` は変更不要（既に `save_baseline_actions` で分岐済み）
- テスト: fast path が `select_discard_with_best_set` を呼ばないことを mock で検証

### CQ-0134
- Status: [Implemented]
- Type: RL | Test | Docs
- Priority: High
- Title: imitation mode 別集約の None 混在クラッシュを修正し、追跡仕様を明文化する

#### 背景
CQ-0133 実装後のレビューで、後方互換性に関わる実害リスクが見つかった。  
`batch_report` では mode 別集約を `sorted(by_mode.items())` で生成しているが、`imitation_loss_mode` が `None` の run（旧成果物や中間成果物）と `str` の run が混在すると、Python の比較不能により `TypeError` で batch 集約が落ちる可能性がある。

再現イメージ:
- run A: `imitation_metrics.imitation_loss_mode = None`
- run B: `imitation_metrics.imitation_loss_mode = "strict_top1"`
- 上記混在の `results` で `generate_batch_report()` 実行
- `sorted()` 時に `None < "strict_top1"` 比較が発生し失敗

この挙動は「既存成果物との混在でも batch 集約は落ちない」という運用期待に反するため、明確に修正する。

#### 要求内容
- `imitation_by_loss_mode` 集約時の mode 正規化ルールを固定する
  - `None` / 空文字 / 未設定を同一カテゴリ（例: `"unknown"` または `"strict_top1"`）へ正規化
  - 正規化カテゴリ名は仕様として明記し、実装・テスト・文書で一致させる
- 並び順生成で型混在例外を起こさないようにする
  - `sorted` による直接比較で `NoneType` と `str` を混在させない
  - 必要なら sort key で明示的に文字列化する
- 既存互換を維持する
  - 既存 `aggregate.imitation` はそのまま維持
  - 既存 CSV 列構成を壊さない
  - `runs[*].imitation_metrics` の既存キーを削除しない
- 文書追記
  - `RL_SPEC.md` に `aggregate.imitation_by_loss_mode` の存在と mode 正規化ルールを追記
  - runbook/report での解釈注意（unknown mode をどう扱うか）を短く明記

#### 関連文書
- RL_SPEC.md: imitation metrics / batch 集約キー仕様
- その他: `python/mahjong_rl/batch_report.py`, `tests/python/test_batch_report.py`

#### 受け入れ条件
- `imitation_loss_mode=None` と `imitation_loss_mode="strict_top1"` が混在する入力で `generate_batch_report()` が成功する
- `batch_summary.json.aggregate.imitation_by_loss_mode` が生成され、該当 run が正規化ルールどおりの mode 群に入る
- 既存テスト（single mode / mixed mode）は通過し、新規の `None` 混在ケースのテストが追加される
- 既存 `aggregate.imitation` の出力仕様は変わらない
- `RL_SPEC.md` に mode 正規化ルールと `imitation_by_loss_mode` の参照先が記載される

#### テスト方針
- `test_batch_report.py` に最小 2 ケース追加
  1. `None + strict_top1` 混在でクラッシュしない
  2. `None + tie_aware_best_set` 混在で mode 別 count が期待通り
- 既存 CQ-0133 の mode 混在テストと合わせて、後方互換（既存キー維持）も回帰確認する

#### 実装スコープ最小化方針
- 学習ロジック（learner / loss 計算）には触れない
- 変更対象は batch 集約の mode 正規化とそのテスト、文書追記に限定する
- 既存成果物キーの互換を最優先し、追加キーは `imitation_by_loss_mode` 系のみとする
- `batch_report.py`: `None`/空文字/未設定を `"unknown"` に正規化してから `sorted()` 実行
- `RL_SPEC.md` §17.6 に mode 正規化ルールと `imitation_by_loss_mode` の解釈注意を追記
- テスト: None+strict_top1 混在、None+tie_aware 混在の 2 ケース追加
