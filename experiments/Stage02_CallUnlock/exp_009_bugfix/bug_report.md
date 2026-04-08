# Bug Report: Open-Hand Shanten / Baseline Teacher Inconsistency (exp_009_bugfix)

作成日: 2026-04-01  
起点実験: `experiments/Stage02_CallUnlock/exp_009_bugfix`  
状態: **修正・再確認済み**

## 1. 概要

Stage02a の副露学習を調査した結果、rule-based teacher は

- 副露 (`call`) 自体は大量に発生している
- にもかかわらず `round_terminal_label == "win_called"` をほぼ作れていない

ことが分かった。

原因を追うと、`RuleBasedBaseline` の discard 評価が

- concealed hand の 34 種カウントだけを用い
- 副露済み面子数を見ない closed-hand 前提の `compute_shanten()` / `acceptance` で
- open hand の discard を選んでいた

ことが主因だった。

さらに、この open-hand 前提の欠落は

- Stage2 selfplay / imitation teacher
- Stage2 evaluator の baseline seat
- `FlatFeatureEncoder` の `current_shanten` / `shanten_hint` / `discard_ukeire_hint`
- Python fallback 経路
- opponent `danger_mask`

にも波及していた。

## 2. 発見経緯

`exp_009_bugfix` の semantic auxiliary 実験を読んでいる最中に、次の不自然さが見つかった。

1. semantic diagnostic で `win_called` support が 0
2. imitation teacher data を直接数えても `win_called` が 0
3. しかし `call_count` 自体は大量に存在する

この時点で、

- 副露しないのではなく
- 副露しても open win に結びついていない

ことが疑われた。

その後、rule-based discard の実装を確認すると、open meld 数をまったく使っていないことが分かった。

## 3. 確認した事実

### 3.1 imitation teacher では `win_called` が 0 だった

`exp_009_bugfix` の `A1_semaux_default` imitation data を集計すると、

- `call` decision sample は大量に存在する
- それでも `win_called` は 0

だった。

つまり、teacher data は open-hand 成功例を事実上供給できていなかった。

### 3.2 RuleBasedBaseline は open meld 数を見ていなかった

`RuleBasedBaseline.select_discard()` は

- `hand_tile_ids`
- `legal_mask`

だけを入力にしており、副露済み面子数を受け取っていなかった。

内部で使う `compute_shanten()` / `find_best_discard()` も concealed hand の 34 種カウントだけを見ていたため、open hand を closed hand と同じ式で評価していた。

### 3.3 具体例で open tenpai が `2` と誤判定されていた

例として、

- open meld = 白ポン 1 面子
- concealed hand = `123m 456p 78s 55p`

のような、open hand では明らかな tenpai 形を入れると、現行 `compute_shanten()` は `0` ではなく `2` を返した。

これは、

- engine 側の和了 legality の問題ではなく
- baseline discard の評価が open hand を正しく見ていない

ことを直接示していた。

### 3.4 engine の Ron/Tsumo 自体が壊れている証拠はなかった

Stage2 では Ron / Tsumo は env が legal action を見て自動実行する。  
そのため、「シャンテン数の誤判定のせいで本来和了なのに Ron/Tsumo できない」というより、

- open hand の途中経路で誤った discard を選ぶ
- その結果 open win に至らない

と考える方が自然だった。

## 4. 影響範囲

今回の不整合は、少なくとも以下に影響していた。

### 4.1 rule-based teacher

- `Stage2SelfPlayWorker` の baseline discard
- teacher top1
- teacher best-set

### 4.2 評価

- `Stage2aEvaluator` の baseline seat discard

### 4.3 policy 入力特徴

- `current_shanten`
- `shanten_hint`
- `discard_ukeire_hint`
- opponent `current_shanten`
- opponent `danger_mask`

### 4.4 Python fallback

C++ analyze 非使用時の fallback 経路でも、当初は `meld_count` が落ちていた。

## 5. 修正内容

今回の bugfix は段階的に行った。

### 5.1 CQ-0259

open-hand shanten / acceptance の本体修正。

- `compute_shanten(counts, meld_count)` を導入
- `meld_count > 0` では chiitoi / kokushi を混ぜず、open-hand regular shanten を使う
- `find_best_discard()` / `analyze_discards()` / `acceptance` に `meld_count` を伝播
- `RuleBasedBaseline` の API に `meld_count` を追加
- `Stage2SelfPlayWorker` の baseline discard と teacher で `len(player.melds)` を渡す

この修正後、small selfplay smoke で `win_called` が出現することを確認した。

### 5.2 CQ-0260

open-hand semantics を evaluator / encoder の主要経路へ拡張。

- `Stage2aEvaluator` の baseline seat discard に `meld_count` を渡す
- `FlatFeatureEncoder` の
  - `current_shanten`
  - `shanten_hint`
  - `discard_ukeire_hint`
  - opponent `current_shanten`
  に open-hand `meld_count` を反映

### 5.3 CQ-0261

`FlatFeatureEncoder` の残件修正。

- Python fallback でも `meld_count` を伝播
- opponent `danger_mask` の待ち牌探索でも `meld_count` を使う

これで

- C++ 経路
- Python fallback 経路
- opponent 側の open-hand 判定

まで一通り揃った。

## 6. 修正後に確認できたこと

### 6.1 `win_called` は smoke で出現するようになった

修正前は、確認した small smoke / imitation teacher 集計で `win_called` が 0 固定だった。  
修正後は Stage2 selfplay smoke で `win_called` が実際に出現することを確認した。

これは、少なくとも

- open-hand 和了が engine 上は可能であり
- teacher 側の discard 選択がその方向へ進めるようになった

ことを示す。

追加で、`.venv` の通常実行経路でも 40 match の Stage2 selfplay smoke を回し、`win_called` が実際に生成されることを再確認した。

- `config={}`
  - `call_count = 5042`
  - `overall win_called = 2342`
  - `call rows の win_called = 524`
  - `has_win_called = True`
- `exp_009_bugfix A1` 相当 config
  - `call_count = 5042`
  - `overall win_called = 2342`
  - `call rows の win_called = 524`
  - `has_win_called = True`

ここでの `win_called` は局数ではなく sample row 数である。  
ただし、少なくとも「修正後も `win_called` が 0 のまま」という状態ではないことは明確に確認できた。

### 6.2 C++ 拡張の反映漏れも併発していた

`win_called` 再確認の途中で、Python source は最新なのに `.venv` が古い `_mahjong_core` を読んでいることも見つかった。

- Python package: source tree (`python/mahjong_rl`) を参照
- C++ 拡張: `.venv/lib/python3.10/site-packages/mahjong_rl/_mahjong_core...so` を参照

そのため、C++ 側の bugfix が入っていても、再ビルドと `.venv` 側バイナリの更新をしない限り通常実行には反映されない状態だった。

今回は

- `cmake --build build --target _mahjong_core`
- `.venv` 側 `_mahjong_core` を `build/_mahjong_core...so` への symlink に差し替え

を行い、その後に `.venv` の通常実行で `meld_count` 対応シグネチャと `win_called` 生成を確認した。

### 6.3 open-hand tenpai / iishanten の具体例は正しい値になった

以前 closed-hand 値 `2` になっていた open tenpai の具体例は、修正後 `0` になった。  
また、open iishanten / 2副露 / 3副露のケースも unit test で押さえた。

### 6.4 encoder と evaluator も同じ semantics に揃った

これにより、

- teacher が open-hand semantics で打つ
- evaluator の baseline seat も同じ semantics で打つ
- policy 入力の shanten 系 hint も同じ semantics を使う

という最低限の整合が取れた。

## 7. 何が言えるか

この bug は、単に「シャンテン数が少しズレていた」ではない。

Stage02a の副露学習において、teacher / evaluator / feature が一貫して open hand を誤評価していたため、

- imitation が `win_called` 成功例をほぼ作れない
- semantic trunk が open-hand semantic を学びにくい
- Optional branch の改善が見えにくい

という大きな歪みを作っていた可能性が高い。

したがって、`exp_009_bugfix` 以前の Optional 学習の弱さは、

- reward の疎さ
- value / semantic trunk の弱さ

だけでなく、**teacher と feature の open-hand バグ**も強く絡んでいたと考えるべきである。

## 8. 残る注意点

現時点で major な open-hand shanten 不整合は一通り修正済みだが、今回の bugfix はまず

- rule-based teacher
- evaluator baseline
- shanten 系 hint

を直した段階である。

ここから先は改めて、

1. `exp_009_bugfix` 条件を再実行する
2. semantic trunk が `win_called` / open yaku を本当に学べるか診断する
3. 必要なら rule-based call policy 自体の弱さも別で見る

という順で進めるのが自然である。

## 9. 次アクション

1. `exp_009_bugfix` 条件を再実行する
2. semantic head diagnostics を再度取り、`win_called` support が復活した状態で比較する
3. そのうえで multi-seed 化するか判断する
