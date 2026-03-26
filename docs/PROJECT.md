# PROJECT.md

最終更新: 2026-03-27  
この文書の役割: このプロジェクトの**大目標・現在地・次の段階**を短時間で復元するための意思決定ハブ。  
実装詳細は `GAME_SPEC.md` / `RL_SPEC.md`、個別実験の条件と結果は `experiments/.../runbook.md` / `report.md` を正とする。

---

## 1. 北極星

最終目標は、**完全な日本式リーチ麻雀ルールで強いエージェントを作ること**。

ここでいう「強い」は、単に Stage 1 の簡略ルールで高スコアを出すことではなく、

- 行動を段階的に解放し
- ルールを段階的に実麻雀へ近づけ
- 最終的に PartialObservation でも使える形へつなぐ

ことを含む。

そのため、このプロジェクトでは

- 学習問題を段階的に立ち上げる
- 各段階で「何ができたか」「何がまだ未解決か」を残す
- 実験を積み上げ可能な形で管理する

ことを重視する。

---

## 2. この文書が担うこと

この文書は、次の判断をまとめる。

1. いまどの段階にいるか
2. 次にどこへ進むか
3. 何を当面やらないか
4. どの実験結果を現在の基準として使うか

逆に、次はここに書かない。

- 細かな engine 実装詳細
- learner / shard / worker の仕様詳細
- 各実験の全ログ
- コードを読めばすぐ分かる設定列挙

---

## 3. 学習段階の整理

### Stage01 `DiscardOnly`

- 学習対象は自摸直後の打牌のみ
- 副露は行わない
- `Ron / Tsumo / Riichi` は自動処理
- 最初の imitation / PPO 基盤を立ち上げ、観測・特徴量・報酬・更新則の不具合を洗い出す段階

### Stage02a `CallUnlock`

- `DiscardPolicy` は維持したまま、response phase の副露判断を追加する
- 最初に解放するのは
  - `Chi`
  - `Pon`
  - `Daiminkan`
  - `Skip`
- `Ron / Tsumo / Riichi / Ankan / Kakan / Kyuushu` は当面自動処理のまま
- 目的は「副露判断を含む学習が自然に立ち上がるか」を確認すること

### Stage02b 以降

- `Riichi`、和了判断、見逃し、加槓・暗槓などを段階的に追加する
- 必要なら FullObservation teacher → PartialObservation student の蒸留を行う

### Stage03 `FullAction`

- 打牌・副露・立直・和了・スキップを含む完全行動を対象とする
- ルール側の完全化と並行して進める

---

## 4. 現在地

2026-03-27 時点の現在地は、**Stage01 の主な問いには一通り答えが出て、Stage02a へ移る直前**である。

### Stage01 で分かったこと

- 以前の停滞は「PPO が本質的に伸びない」ことを示していたわけではなかった
- `danger_mask` など、防御に直結する学習余地を与えると、PPO は imitation を安定して上回れる
- したがって、Stage01 で残っていた本質的ボトルネックは、更新則そのものより **課題設定と情報量** にあった

### 現在の Stage01 best

現時点の Stage01 FullObservation の基準は、  
`experiments/Stage01_DiscardOnly/exp_070/report.md` の **`C context_plus_danger`**。

重要な要約値:

- `final mean = 6518.75`
- `cycle 20-29 mean = 6073.86`
- `drawdown mean = 133.25`

この結果により、

- `danger_mask` は Stage01 FullObservation では大当たり
- PPO は「学習余地があれば imitation を超えられる」
- Stage01 をこれ以上細かく掘るより、次の行動解放に進む方が情報価値が高い

と判断している。

---

## 5. いま固定してよい判断

### 5.1 Stage01 は「打ち切り」ではなく「回帰基準」

Stage01 は捨てない。  
ただし今後の主戦場ではなく、次の用途に使う。

- regression harness
- feature / PPO / diagnostics の sanity check
- FullObservation 上限比較

### 5.2 次の主戦場は Stage02a

次に確かめたいのは、

- 副露を解放しても学習が立ち上がるか
- `DiscardPolicy` と `CallPolicy` を分けた設計が自然に回るか
- 追加役込みで副露判断に意味が出るか

である。

### 5.3 最終目標は PartialObservation だが、直近は FullObservation を許容する

Stage02a の最初から PartialObservation に縛らない。  
まずは

- action 境界
- shard schema
- call policy
- outcome label

を自然に立ち上げることを優先する。

ただし、将来的に Partial へ移せない情報依存を増やしすぎないことは意識する。

---

## 6. Stage02a の合意仕様

詳細は `reference/stage2/README.md` と  
`reference/stage2/stage2a_call_policy_design.md` を参照。

ここでは、実装前提として固定した点だけを書く。

### 6.1 ルール範囲

Stage02a では、現行役に加えて次を追加する。

- `対々和`
- `一気通貫`
- `三色同順`

目的:

- `Pon / Daiminkan` に直接意味を持たせる
- `Chi` にも学習価値を持たせる

### 6.2 行動範囲

学習対象:

- `Discard`
- `Chi`
- `Pon`
- `Daiminkan`
- `Skip`

当面の自動処理:

- `Ron`
- `Tsumo`
- `Riichi`
- `Ankan`
- `Kakan`
- `Kyuushu`

### 6.3 モデル方針

- 打牌と副露は別ポリシーとして扱う
- ただし candidate ごとに trunk を回し直さない
- 最初は learned trunk を共有しない
  - `discard_trunk`
  - `call_trunk`
- `CallPolicy` は legal candidate ごとの scalar score を出す
- `Skip` も candidate の 1 つとして比較する

### 6.4 特徴量方針

- call 側は **response context + compact candidate feature** を使う
- `phase` one-hot は trunk に入れない
- `danger_mask` を call 側へ直接入れるのは当面やらない
- 副露後 discard を見た advanced summary は Stage02a v1 では後回し

### 6.5 baseline / imitation 方針

Stage02a v1 では actor / teacher を分けない。

- `RuleBasedCallPolicy.select_action()`
  だけを持ち
- baseline actor が実際に選んだ action を imitation ラベルとして使う

`abstain` を含む teacher 分離は、必要になったときに次段階で導入する。

### 6.6 outcome / yaku label

将来の `yaku head` や diagnostics に備え、Stage02a の sample/shard では
少なくとも将来的に次を持てる構造を選ぶ。

- `round_terminal_label`
- `eventual_win_yaku_ids`
- `eventual_total_han`
- `eventual_fu`

ここは最初から学習に使わなくてよいが、後で入れにくい設計にはしない。

---

## 7. 現在の実装順

Stage02a の実装は、`docs/CHANGE_QUEUE.md` の CQ 群に分けて進める。

推奨順:

1. 役追加と candidate 表現の露出
2. `Stage2Env` と shard/schema 拡張
3. rule-based call policy と selfplay / imitation データ生成
4. `DiscardPolicy` / `CallPolicy` 分離モデル
5. learner / selector / PPO / eval 統合
6. outcome / yaku summary
7. smoke config と最小 end-to-end 確認

---

## 8. 当面やらないこと

当面は、次を優先しない。

- Stage01 の細かいハイパラ再探索
- Stage01 用 feature flag / profile の大掃除
- call 側への `danger_mask` 直結
- call teacher の actor/teacher 分離
- 副露後の best discard を読む advanced call feature
- Stage02a の初手から PartialObservation に寄せること

これらは Stage02a が自然に回ってから再検討する。

---

## 9. 次に満たしたい条件

Stage02a の最初の通過条件は、強さよりまず **自然に学習が回ること**。

最低限ほしいこと:

1. `Stage2Env` で imitation-only が完走する
2. selfplay / eval が call decision を含んで完走する
3. call decision が shard に正しく保存される
4. `DiscardPolicy` と `CallPolicy` の両 branch が正しく更新される
5. 追加役を使った副露が baseline / selfplay 上で実際に発生する

この段階を越えたら、初めて

- Stage02a の imitation ceiling
- short PPO
- Full → Partial の接続

を評価する。

---

## 10. 更新ルール

この文書は、細かな実験ログ置き場ではない。  
更新するときは、次のどれかが変わったときだけに絞る。

1. 主系列の基準条件が変わった
2. 次の段階への判断が変わった
3. 当面やらないことが変わった
4. Stage 定義や実装順に本質的な変更が入った

数値の詳細比較は `experiments/.../report.md` に残し、ここには判断だけを残す。

---

## 11. 一文要約

**Stage01 では「学習余地があれば PPO は imitation を安定して超えられる」と確認できた。現在の主戦場は Stage02a であり、副露だけを先に解放した中間段階を、分離された `DiscardPolicy` / `CallPolicy` と新しい `Stage2Env` で立ち上げる。**
