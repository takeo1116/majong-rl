# PROJECT.md

最終更新: 2026-03-08  
この文書の役割: プロジェクトの大目標・現在地・次アクションを短時間で復元する意思決定ハブ。

---

## 1. 北極星（不変）

最終目標は、麻雀AIの学習基盤を段階的に拡張し、最終的に実戦ルールに近い環境で強い方策を学習できる状態に到達すること。  
短期的には、**imitation 初期方策を PPO で壊さず改善する条件の確立**が最優先。

---

## 2. 現在地（2026-03-08）

フェーズ: **Stage1後半（診断フェーズ）**

- できている
  - runbook/report/driver 運用、phase 再利用、resume
  - imitation 学習の成立
  - PPO ハイパラ基礎探索
  - `shanten_hint` と `tie_aware_best_set` の導入・比較
  - learner 診断統計（`ppo_diag`）の成果物化
- 未解決
  - warm start + PPO での平均悪化（`eval_before -> eval`）
  - 改善阻害要因が「更新強度」か「更新方向/target品質」かの確定

---

## 3. Current Focus（今の実験目的）

現在の主目的は次の1点。

> PPO が imitation 初期方策を壊す理由を、統計で切り分ける。

現行固定軸（診断基準点）:

- `feature_encoder.shanten_hint=true`
- `training.imitation_loss_mode=tie_aware_best_set`
- `training.lr=0.0001`
- `training.epochs=4`
- `training.value_loss_coef=0.25`
- `training.clip_epsilon=0.2`
- `training.batch_size=256`
- `training.gamma=0.99`
- `training.gae_lambda=0.95`

評価の優先順:

1. `eval_before -> eval` の delta（悪化幅）
2. after 指標（`avg_rank`, `avg_score`, `win_rate`, `deal_in_rate`）
3. learner 診断統計（`advantage/return/value_error/ratio/clip_fraction`）
4. self-play / reward 分布

---

## 4. これまでの意思決定ログ（Decision Log）

短く履歴を残し、同じ探索を繰り返さないための要約。

- `exp_005〜013`: PPO ノブ探索  
  - 採用寄り: `lr=1e-4`, `epochs=4`, `value_loss_coef=0.25`, `clip=0.2`, `batch_size=256`, `gae=0.95`
  - ただし PPO 後悪化は残存
- `exp_014`（imitation-only, shanten_hint）  
  - on は小幅改善も決定打なし
- `exp_015`（warm start + PPO, shanten_hint）  
  - on は総合で悪化、単純採用見送り
- `exp_016`（教師再現率診断）  
  - shanten_hint は teacher 再現には効く
- `exp_017`（大型比較: strict vs tie-aware）  
  - imitation-only では tie-aware 優位
  - PPO込みでは優位が縮小、明確勝者なし
- `exp_018`（更新強度比較）  
  - baseline（`epochs=4, lr=1e-4`）を維持
  - `epochs=2` は明確に悪化、`lr=5e-5` は learner 診断統計を穏やかにするが主評価では更新できず
  - PPO 悪化の主因は更新強度だけではなく、reward / target 品質側も疑う段階へ進んだ

---

## 5. Do Not Re-test Yet（再検証保留）

以下は「今の実装条件のまま」では優先度を下げる。

- `shanten_hint` の on/off 単純比較（PPO込みで on 採用根拠が弱い）
- `value_loss_coef=1.0`（悪化傾向）
- 極端な更新強度（例: epochs 過大、clip 過大）
- `epochs=2` の weak-epochs 比較（baseline より明確に悪化）

再検討条件:

- reward 設計や target 定義を変更したとき
- ルール拡張で学習分布が変わったとき

---

## 6. 可観測性（いま見えるもの）

run 成果物で確認可能:

- `eval_before/eval/eval_diff`（rotation対応）
- self-play 統計（wins/deal-ins/draws/tsumo/ron/ryukyoku/num_rounds）
- learner 診断統計 `ppo_diag`
  - `advantage_*`, `return_*`, `old_value_*`, `value_error_*`, `ratio_*`, `clip_fraction`
- batch 側 run 別 learner 診断統計

不足が出たら CQ 化して可観測性を先に上げる（推測で進めない）。

---

## 7. Trigger to Move Stage（次段へ進む条件）

「PPO 改善が回る」と判定する目安（暫定）。

必須条件:

1. 主比較条件で `eval_before -> eval` の悪化が消える（少なくとも 5 seeds で一貫して改善または非悪化）
2. learner 診断統計が安定（ratio tail / clip_fraction / value_error が過大でない）
3. 追加1回の再現実験でも同傾向が出る

達成後の次段:

- ゲームルールを段階的に実麻雀へ拡張
- 各拡張ごとに runbook を分離し、劣化点を局所化
- 「実麻雀への近似」と「学習安定」の両立を判定基準にする

---

## 8. 運用原則

1. 実験は runbook 定義→レビュー→実行→report の順で進める  
2. 実行は原則 `scripts/local/exp_XXX_driver.py` で自動化する  
3. `run_map.json` はローカル管理、最終対応表は `report.md` に残す  
4. レビューNG時は旧CQを削除し、新規CQで再定義する  
5. 仕様変更（CQ）と実験解釈（report）を混ぜない

---

## 9. 更新ルール（この文書）

各実験完了後に最低限更新する:

1. `Current Focus`（必要なら）
2. `Decision Log`（1〜2行追加）
3. `Do Not Re-test Yet`（採否変化があれば）
4. `Trigger to Move Stage`（判定条件が変われば）

詳細な数値表は `experiments/exp_XXX/report.md` に置き、この文書は判断軸だけを保つ。

---

## 10. 一文要約

**imitation 改善は成立。次の勝負は PPO 悪化の主因（強度か方向か）を診断統計で確定し、改善条件を固定したうえでルール拡張フェーズへ移行すること。**
