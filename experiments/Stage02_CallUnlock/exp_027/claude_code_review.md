# Stage2a CallUnlock 直近実験 独立レビュー (exp_027)

作成日: 2026-05-07
作成者: Claude Code (Anthropic) / model: Claude Opus 4.7 (1M context)
対象: `experiments/Stage02_CallUnlock/exp_027` および周辺。
   CQ-0285 後に性能が落ちた理由・terminal signal が性能に効いていた機序・
   次のアクションについて、ソースコードと実験結果を独立に分析する。
参照ソース:

- `python/mahjong_rl/stage2a_learner.py`
- `python/mahjong_rl/models/stage2a_model.py`
- `python/mahjong_rl/stage2_selfplay_worker.py`
- `python/mahjong_rl/runner.py`
- `python/mahjong_rl/outcome_vocab.py`
- `configs/stage2a_core_minimal_mixed_s1_baseline.yaml`
- `docs/CHANGE_QUEUE.md` (CQ-0282/0283/0284/0285)
- `experiments/Stage02_CallUnlock/exp_026/report.md`
- `experiments/Stage02_CallUnlock/exp_027/report.md`
- `experiments/Stage02_CallUnlock/exp_027/runbook.md`
- `experiments/Stage02_CallUnlock/exp_025/claude_code_review.md`
- 重要 run の `summary.json` (gradnorm 系 4 本 + exp_027 3 本)

---

## 0. 結論先出し (3 行)

> 1. **CQ-0285 前の terminal gradient 支配は「scale バグ」と「性能の主因」を同時に
>    やっていた**。loss 定義は確かに不整合だったが、その大きな gradient が
>    *value_trunk を terminal class 識別器として育てる* 主信号で、
>    その結果 `semantic_summary` がpolicyに有用な outcome 表現を渡していた。
> 2. **CQ-0285 自体は数学的に正しいが、それだけだと value_trunk への有効
>    auxiliary 信号が約 30 倍弱まり**、`reward.point_delta_scale=0.0001` で
>    元から極小な value_loss と並んで、value_trunk 上の学習信号が全体的に
>    薄くなった。COEF50x で底上げしても、terminal が他成分を圧倒する比に
>    まで戻らないため exp_026 baseline には届かない。
> 3. 次の 1seed probe は **「CQ-0285 の式は維持して、`terminal_loss_coef`
>    だけ ~30 倍に上げる (yaku/value は base のまま)」** を最優先で当てる。
>    これで exp_026 と同等以上が出れば、機序は terminal-driven trunk shaping
>    で確定する。出ない場合のみ構造側 (semantic_proj が dead weight である件
>    などを含む) を疑う。

---

## 1. 観察した事実の整理

### 1.1 Performance ladder

3-seed まで取れたのは exp_026 だけ。残りは seed42 1seed なので、絶対値より
*差分のサイズと方向* に注目する。

| condition | final | best | best10 | tail10 | tail20 | win | deal |
|---|---:|---:|---:|---:|---:|---:|---:|
| **exp_026 seed42 base (target)** | **1.970** | **1.970** | **2.124** | **2.124** | **2.137** | 0.2368 | 0.1675 |
| CQ-0285 base coef (60c) | 2.295 | 2.165 | 2.270 | 2.374 | 2.361 | 0.2576 | 0.1842 |
| COEF10x | 2.515 | 2.155 | 2.275 | 2.472 | 2.388 | 0.2772 | 0.1743 |
| COEF50x | 2.245 | 2.055 | 2.220 | 2.292 | 2.259 | 0.3046 | 0.1822 |
| COEF100x | 2.300 | 2.105 | 2.227 | 2.275 | 2.253 | 0.2485 | 0.1887 |

要点:

- CQ-0285 のみで `final` は `1.970 → 2.295` (≈ +0.32) 悪化。
- COEF50x で部分回復し `2.245`、それでも exp_026 baseline までは ≈ +0.27 残る。
- COEF100x は COEF50x より明確には伸びず、頭打ち。
- `final win_rate` は逆に CQ-0285 系のほうが高い (0.24 → 0.30)
  → policy は和了確率を上げているが順位は伸びない。**点数効率が落ちている**。

### 1.2 Late-cycle diagnostics

| condition | entropy | clip | max_prob | log_ratio_p01 | ratio_max | T/Y | T/V | Y/V |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp_026 base | 0.2531 | 0.0591 | 0.8981 | -0.3222 | 8.89 | n/a | n/a | n/a |
| CQ-0285 base | 0.2056 | 0.0577 | 0.9180 | -0.3460 | 9.08 | 6.75 | 7.15 | 1.07 |
| COEF10x | 0.1917 | 0.0605 | 0.9222 | -0.3673 | 6.52 | 5.96 | 8.23 | 1.39 |
| COEF50x | 0.1675 | 0.0611 | 0.9319 | -0.4150 | 20.20 | 7.15 | 7.50 | 1.05 |
| COEF100x | 0.1626 | 0.0592 | 0.9356 | -0.4133 | 8.68 | 7.43 | 7.44 | 1.00 |

要点:

- coef を上げるほど **entropy 下がり / max_prob 上がり** で、policy は鋭くなる
  方向。これは PPO ratio の崩壊ではなく、policy がより確信的になる方向の変化。
- `T / Y` は CQ-0285 後はほぼ 6〜7 で coef 倍率に依存しない。
  つまり 4 条件すべて gradient 比は近似同じ → **比は coef sweep では動かせない**。
- 一方、CQ-0285 前 (gradnorm probe) では `T / Y` が 128〜318、
  `T / V` が 238〜1344 と桁違いに大きかった。
  → coef sweep 範囲 (10x〜100x) ではこの比に絶対届かない。

### 1.3 Loss magnitude (late, value_semantic group)

| condition | terminal_loss | yaku_loss | value_loss | weighted_terminal | weighted_yaku | weighted_value |
|---|---:|---:|---:|---:|---:|---:|
| CQ-0285 base | 0.854 | 0.132 | 0.00927 | 0.108 | 0.016 | 0.016 |
| COEF10x | 0.827 | 0.112 | 0.00698 | 1.05 | 0.18 | 0.13 |
| COEF50x | 0.880 | 0.131 | 0.01042 | 5.95 | 0.83 | 0.82 |
| COEF100x | 0.904 | 0.138 | 0.01125 | 11.77 | 1.59 | 1.72 |

要点:

- raw `terminal_loss` (≈ 0.9) / `yaku_loss` (≈ 0.13) / `value_loss` (≈ 0.01)
  はほぼ条件不変。これは coef sweep が learning dynamics を大きく変えるほど
  には効いていない (= Adam + grad clip でほぼ相殺) 証拠。
- weighted 表示はそれぞれ |coef| 倍されるが、**比は coef でしか変わらない**。

### 1.4 CQ-0285 前後の terminal scale

CQ-0285 5-cycle probe比較:

```text
terminal_loss:                   12.27 → 0.90  (≈ 14x 減)
weighted_terminal (value_sem):   3.86  → 0.26 (≈ 15x 減)
T/Y:                             169 → 12.3   (≈ 14x 減)
T/V:                             508 → 21.0   (≈ 24x 減)
```

理屈通り。`tl_old = (tl_per * w).sum()` ≈ `num_groups × group_mean` で、
`weight_sum = num_groups`。CQ-0285 で `tl_new = tl_old / num_groups`
になるため、**`num_groups` (= 1 batch の player-round group 数 ≈ 14〜30)
の倍率で terminal がそのまま縮む**。

---

## 2. コード上の重要ポイント

### 2.1 `_compute_semantic` の forward と detach

[python/mahjong_rl/models/stage2a_model.py:293-308](python/mahjong_rl/models/stage2a_model.py#L293-L308)

```python
def _compute_semantic(self, h_value):
    terminal_logits = self.terminal_head(h_value)   # (B, 5)
    yaku_logits = self.yaku_head(h_value)           # (B, 14)
    proj = self.semantic_proj(h_value)              # (B, sa_proj)
    summary = torch.cat([
        torch.softmax(terminal_logits, dim=-1).detach(),   # (B, 5)
        torch.sigmoid(yaku_logits).detach(),               # (B, 14)
        proj.detach(),                                      # (B, 16)
    ], dim=-1)
    return {
        "terminal_logits": terminal_logits,   # → loss flows back to value_trunk
        "yaku_logits": yaku_logits,           # → loss flows back to value_trunk
        "semantic_summary": summary,           # detached: no policy gradient back
    }
```

- summary の 3 成分は **すべて detach 済み** で policy_loss → value_trunk への
  gradient backflow は無い。
- 一方 `terminal_logits` / `yaku_logits` は detach されていないため、
  `terminal_loss` / `yaku_loss` は value_trunk まで戻る。

→ **value_trunk は terminal/yaku/value の 3 補助 loss だけで shape され、
policy loss は触らない。**

### 2.2 forward_discard / forward_optional での合流

[python/mahjong_rl/models/stage2a_model.py:380-399](python/mahjong_rl/models/stage2a_model.py#L380-L399)
[python/mahjong_rl/models/stage2a_model.py:441-475](python/mahjong_rl/models/stage2a_model.py#L441-L475)

```python
if self._semantic_aux_enabled:
    h_v = self._compute_value_hidden(...)                 # value_trunk hidden
    semantic = self._compute_semantic(h_v)                 # → terminal/yaku/proj/summary
    policy_input = torch.cat([policy_features,
                               semantic["semantic_summary"]], dim=-1)
h_d = self.discard_trunk(policy_input)
...
if compute_value:
    if self._semantic_aux_enabled:
        values["round_delta"] = self.value_head(h_v)       # value も同じ h_v 経由
```

- `value_head` も `terminal_head` も `yaku_head` も同じ `h_v` を入力にする。
- policy 側は `semantic_summary` (= terminal softmax + yaku sigmoid + proj、
  すべて detach) だけ受け取る。

これにより、**value_trunk は「terminal/yaku を当てるための表現」を学び、
それを value_head/policy summary が借りる構造** になっている。

### 2.3 semantic_proj は学習されていない可能性が極めて高い

[python/mahjong_rl/models/stage2a_model.py:194](python/mahjong_rl/models/stage2a_model.py#L194)
で `self.semantic_proj = nn.Linear(prev_v, sa_proj)` を定義。

[python/mahjong_rl/models/stage2a_model.py:297-303](python/mahjong_rl/models/stage2a_model.py#L297-L303)
で `proj = self.semantic_proj(h_value)` を計算するが、
`proj.detach()` を summary に詰めるだけで、`proj` 自体に対する loss は
このリポジトリ内のどこにも存在しない (`grep` 確認済み)。

つまり **`semantic_proj` の重みは初期化値のまま固定** である。
初期化のランダム射影として h_v を 16 次元に変換し続けている (= random
features と同じ)。

これは性能の主因ではないが、

- summary の dim 35 のうち 16 dim (≈ 46%) が学習されない random projection
- `policy_projection_dim` を変えても summary の真の情報量は変わらない
- diagnostics 上 `semantic_proj` group の grad norm は常に 0 になっているはず

ことを意味する。**Q3 / Q5 で再触する**。

### 2.4 reward.point_delta_scale の効果

[configs/stage2a_core_minimal_mixed_s1_baseline.yaml:35-37](configs/stage2a_core_minimal_mixed_s1_baseline.yaml#L35-L37)
で CQ-0283 後の `reward.point_delta_scale: 0.0001`。

- value targets が ±0.5〜±1.5 程度の小さい値になる
- `value_loss = MSE` は 0.01 オーダー
- `value_loss_coef * value_loss = 0.125 × 0.01 ≈ 0.0013`
- → **value_loss から value_trunk への gradient はもともと極小**
- pre-CQ-0285 でも post-CQ-0285 でも、value 側からの value_trunk 更新は
  ほぼ「無い」に近い

つまり value_trunk を意味のある方向に育てる主信号は、reward scale 修正後は
**事実上 terminal_loss と yaku_loss の 2 つだけ**。
yaku は winner-only で sparse なので、**実質 terminal が唯一の dense
auxiliary supervision**。

### 2.5 CQ-0285 が変えたのは loss の絶対 scale

[python/mahjong_rl/stage2a_learner.py:1387-1391](python/mahjong_rl/stage2a_learner.py#L1387-L1391)

```python
if terminal_weights is not None:
    w_sum = terminal_weights.sum().clamp_min(1e-8)
    tl = (tl_per * terminal_weights).sum() / w_sum   # CQ-0285
```

- `terminal_weights[i] = 1 / count[(eid, rid, pid)]` なので
  `weight_sum = (eid, rid, pid) ユニーク数 = num_groups`。
- pre-CQ-0285 の `tl = (tl_per * w).sum()` は ≈ `num_groups × group_mean`
- post-CQ-0285 の `tl = ... / num_groups` は ≈ `group_mean`
- **batch 内の player-round group 数で約 14〜30x の絶対 scale 縮小**。

### 2.6 Adam + max_grad_norm の隠し効果

[python/mahjong_rl/stage2a_learner.py:1607-1608](python/mahjong_rl/stage2a_learner.py#L1607-L1608)

```python
nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
self._optimizer.step()
```

- `max_grad_norm = 0.5` (config L61)
- coef を 50x / 100x にすると total gradient norm は十分に 0.5 を超える
  → clip 発動でグローバルに同倍率縮小
- Adam は per-parameter に second moment で正規化
- → **coef sweep 同倍率は (clip 発動領域では) 学習挙動を変えにくい**

これが exp_027 で COEF10x/50x/100x の `T/Y` ratio がほぼ一定 (5.96〜7.43)、
`raw_terminal_loss` も ≈ 0.85 でほぼ動かない理由。
**「coef を全体的に上げる」は本問題の解決軸ではない**ことが分かる。

---

## 3. 原因仮説 ranked list

### 仮説 A (確信度: 高) **terminal-on-value_trunk が事実上の表現学習主信号で、CQ-0285 でその信号が ~30x 弱まったため policy 用 summary の表現品質が落ちた**

- 支持する証拠
  - 2.1 / 2.2: value_trunk は terminal/yaku/value の 3 loss だけで shape され、
    かつ policy は h_v 由来の semantic_summary を入力に持つ
  - 2.3: semantic_proj が dead weight な以上、summary の有意味な学習信号は
    terminal+yaku のみ。yaku は winner-only で sparse → terminal が dense。
  - 2.4: reward.point_delta_scale=0.0001 のため value_loss は元から ≈ 0.01。
    value 側からの value_trunk 更新は極小。
  - 2.5: CQ-0285 で terminal は 14〜30x 縮小。これが value_trunk への
    auxiliary 信号を直接弱める。
  - 1.1: `final win_rate` は CQ-0285 後の方が高いのに `final avg_rank` が
    悪い → 和了は取れるが「点数を取りに行く局面」と「降りる局面」の
    判別 (= terminal-class 的な outcome 推定) が弱まった解釈と整合。
  - 1.2: coef を上げても entropy だけ下がる (policy が荒れずに鋭くなる)。
    PPO 不安定ではなく「policy への入力 (summary) が劣化した」整合。
- 反証 / 弱点
  - exp_026 baseline でも T/Y 比は本来 100+ レベルだったはずだが、
    そこでも policy の win_rate (0.2368) は CQ-0285 後 (0.2576) より低い。
    → "良い性能" の必要条件として「win_rate を抑えて点数効率を取る」が
    あるなら、terminal supervision が直接効いている所以は十分。
- 追加で見るべきデータ
  - **exp_026 P100 scaled の semantic eval**: terminal head accuracy /
    deal_in PR AUC がどれくらいか (既出の数値あり: terminal accuracy
    0.6017 → 0.6351 と yaku F1 0.49 → 0.68 で改善)
  - **CQ-0285 後 60-cycle の semantic eval**: terminal head の品質が
    どこまで落ちているか (これがあれば仮説 A の直接証拠になる)
  - cycle ごとの `weighted_terminal_loss.value_trunk.mean` 推移と
    eval avg_rank の相関

### 仮説 B (確信度: 中) **`semantic_proj` が学習されないため summary の 16/35 dim が random projection で、policy への有効情報が terminal+yaku の 19 dim しかない (構造的キャップ)**

- 支持する証拠
  - 2.3: `semantic_proj` には逆伝播路が無い (summary 経由では detach、
    他に loss なし)
  - `policy_projection_dim=16` で summary に占める比重が大きい
- 反証 / 弱点
  - 仮説 A だけでも CQ-0285 性能差は説明可能
  - random projection でも h_v 経由で state-dependent な情報は流れる
- 追加で見るべきデータ
  - `semantic_proj` の grad norm が常に 0 であることを diagnostics で確認
  - `semantic_proj` を summary から外したときの性能変化

### 仮説 C (確信度: 中) **gae_lambda=0.0 + gamma=0.5 + reward 微小化 で value 側の credit assignment が薄く、policy が summary 入力 (terminal-aware) に重く依存している**

- 支持する証拠
  - exp_025 review でも指摘済み (`gae_lambda=0.0` は短期 credit のみ)
  - reward.point_delta_scale=0.0001 で value_loss も極小
  - → 「policy にとって最も濃い入力は terminal softmax」になっている可能性
- 反証 / 弱点
  - exp_026 baseline はこの設定でも 1.970 を達成している
  - 仮説 A の対角に独立して効いている因子で、相補
- 追加で見るべきデータ
  - `value_loss` 推移と eval avg_rank の関係
  - explained_variance を出して、value 関数が何を予測しているか確認

### 仮説 D (確信度: 低〜中) **CQ-0285 によって `terminal_loss` のスケールが batch 構成に弱依存となり、batch 多様性 (= player-round group 数) が学習信号として失われた**

- 支持する証拠
  - 旧式 (sum) は num_groups に比例 → batch ごとの diversity が
    自動で重み付けされていた可能性
  - 新式 (sum/w_sum) は num_groups 不変 → 全 batch を平等扱いする
- 反証 / 弱点
  - これは「sum か mean か」の正規化選択の問題で、性能差の本筋ではない
  - 仮説 A の effective gradient scale 説に包含される
- 追加で見るべきデータ
  - 1 cycle 内の num_groups 分布 (batch ごとの group 数)
  - num_groups と weighted_terminal_loss の相関

### 仮説 E (確信度: 低) **PPO 側の clip / entropy 不足で policy が早く鋭くなり、後半 cycle で改善が止まる**

- 支持する証拠
  - 全条件で entropy がだんだん下がる (max_prob 0.9 超)
- 反証 / 弱点
  - exp_026 baseline でも同様の entropy 推移なのに 1.970 を達成
  - → 主因ではない
- 追加で見るべきデータ
  - 必須ではない

---

## 4. 次にやるべき実験・実装案

優先順位は「最小コストで最大情報量」を基準に並べる。

### Probe 1 (P0, 必須) — **terminal_loss_coef だけ ~30x、yaku/value は base のまま**

仮説 A の直接検証。CQ-0285 で失われたのが「terminal の絶対 gradient scale」
ならこれで戻る。COEF50x で戻らなかったのは yaku/value も同倍率にしたため
T/Y 比が pre-CQ-0285 の 169 まで戻らなかったから。

```yaml
training:
  value_loss_coef: 0.125               # base
  semantic_aux:
    terminal_loss_coef: 3.0             # 30x base (CQ-0285 で縮んだ scale を相殺)
    yaku_loss_coef: 0.05                # base
training.lr: 0.0001
training.diagnostics.gradient_norms.enabled: true
```

(他は exp_027 と同条件、seed42、60 cycle)

期待される観測:

- `T/Y` ≈ 150〜200 (pre-CQ-0285 帯)
- `T/V` ≈ 400+ (pre-CQ-0285 帯)
- `final avg_rank` が 2.0 以下 / `tail10` が 2.15 以下
  → 仮説 A 確定、CQ-0285 を「式は維持・係数だけ補正」で運用する根拠
- 期待外れ (= 性能戻らない) なら仮説 B/C を疑う

### Probe 2 (P0, 同時に走らせる) — **CQ-0285 を完全 revert (coef 全部 base)**

最も Probe 1 に対する制御群として有用。これで exp_026 が再現すれば
「pre-CQ-0285 の sum 定義そのもの」が機能していた仮説の確定。
両方 1seed なら直接比較できる。

```python
# stage2a_learner.py: revert _compute_semantic_aux_loss terminal path
if terminal_weights is not None:
    tl = (tl_per * terminal_weights).sum()       # ← pre-CQ-0285
else:
    tl = tl_per.mean()
```

config: 全 base coef。

期待される観測:

- `T/Y` ≈ 130〜320, `T/V` ≈ 240〜1300 (gradnorm probe 帯)
- `final ≈ 1.97`, `tail10 ≈ 2.12` (= exp_026 再現)

**Probe 1 と Probe 2 がほぼ同性能になれば、機序は「terminal gradient scale」
で確定**。Probe 1 のみ良ければ、追加で式の中間値も当てられる。

### Probe 3 (P1) — **alpha sweep on terminal weight normalization**

Probe 1 / 2 が両方とも改善しない、または逆に「中間 alpha が一番良い」
可能性を捨てたくない場合。実装は簡単。

```python
# stage2a_learner.py: configurable alpha
alpha = self._terminal_weight_alpha   # config: 0.0 (sum) ... 1.0 (mean) の間
if terminal_weights is not None:
    w_sum = terminal_weights.sum().clamp_min(1e-8)
    tl = (tl_per * terminal_weights).sum() / w_sum.pow(alpha)
```

1seed sweep: alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}。
ただし Probe 1 で十分結論が出るなら省略可。

### Probe 4 (P1, 軽実装) — **`semantic_proj` を summary から外す or 学習可能にする**

現在 dead weight。2 案:

A. summary から `proj.detach()` を削除し summary dim = 5 + 14 = 19 にする。
B. `proj` に loss を付ける (例: `proj` を value_head の入力に concat して
   value 関数の表現を直接補助する、または contrastive loss、または
   そもそも `semantic_proj` モジュール自体を削除)。

いずれも 1seed probe 1本で確認可能。
これが効くと Probe 1 の効果に乗って性能上限が上がる可能性がある。

### Probe 5 (P2, 構造変更) — **semantic_summary を detach せずに policy へ流す**

`_compute_semantic` の `.detach()` を外すと、policy_loss が
`terminal_head` / `yaku_head` / `value_trunk` を直接更新する path ができる。

- メリット: terminal head が「policy にとって有用な outcome 表現」を
  陽に学べる
- リスク: policy の信号で terminal head が真の terminal class から
  バイアスされる
- → 単独では不安定になりやすいので、まず Probe 1〜2 の結果を見てから判断

### Probe 6 (P2, 軽実装で観測強化) — **explained variance / terminal accuracy を cycle ごとに記録**

仮説 A の検証には「terminal head の品質が CQ-0285 後どれだけ落ちたか」を
直接見たい。learner_metrics に以下を追加 (default off で OK):

- `terminal_head_topk_acc` (1 cycle の最後の minibatch で計算)
- `value_head_explained_variance`

これは Probe 1 / 2 の結果解釈を確実にするための観測強化。

---

### 推奨実行順

1. **Probe 1 + Probe 2 を seed42 で同時実行** (60 cycle ×2 本)。
   両方とも先行 run と diagnostics 集計コマンドが共通で再利用可能。
2. 両方の結果から仮説 A の真偽を確定。
3. もし Probe 1 が exp_026 を取り戻すなら、3seed 化に進む。
4. 取り戻さない場合のみ Probe 3 / 4 を試す。

**やらないでよいこと** (informativeness が低いため):

- coef 同倍率 sweep のさらなる拡張 (200x など) — exp_027 で頭打ちは確認済み
- `gae_lambda` 変更や `entropy_coef` 導入 — 主因ではない (仮説 E)
- model dim 拡張 — value_trunk 容量が問題ではなく信号源が問題

---

## 5. 実装バグ / 設計上の見落とし

### 5.1 (確度高) **`semantic_proj` は学習されない dead weight**

[python/mahjong_rl/models/stage2a_model.py:194,297-303](python/mahjong_rl/models/stage2a_model.py#L194)

`proj = self.semantic_proj(h_value)` の戻り値は `summary = cat(..., proj.detach())`
の形でしか使われず、`semantic_proj` を入力にした loss はリポジトリ内に
存在しない。検索して確認:

```bash
$ grep -rn "semantic_proj" python/mahjong_rl/ tests/
# 該当: nn.Linear 定義 / forward での detach 1 箇所 / gradient norm group 列挙のみ
```

→ `semantic_proj` の重みは初期化値のまま。
summary の `policy_projection_dim` (= 16) 次元は random nonlinear projection
of h_v 相当。

**影響**:
- 性能の主因ではないが、policy 入力 35 dim のうち 16 dim が無駄。
- `policy_projection_dim` を変えても effective info は変わらない (random
  projection の dim 数だけ変わる)。
- gradient norm diagnostics で `semantic_proj` group の値が常に 0 のはず
  (= 観測で確認可能、本人 (Claude Code) 環境では未確認)。

**修正案**: Probe 4 と同じ。最小修正は summary から `proj.detach()` を削除
する 1 行。

### 5.2 (中) **`forward_optional` で `candidate_encoder` が 2 回呼ばれる (semantic_aux 有効時)**

[python/mahjong_rl/models/stage2a_model.py:444,460](python/mahjong_rl/models/stage2a_model.py#L444)

```python
if self._semantic_aux_enabled:
    cand_enc_pre = self.candidate_encoder(cand_features)   # 1回目: summary 用
    opt_summary_pre = self._make_optional_summary(cand_enc_pre, ...)
    h_v = self._compute_value_hidden(...)
    semantic = self._compute_semantic(h_v)
    opt_input = torch.cat([policy_features, response_context,
                            semantic["semantic_summary"]], dim=-1)
else:
    opt_input = torch.cat([policy_features, response_context], dim=-1)
h_c = self.optional_trunk(opt_input)

cand_enc = self.candidate_encoder(cand_features)   # 2回目: scoring 用
```

- 同入力・同パラメータなので結果は等しい
- 純粋な perf 重複 (forward 速度に効く)
- 修正は `cand_enc = cand_enc_pre if cand_enc_pre is not None else self.candidate_encoder(cand_features)` 等

**影響**: 性能には効かない、計算時間のみ。優先度低。

### 5.3 (中) **PPO の value/semantic 系勾配が clip + Adam で同一視されやすい設計**

`max_grad_norm=0.5` の global clip は、`value_loss_coef` と
`semantic_aux.*_coef` を同倍率で大きくしても効果を打ち消す
(= exp_027 で観測された "coef sweep が ratio を変えない" 現象)。

これは **「coef 同倍率上げ」では root cause を切り分けられない**
ことを意味する。今後の sweep 設計では「ratio を狙って動かす単軸 sweep」
(Probe 1 のように) を選ぶべき。

### 5.4 (低、ただし将来の罠) **`gradient_norms` group 列挙に `semantic_proj` が含まれている**

[python/mahjong_rl/stage2a_learner.py](python/mahjong_rl/stage2a_learner.py) の
CQ-0284 で追加した `_gn_build_param_groups`。`semantic_proj` も列挙される
が、5.1 の通り常に grad 0。
**gradient norm diagnostics 上で 0 を見たときに「閉じている」のか
「dead weight」なのかが区別しにくい**点だけ注意。docstring か report 側に
明記しておくと、将来の解析者の混乱を防げる。

### 5.5 (低) **`gae_lambda=0.0` + `value_loss_coef=0.125` + `point_delta_scale=0.0001` の組み合わせで value_loss 由来の gradient はほぼゼロ**

これはバグではなく仕様だが、結果として **value_trunk は事実上 terminal/yaku
だけで shape されている**。Probe 1 で性能が戻れば設計上は問題ないが、
将来 gae_lambda を上げる場合は value_loss_coef も同時に見直す前提とすべき。

---

## まとめ (1 行)

> CQ-0285 は数学的に正しいが、**この設計では terminal_loss の絶対 scale が
> value_trunk 上の唯一の dense 表現学習信号** だったため、定義変更で
> 信号が ~30x 弱まり exp_026 が再現できなくなった。次は **`terminal_loss_coef`
> だけ ~30x に上げる** 1seed probe (Probe 1) と **CQ-0285 の式 revert**
> 1seed probe (Probe 2) を同時に当て、機序を確定するのが最小コストで
> 最大情報量。

---

## 署名

このレビューは Anthropic の CLI コーディング・アシスタント Claude Code が、
ユーザー (takeo1116) の依頼を受けて、リポジトリ内のソースコードと
実験結果のみを直接参照して独立に作成したものです。

- 作成: Claude Code (Anthropic)
- model: Claude Opus 4.7 (1M context)
- 作成日: 2026-05-07
- 対話セッション: ローカルの Claude Code CLI

(過去レビューの履歴: `experiments/Stage02_CallUnlock/exp_022/claude_code_review.md`
で mixed PPO baseline ratio 問題を指摘 → CQ-0282 で fix され exp_023 で
validated。`experiments/Stage02_CallUnlock/exp_025/claude_code_review.md`
で `point_delta_scale=1.0` 問題を指摘 → CQ-0283 で fix され exp_026 で
validated。本レビューはその次の段階で、CQ-0285 後の性能悪化が
"loss scale バグ" と "事実上の表現学習主信号" を同時に変えた帰結である
という仮説を提示するもの。)
