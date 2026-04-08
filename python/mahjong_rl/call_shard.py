"""CQ-0223: Stage2a call decision 用の shard schema

discard decision (Stage1 互換) と call decision (Stage2a 新規) の
両方を保存できる decision-level shard。

Stage1 の LearningSample / ShardWriter / ShardReader はそのまま残す。
Stage2a では本モジュールの DecisionSample / DecisionShardWriter / DecisionShardReader を使う。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class CandidateRecord:
    """1 candidate の action 表現 (call decision 用)

    Attributes:
        action_type: ActionType の int 値 (Chi=3, Pon=4, Daiminkan=5, Skip=8, ...)
        tile_type: 関連牌種 (0-33, Skip は -1)
        target_rel_seat: 鳴き先の相対席 (-1 for Skip)
        consumed_tile_ids: 消費牌の TileId リスト (最大3, 余りは -1 padding)
    """
    action_type: int
    tile_type: int = -1
    target_rel_seat: int = -1
    consumed_tile_ids: tuple[int, ...] = ()


@dataclass
class DecisionSample:
    """1 decision の学習サンプル (discard / call 共通)

    decision_type:
        "discard": Stage1 互換。action=TileType, legal_mask=(34,)
        "call": Stage2a 新規。selected_candidate_index を使用
    """
    # 共通フィールド
    decision_type: str              # "discard" or "call"
    observation: np.ndarray         # エンコード済み特徴量 (flat float32)
    reward: float                   # 累積報酬
    log_prob: float                 # 行動選択時の log_prob
    value: float                    # value 推定
    terminated: bool
    round_over: bool
    player_id: int = 0
    episode_id: str = ""
    round_id: int = 0
    step_id: int = 0
    actor_type: str = "policy"
    sample_semantics_version: int = 2

    # discard 用
    action: int = -1                # TileType (discard のみ)
    legal_mask: np.ndarray | None = None  # (34,) (discard のみ)

    # call 用
    selected_candidate_index: int = -1
    candidate_count: int = 0
    candidates: list[CandidateRecord] = field(default_factory=list)

    # CQ-0242: response context (optional decision のみ有効)
    response_context: np.ndarray | None = None  # (3,) float32: [tile/34, seat/4, menzen]

    # CQ-0239: teacher info
    teacher_top1_index: int = -1       # teacher の top1 action/candidate index
    teacher_best_indices: list[int] = field(default_factory=list)  # best-set indices (discard のみ)
    teacher_source: str = ""           # "rule_based" etc.

    # CQ-0227: future labels
    round_terminal_label: str = ""  # win/ron_loss/tsumo_loss/ryukyoku_tenpai/ryukyoku_noten/abortive_draw
    eventual_win_yaku_ids: list[int] = field(default_factory=list)
    eventual_total_han: int = -1   # -1 = 不明/非和了
    eventual_fu: int = -1          # -1 = 不明/非和了

    # メタデータ
    experiment_id: str = ""
    run_id: str = ""
    worker_id: str = ""
    shard_id: str = ""
    timestamp: float = 0.0


# ---------- Writer ----------

class DecisionShardWriter:
    """DecisionSample をバッファリングして parquet に書き出す"""

    def __init__(self, output_dir: Path, max_samples: int = 10000):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._max_samples = max_samples
        self._buffer: list[DecisionSample] = []
        self._shard_counter = 0

    def add(self, sample: DecisionSample) -> None:
        if sample.shard_id == "":
            sample.shard_id = f"shard_{self._shard_counter:04d}"
        self._buffer.append(sample)
        if len(self._buffer) >= self._max_samples:
            self.flush()

    def flush(self) -> Path | None:
        if not self._buffer:
            return None

        path = self._output_dir / f"shard_{self._shard_counter:04d}.parquet"

        # 固定幅カラム
        data: dict[str, list] = {
            "decision_type": [s.decision_type for s in self._buffer],
            "observation": [s.observation.astype(np.float32).tobytes()
                            for s in self._buffer],
            "observation_dim": [len(s.observation) for s in self._buffer],
            "reward": [float(s.reward) for s in self._buffer],
            "log_prob": [float(s.log_prob) for s in self._buffer],
            "value": [float(s.value) for s in self._buffer],
            "terminated": [s.terminated for s in self._buffer],
            "round_over": [s.round_over for s in self._buffer],
            "player_id": [s.player_id for s in self._buffer],
            "episode_id": [s.episode_id for s in self._buffer],
            "round_id": [s.round_id for s in self._buffer],
            "step_id": [s.step_id for s in self._buffer],
            "actor_type": [s.actor_type for s in self._buffer],
            "sample_semantics_version": [s.sample_semantics_version
                                          for s in self._buffer],
            # discard fields (-1 sentinel for call)
            "action": [s.action for s in self._buffer],
            # call fields (-1 sentinel for discard)
            "selected_candidate_index": [s.selected_candidate_index
                                          for s in self._buffer],
            "candidate_count": [s.candidate_count for s in self._buffer],
            # CQ-0227: future labels
            # CQ-0239: teacher info
            # CQ-0242: response_context
            "response_context": [
                s.response_context.astype(np.float32).tobytes()
                if s.response_context is not None
                else np.zeros(3, dtype=np.float32).tobytes()
                for s in self._buffer
            ],
            "teacher_top1_index": [s.teacher_top1_index for s in self._buffer],
            "teacher_source": [s.teacher_source for s in self._buffer],
            "teacher_best_indices_json": [
                ",".join(str(i) for i in s.teacher_best_indices)
                if s.teacher_best_indices else ""
                for s in self._buffer
            ],
            "round_terminal_label": [s.round_terminal_label for s in self._buffer],
            "eventual_total_han": [s.eventual_total_han for s in self._buffer],
            "eventual_fu": [s.eventual_fu for s in self._buffer],
            # eventual_win_yaku_ids は可変長 → JSON 文字列で保存
            "eventual_win_yaku_ids_json": [
                ",".join(str(y) for y in s.eventual_win_yaku_ids)
                if s.eventual_win_yaku_ids else ""
                for s in self._buffer
            ],
            # metadata
            "experiment_id": [s.experiment_id for s in self._buffer],
            "run_id": [s.run_id for s in self._buffer],
            "worker_id": [s.worker_id for s in self._buffer],
            "shard_id": [s.shard_id for s in self._buffer],
            "timestamp": [s.timestamp for s in self._buffer],
        }

        # legal_mask: discard のみ。call は 0 埋め
        data["legal_mask"] = [
            s.legal_mask.astype(np.float32).tobytes()
            if s.legal_mask is not None
            else np.zeros(34, dtype=np.float32).tobytes()
            for s in self._buffer
        ]

        # candidates: JSON-like encoding per sample
        # 各 candidate を固定幅 4-int にパック: [action_type, tile_type, rel_seat, consumed_count]
        # + consumed_tile_ids を別カラムに
        max_cands = max(s.candidate_count for s in self._buffer) if self._buffer else 0
        if max_cands > 0:
            # candidate_action_types: (N, max_cands) padded with -1
            _MAX_CONSUMED = 3  # Chi=2, Pon=2, Daiminkan=3
            cand_types_flat = []
            cand_tiles_flat = []
            cand_seats_flat = []
            cand_consumed_flat = []
            for s in self._buffer:
                row_types = []
                row_tiles = []
                row_seats = []
                row_consumed = []
                for c in s.candidates:
                    row_types.append(c.action_type)
                    row_tiles.append(c.tile_type)
                    row_seats.append(c.target_rel_seat)
                    # consumed_tile_ids: 固定幅 _MAX_CONSUMED, padding -1
                    ct = list(c.consumed_tile_ids)[:_MAX_CONSUMED]
                    ct += [-1] * (_MAX_CONSUMED - len(ct))
                    row_consumed.extend(ct)
                # pad candidates
                while len(row_types) < max_cands:
                    row_types.append(-1)
                    row_tiles.append(-1)
                    row_seats.append(-1)
                    row_consumed.extend([-1] * _MAX_CONSUMED)
                cand_types_flat.append(
                    np.array(row_types, dtype=np.int32).tobytes())
                cand_tiles_flat.append(
                    np.array(row_tiles, dtype=np.int32).tobytes())
                cand_seats_flat.append(
                    np.array(row_seats, dtype=np.int32).tobytes())
                cand_consumed_flat.append(
                    np.array(row_consumed, dtype=np.int32).tobytes())
            data["candidate_action_types"] = cand_types_flat
            data["candidate_tile_types"] = cand_tiles_flat
            data["candidate_rel_seats"] = cand_seats_flat
            data["candidate_consumed_tiles"] = cand_consumed_flat
            data["max_candidates"] = [max_cands] * len(self._buffer)

        table = pa.table(data)
        pq.write_table(table, path)
        self._buffer.clear()
        self._shard_counter += 1
        return path

    def close(self) -> None:
        self.flush()


# ---------- Reader ----------

class DecisionShardReader:
    """decision shard を読み込む"""

    def __init__(self, shard_dir: Path):
        self._shard_dir = Path(shard_dir)

    def _find_shards(self) -> list[Path]:
        """flat shard + worker nested shard を重複なく返す"""
        flat = set(self._shard_dir.glob("shard_*.parquet"))
        nested = set(self._shard_dir.glob("worker_*/shard_*.parquet"))
        return sorted(flat | nested)

    def read_all(self) -> list[DecisionSample]:
        """全 shard を読んで DecisionSample リストを返す"""
        samples: list[DecisionSample] = []
        for path in self._find_shards():
            table = pq.read_table(path)
            n = table.num_rows
            has_cands = "candidate_action_types" in table.column_names
            max_cands = (table.column("max_candidates")[0].as_py()
                         if has_cands else 0)

            for i in range(n):
                obs_bytes = table.column("observation")[i].as_py()
                obs_dim = table.column("observation_dim")[i].as_py()
                obs = np.frombuffer(obs_bytes, dtype=np.float32).copy()
                assert len(obs) == obs_dim

                mask_bytes = table.column("legal_mask")[i].as_py()
                mask = np.frombuffer(mask_bytes, dtype=np.float32).copy()

                dt = table.column("decision_type")[i].as_py()

                # candidates 復元
                cands: list[CandidateRecord] = []
                cand_count = table.column("candidate_count")[i].as_py()
                if has_cands and cand_count > 0:
                    ct_bytes = table.column("candidate_action_types")[i].as_py()
                    tt_bytes = table.column("candidate_tile_types")[i].as_py()
                    rs_bytes = table.column("candidate_rel_seats")[i].as_py()
                    ct = np.frombuffer(ct_bytes, dtype=np.int32)
                    tt = np.frombuffer(tt_bytes, dtype=np.int32)
                    rs = np.frombuffer(rs_bytes, dtype=np.int32)
                    # consumed_tile_ids 復元
                    has_consumed = "candidate_consumed_tiles" in table.column_names
                    consumed_arr = None
                    if has_consumed:
                        cons_bytes = table.column("candidate_consumed_tiles")[i].as_py()
                        consumed_arr = np.frombuffer(cons_bytes, dtype=np.int32)
                    _MAX_CONSUMED = 3
                    for j in range(cand_count):
                        consumed_ids: tuple[int, ...] = ()
                        if consumed_arr is not None:
                            base = j * _MAX_CONSUMED
                            raw = consumed_arr[base:base + _MAX_CONSUMED]
                            consumed_ids = tuple(int(v) for v in raw if v >= 0)
                        cands.append(CandidateRecord(
                            action_type=int(ct[j]),
                            tile_type=int(tt[j]),
                            target_rel_seat=int(rs[j]),
                            consumed_tile_ids=consumed_ids,
                        ))

                samples.append(DecisionSample(
                    decision_type=dt,
                    observation=obs,
                    reward=table.column("reward")[i].as_py(),
                    log_prob=table.column("log_prob")[i].as_py(),
                    value=table.column("value")[i].as_py(),
                    terminated=table.column("terminated")[i].as_py(),
                    round_over=table.column("round_over")[i].as_py(),
                    player_id=table.column("player_id")[i].as_py(),
                    episode_id=table.column("episode_id")[i].as_py(),
                    round_id=table.column("round_id")[i].as_py(),
                    step_id=table.column("step_id")[i].as_py(),
                    actor_type=table.column("actor_type")[i].as_py(),
                    sample_semantics_version=table.column(
                        "sample_semantics_version")[i].as_py(),
                    action=table.column("action")[i].as_py(),
                    legal_mask=mask if dt == "discard" else None,
                    selected_candidate_index=table.column(
                        "selected_candidate_index")[i].as_py(),
                    candidate_count=cand_count,
                    candidates=cands,
                    # CQ-0242: response_context
                    response_context=(
                        np.frombuffer(
                            table.column("response_context")[i].as_py(),
                            dtype=np.float32).copy()
                        if "response_context" in table.column_names else None),
                    # CQ-0239: teacher info
                    teacher_top1_index=(
                        table.column("teacher_top1_index")[i].as_py()
                        if "teacher_top1_index" in table.column_names else -1),
                    teacher_best_indices=(
                        [int(x) for x in
                         table.column("teacher_best_indices_json")[i].as_py().split(",")
                         if x]
                        if "teacher_best_indices_json" in table.column_names
                        and table.column("teacher_best_indices_json")[i].as_py()
                        else []),
                    teacher_source=(
                        table.column("teacher_source")[i].as_py()
                        if "teacher_source" in table.column_names else ""),
                    # CQ-0227: future labels
                    round_terminal_label=(
                        table.column("round_terminal_label")[i].as_py()
                        if "round_terminal_label" in table.column_names else ""),
                    eventual_win_yaku_ids=(
                        [int(x) for x in
                         table.column("eventual_win_yaku_ids_json")[i].as_py().split(",")
                         if x]
                        if "eventual_win_yaku_ids_json" in table.column_names
                        and table.column("eventual_win_yaku_ids_json")[i].as_py()
                        else []),
                    eventual_total_han=(
                        table.column("eventual_total_han")[i].as_py()
                        if "eventual_total_han" in table.column_names else -1),
                    eventual_fu=(
                        table.column("eventual_fu")[i].as_py()
                        if "eventual_fu" in table.column_names else -1),
                    experiment_id=table.column("experiment_id")[i].as_py(),
                    run_id=table.column("run_id")[i].as_py(),
                    worker_id=table.column("worker_id")[i].as_py(),
                    shard_id=table.column("shard_id")[i].as_py(),
                    timestamp=table.column("timestamp")[i].as_py(),
                ))
        return samples

    def read_as_tensors(self, filter_actor_type: str | None = None) -> dict:
        """CQ-0251: direct parquet → numpy tensor 変換 (DecisionSample object を経由しない)"""
        from mahjong_rl.candidate_encoding import (
            ACTION_TYPE_MAP, MAX_CONSUMED, CAND_FEAT_DIM)

        from mahjong_rl.outcome_vocab import (
            terminal_label_to_class, yaku_ids_to_multihot, NUM_YAKU)

        # 1st pass: collect rows per branch
        d_rows: list[tuple] = []  # (table, row_index)
        c_rows: list[tuple] = []
        tables: list = []
        for path in self._find_shards():
            table = pq.read_table(path)
            tables.append(table)
            n = table.num_rows
            dt_col = table.column("decision_type")
            at_col = (table.column("actor_type")
                      if "actor_type" in table.column_names else None)
            for i in range(n):
                if filter_actor_type and at_col is not None:
                    if at_col[i].as_py() != filter_actor_type:
                        continue
                if dt_col[i].as_py() == "discard":
                    d_rows.append((table, i))
                else:
                    c_rows.append((table, i))

        def _col(tbl, name, idx):
            return tbl.column(name)[idx].as_py()

        def _col_safe(tbl, name, idx, default=None):
            if name in tbl.column_names:
                return tbl.column(name)[idx].as_py()
            return default

        def _bytes_to_f32(b, size):
            return np.frombuffer(b, dtype=np.float32).copy() if b else np.zeros(size, dtype=np.float32)

        # build discard
        d_result = None
        nd = len(d_rows)
        if nd > 0:
            obs_dim = _col(d_rows[0][0], "observation_dim", d_rows[0][1])
            obs = np.zeros((nd, obs_dim), dtype=np.float32)
            masks = np.zeros((nd, 34), dtype=np.float32)
            actions = np.zeros(nd, dtype=np.int64)
            rewards = np.zeros(nd, dtype=np.float32)
            log_probs = np.zeros(nd, dtype=np.float32)
            values = np.zeros(nd, dtype=np.float32)
            terminateds = np.zeros(nd, dtype=np.float32)
            step_ids = np.zeros(nd, dtype=np.int64)
            actor_types = []
            teacher_top1 = np.full(nd, -1, dtype=np.int64)
            teacher_best_mask = np.zeros((nd, 34), dtype=np.float32)
            has_best = False
            episode_ids = []
            player_ids = np.zeros(nd, dtype=np.int64)
            round_ids = np.zeros(nd, dtype=np.int64)
            resp_ctx = np.zeros((nd, 3), dtype=np.float32)
            # CQ-0256: semantic aux targets
            terminal_classes = np.zeros(nd, dtype=np.int64)
            yaku_multihot = np.zeros((nd, NUM_YAKU), dtype=np.float32)
            is_winner = np.zeros(nd, dtype=np.float32)

            for i, (tbl, ri) in enumerate(d_rows):
                obs[i] = _bytes_to_f32(_col(tbl, "observation", ri), obs_dim)
                masks[i] = _bytes_to_f32(_col(tbl, "legal_mask", ri), 34)
                actions[i] = _col(tbl, "action", ri)
                rewards[i] = _col(tbl, "reward", ri)
                log_probs[i] = _col(tbl, "log_prob", ri)
                values[i] = _col(tbl, "value", ri)
                terminateds[i] = float(_col(tbl, "terminated", ri))
                step_ids[i] = _col(tbl, "step_id", ri)
                actor_types.append(_col_safe(tbl, "actor_type", ri, "policy"))
                t1 = _col_safe(tbl, "teacher_top1_index", ri, -1)
                teacher_top1[i] = t1 if t1 is not None else -1
                # best-set from JSON
                bj = _col_safe(tbl, "teacher_best_indices_json", ri, "")
                if bj:
                    for idx_str in bj.split(","):
                        if idx_str:
                            teacher_best_mask[i, int(idx_str)] = 1.0
                            has_best = True
                episode_ids.append(_col(tbl, "episode_id", ri))
                player_ids[i] = _col(tbl, "player_id", ri)
                round_ids[i] = _col_safe(tbl, "round_id", ri, 0)
                rc_bytes = _col_safe(tbl, "response_context", ri, None)
                if rc_bytes:
                    resp_ctx[i] = np.frombuffer(rc_bytes, dtype=np.float32)
                # CQ-0256: semantic targets
                rtl = _col_safe(tbl, "round_terminal_label", ri, "")
                terminal_classes[i] = terminal_label_to_class(rtl)
                yids_json = _col_safe(tbl, "eventual_win_yaku_ids_json", ri, "")
                if yids_json:
                    yids = [int(x) for x in yids_json.split(",") if x]
                    yaku_multihot[i] = yaku_ids_to_multihot(yids)
                    if rtl in ("win", "win_menzen", "win_called"):
                        is_winner[i] = 1.0

            d_result = {
                "observations": obs, "legal_masks": masks, "actions": actions,
                "rewards": rewards, "log_probs": log_probs, "values": values,
                "terminateds": terminateds, "step_ids": step_ids,
                "actor_types": actor_types, "teacher_top1": teacher_top1,
                "teacher_best_mask": teacher_best_mask if has_best else None,
                "episode_ids": episode_ids, "player_ids": player_ids,
                "round_ids": round_ids, "response_context": resp_ctx,
                "terminal_classes": terminal_classes,
                "yaku_multihot": yaku_multihot,
                "is_winner": is_winner,
                "n": nd,
            }

        # build call
        c_result = None
        nc = len(c_rows)
        if nc > 0:
            obs_dim = _col(c_rows[0][0], "observation_dim", c_rows[0][1])
            # determine max_cands
            max_cands = 0
            for tbl, ri in c_rows:
                cc = _col(tbl, "candidate_count", ri)
                if cc > max_cands:
                    max_cands = cc

            obs = np.zeros((nc, obs_dim), dtype=np.float32)
            selected_idx = np.zeros(nc, dtype=np.int64)
            rewards = np.zeros(nc, dtype=np.float32)
            log_probs_c = np.zeros(nc, dtype=np.float32)
            values_c = np.zeros(nc, dtype=np.float32)
            terminateds_c = np.zeros(nc, dtype=np.float32)
            step_ids_c = np.zeros(nc, dtype=np.int64)
            actor_types_c = []
            teacher_top1_c = np.full(nc, -1, dtype=np.int64)
            episode_ids_c = []
            player_ids_c = np.zeros(nc, dtype=np.int64)
            round_ids_c = np.zeros(nc, dtype=np.int64)
            resp_ctx_c = np.zeros((nc, 3), dtype=np.float32)
            cand_feats = np.zeros((nc, max_cands, CAND_FEAT_DIM), dtype=np.int64)
            cand_mask = np.zeros((nc, max_cands), dtype=np.float32)
            terminal_classes_c = np.zeros(nc, dtype=np.int64)
            yaku_multihot_c = np.zeros((nc, NUM_YAKU), dtype=np.float32)
            is_winner_c = np.zeros(nc, dtype=np.float32)

            for i, (tbl, ri) in enumerate(c_rows):
                obs[i] = _bytes_to_f32(_col(tbl, "observation", ri), obs_dim)
                selected_idx[i] = _col(tbl, "selected_candidate_index", ri)
                rewards[i] = _col(tbl, "reward", ri)
                log_probs_c[i] = _col(tbl, "log_prob", ri)
                values_c[i] = _col(tbl, "value", ri)
                terminateds_c[i] = float(_col(tbl, "terminated", ri))
                step_ids_c[i] = _col(tbl, "step_id", ri)
                actor_types_c.append(_col_safe(tbl, "actor_type", ri, "policy"))
                t1 = _col_safe(tbl, "teacher_top1_index", ri, -1)
                teacher_top1_c[i] = t1 if t1 is not None else -1
                episode_ids_c.append(_col(tbl, "episode_id", ri))
                player_ids_c[i] = _col(tbl, "player_id", ri)
                round_ids_c[i] = _col_safe(tbl, "round_id", ri, 0)
                rc_bytes = _col_safe(tbl, "response_context", ri, None)
                if rc_bytes:
                    resp_ctx_c[i] = np.frombuffer(rc_bytes, dtype=np.float32)

                # candidate direct encoding from parquet columns
                cc = _col(tbl, "candidate_count", ri)
                if "candidate_action_types" in tbl.column_names and cc > 0:
                    mc_in_shard = _col_safe(tbl, "max_candidates", ri, cc)
                    ct = np.frombuffer(
                        tbl.column("candidate_action_types")[ri].as_py(),
                        dtype=np.int32)
                    tt = np.frombuffer(
                        tbl.column("candidate_tile_types")[ri].as_py(),
                        dtype=np.int32)
                    rs = np.frombuffer(
                        tbl.column("candidate_rel_seats")[ri].as_py(),
                        dtype=np.int32)
                    cons = None
                    if "candidate_consumed_tiles" in tbl.column_names:
                        cons = np.frombuffer(
                            tbl.column("candidate_consumed_tiles")[ri].as_py(),
                            dtype=np.int32)
                    for j in range(min(cc, max_cands)):
                        # raw → learner encoding
                        at_raw = int(ct[j])
                        at_idx = ACTION_TYPE_MAP.get(at_raw, 0)
                        tile_1 = int(tt[j]) + 1 if tt[j] >= 0 else 0
                        seat_1 = int(rs[j]) + 1 if rs[j] >= 0 else 0
                        c0 = c1 = c2 = 0
                        if cons is not None:
                            base = j * MAX_CONSUMED
                            for k in range(MAX_CONSUMED):
                                raw = int(cons[base + k]) if base + k < len(cons) else -1
                                if raw >= 0:
                                    if k == 0: c0 = (raw // 4) + 1
                                    elif k == 1: c1 = (raw // 4) + 1
                                    else: c2 = (raw // 4) + 1
                        cand_feats[i, j] = [at_idx, tile_1, seat_1, c0, c1, c2]
                        cand_mask[i, j] = 1.0

                # CQ-0256: semantic targets
                rtl = _col_safe(tbl, "round_terminal_label", ri, "")
                terminal_classes_c[i] = terminal_label_to_class(rtl)
                yids_json = _col_safe(tbl, "eventual_win_yaku_ids_json", ri, "")
                if yids_json:
                    yids = [int(x) for x in yids_json.split(",") if x]
                    yaku_multihot_c[i] = yaku_ids_to_multihot(yids)
                    if rtl in ("win", "win_menzen", "win_called"):
                        is_winner_c[i] = 1.0

            c_result = {
                "observations": obs, "selected_idx": selected_idx,
                "rewards": rewards, "log_probs": log_probs_c,
                "values": values_c, "terminateds": terminateds_c,
                "step_ids": step_ids_c, "actor_types": actor_types_c,
                "teacher_top1": teacher_top1_c, "episode_ids": episode_ids_c,
                "player_ids": player_ids_c, "round_ids": round_ids_c,
                "response_context": resp_ctx_c,
                "cand_feats": cand_feats, "cand_mask": cand_mask,
                "terminal_classes": terminal_classes_c,
                "yaku_multihot": yaku_multihot_c,
                "is_winner": is_winner_c,
                "max_cands": max_cands, "n": nc,
            }

        return {"discard": d_result, "call": c_result}
