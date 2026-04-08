"""Stage 1 統合ランナー: config.yaml → self-play → learner → eval"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from mahjong_rl.experiment import ExperimentConfig, RunDirectory
from mahjong_rl.profiler import Profiler

import hashlib
import traceback

VALID_DEVICES = {"cpu", "cuda", "auto"}


def _utc_now_str() -> str:
    """UTC タイムスタンプを ISO8601 Z 形式で返す (ミリ秒精度)"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def derive_worker_seed(base_seed: int, worker_id: int) -> int:
    """worker 用 seed を base_seed から派生する

    SHA-256 ハッシュで衝突しにくい seed を生成する。

    Args:
        base_seed: 実験のベース seed
        worker_id: worker 識別子

    Returns:
        worker 用 seed (0 〜 2**32-1)
    """
    data = f"worker:{base_seed}:{worker_id}".encode()
    h = hashlib.sha256(data).hexdigest()
    return int(h[:8], 16)  # 先頭 8 hex chars → 32bit


def derive_match_seed(worker_seed: int, match_index: int) -> int:
    """match 用 seed を worker_seed から派生する

    Args:
        worker_seed: worker のベース seed
        match_index: match のインデックス (0-based)

    Returns:
        match 用 seed
    """
    data = f"match:{worker_seed}:{match_index}".encode()
    h = hashlib.sha256(data).hexdigest()
    return int(h[:8], 16)


def configure_worker_threads(num_threads: int = 1) -> dict:
    """worker 内部スレッド数を抑制する

    multi-process 環境で各 worker のスレッド数を固定し、
    スレッド競合を防ぐ。

    Args:
        num_threads: スレッド数 (デフォルト: 1)

    Returns:
        設定結果の dict (記録用)
    """
    torch.set_num_threads(num_threads)
    # 注: torch.set_num_interop_threads は subprocess 内でのみ呼ぶ
    # (プロセス起動後に変更すると abort する)

    env_vars = {
        "OMP_NUM_THREADS": str(num_threads),
        "MKL_NUM_THREADS": str(num_threads),
        "OPENBLAS_NUM_THREADS": str(num_threads),
    }
    for key, val in env_vars.items():
        os.environ[key] = val

    return {
        "torch_num_threads": torch.get_num_threads(),
        "env_vars": env_vars,
    }


def resolve_device(requested: str) -> torch.device:
    """デバイス文字列を torch.device に解決する

    Args:
        requested: 'cpu', 'cuda', 'auto' のいずれか

    Returns:
        解決された torch.device

    Raises:
        RuntimeError: 'cuda' が要求されたが利用不可の場合
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda が要求されましたが利用できません")
    return torch.device(requested)
import multiprocessing as mp

from mahjong_rl.encoders import FlatFeatureEncoder, ChannelTensorEncoder
from mahjong_rl.models import MLPPolicyValueModel
from mahjong_rl.selfplay_worker import SelfPlayWorker
from mahjong_rl.learner import Learner
from mahjong_rl.evaluator import (
    EvaluationRunner, compute_eval_diff,
    PartialEvalMetrics, aggregate_partials,
    save_partial, load_partials, aggregate_and_save,
    aggregate_rotation_partials,
)


def _first_not_none(*values):
    """最初の non-None 値を返す。全て None なら None。0.0 は有効値として扱う。"""
    for v in values:
        if v is not None:
            return v
    return None


def _parse_encoder_flag(enc_cfg: dict, key: str) -> bool:
    """encoder config のフラグを dict 形式/bool 形式の両方に対応して取得する"""
    v = enc_cfg.get(key, {})
    return v.get("enabled", False) if isinstance(v, dict) else bool(v)


def _rebuild_encoder(encoder_config: dict, obs_mode: str):
    """encoder_config からエンコーダを再構築する (worker 用ヘルパー, CQ-0119, CQ-0171)"""
    enc_name = encoder_config.get("name", "FlatFeatureEncoder")
    enc_obs = encoder_config.get("observation_mode", obs_mode)
    if enc_name == "ChannelTensorEncoder":
        return ChannelTensorEncoder(observation_mode=enc_obs)
    return FlatFeatureEncoder(
        observation_mode=enc_obs,
        shanten_hint=_parse_encoder_flag(encoder_config, "shanten_hint"),
        discard_ukeire_hint=_parse_encoder_flag(encoder_config, "discard_ukeire_hint"),
        current_shanten_input=_parse_encoder_flag(encoder_config, "current_shanten"),
        shape_hint=_parse_encoder_flag(encoder_config, "shape_hint"),
        turn_context=_parse_encoder_flag(encoder_config, "turn_context"),
        opponent_current_shanten=_parse_encoder_flag(encoder_config, "opponent_current_shanten"),
        opponent_tenpai_flag=_parse_encoder_flag(encoder_config, "opponent_tenpai_flag"),
        danger_mask=_parse_encoder_flag(encoder_config, "danger_mask"),
    )


# CQ-0213: full-only source 一覧（Partial mode では auto-off する）
_FULL_ONLY_SOURCES = frozenset({
    "danger_mask_kamicha", "danger_mask_toimen", "danger_mask_shimo",
    "opponent_current_shanten", "opponent_tenpai_flag",
})


def _resolve_direct_hint_ranges(
    pdh_cfg: dict, feature_ranges: dict[str, tuple[int, int]] | None,
) -> dict[str, tuple[int, int]] | None:
    """CQ-0203/CQ-0204/CQ-0213: policy_direct_hints の source → range を解決・検証する

    main process / worker の両方でこのヘルパーを使い、validation を統一する。
    full-only source が feature_ranges にない場合（Partial mode）は自動スキップする。
    それ以外の source が見つからない場合は ValueError を送出する。
    """
    if not pdh_cfg.get("enabled", False):
        return None
    fr = feature_ranges or {}
    sources = pdh_cfg.get("sources", [])
    result = {}
    for src in sources:
        if src not in fr:
            if src in _FULL_ONLY_SOURCES:
                continue  # Partial mode auto-off
            raise ValueError(
                f"policy_direct_hints.sources の '{src}' が "
                f"encoder feature_ranges に見つかりません "
                f"(worker 再構築時)")
        result[src] = fr[src]
    return result


def _rebuild_model(model_config: dict, encoder_meta) -> "MLPPolicyValueModel":
    """model_config + encoder metadata からモデルを再構築する (CQ-0203, CQ-0204)"""
    import math
    input_dim = math.prod(encoder_meta.output_shape)
    _vf = model_config.get("value_features", {})
    _cs = _vf.get("current_shanten", {})
    _vaux_dim = 1 if _cs.get("enabled", False) else 0
    _pt = model_config.get("policy_tower", {})
    _vt = model_config.get("value_tower", {})
    _pdh = model_config.get("policy_direct_hints", {})
    _dhr = _resolve_direct_hint_ranges(_pdh, encoder_meta.feature_ranges)
    return MLPPolicyValueModel(
        input_dim=input_dim,
        hidden_dims=model_config.get("hidden_dims", [256, 128]),
        value_heads=model_config.get("value_heads", ["round_delta"]),
        value_aux_dim=_vaux_dim,
        policy_tower_config=_pt if _pt.get("enabled", False) else None,
        value_tower_config=_vt if _vt.get("enabled", False) else None,
        policy_direct_hints_config=_pdh if _pdh.get("enabled", False) else None,
        direct_hint_ranges=_dhr,
    )


class WorkerSidecar:
    """CQ-0212: worker crash triage 用 sidecar ファイル

    worker 起動時にメタデータを書き出し、match ごとに heartbeat を更新する。
    native abort でも親 runner が sidecar を読んでどの match で落ちたかを特定できる。
    """

    def __init__(self, output_dir: str | Path, worker_id: int, phase: str,
                 base_seed: int, worker_seed: int, **extra):
        self._path = Path(output_dir) / f"worker_{worker_id}_sidecar.json"
        self._data = {
            "worker_id": worker_id,
            "phase": phase,
            "base_seed": base_seed,
            "worker_seed": worker_seed,
            "started_at": _utc_now_str(),
            "status": "running",
            **extra,
        }
        self._flush()

    def heartbeat(self, match_index: int, match_seed: int, **extra):
        """match 開始直前に呼ぶ"""
        self._data["current_match_index"] = match_index
        self._data["current_match_seed"] = match_seed
        self._data["heartbeat_at"] = _utc_now_str()
        self._data.update(extra)
        self._flush()

    def finish(self):
        self._data["status"] = "completed"
        self._data["finished_at"] = _utc_now_str()
        self._flush()

    def _flush(self):
        import json as _json
        try:
            with open(self._path, "w") as f:
                _json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # sidecar 書き込み失敗は worker を止めない


def _eval_worker_fn(
    worker_id: int,
    model_path: str,
    model_config: dict,
    encoder_config: dict,
    obs_mode: str,
    num_matches: int,
    policy_seats: list[int],
    partials_dir: str,
    num_threads: int,
    base_seed: int,
    error_queue: mp.Queue | None = None,
    reward_config_dict: dict | None = None,
) -> None:
    """evaluation worker プロセスのエントリポイント

    subprocess (spawn) として実行される。結果は partials_dir に保存する。
    モデルはファイルから読み込む (shared-memory 非依存)。
    例外発生時は error_queue に詳細を送る。
    """
    try:
        # worker は CPU 推論のため CUDA 初期化を避ける
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # スレッド数固定 (spawn なので interop_threads も設定可能)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        env_vars = {
            "OMP_NUM_THREADS": str(num_threads),
            "MKL_NUM_THREADS": str(num_threads),
            "OPENBLAS_NUM_THREADS": str(num_threads),
        }
        for key, val in env_vars.items():
            os.environ[key] = val

        # seed 派生: base_seed → worker_seed → match_seeds
        worker_seed = derive_worker_seed(base_seed, worker_id)
        match_seeds = [derive_match_seed(worker_seed, i) for i in range(num_matches)]

        # CQ-0212: crash triage sidecar
        sidecar = WorkerSidecar(
            partials_dir, worker_id, phase="eval",
            base_seed=base_seed, worker_seed=worker_seed,
            policy_seats=policy_seats,
            num_matches=num_matches,
            match_seed_range=[match_seeds[0], match_seeds[-1]] if match_seeds else [],
        )

        # モデル・エンコーダ再構築 (ファイルから state_dict を読み込み)
        encoder = _rebuild_encoder(encoder_config, obs_mode)
        meta = encoder.metadata()
        model = _rebuild_model(model_config, meta)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # CQ-0162: reward config を env に渡す
        _reward_config = None
        if reward_config_dict:
            from mahjong_rl import RewardPolicyConfig
            _reward_config = RewardPolicyConfig()
            _reward_config.point_delta_scale = reward_config_dict.get(
                "point_delta_scale", 1.0)

        # CQ-0153, CQ-0204: value current_shanten 有効時は evaluator にも渡す
        _vf_cfg = model_config.get("value_features", {})
        _cs_enabled = _vf_cfg.get("current_shanten", {}).get("enabled", False)
        eval_runner = EvaluationRunner(
            model=model, encoder=encoder, observation_mode=obs_mode,
            value_shanten_enabled=_cs_enabled,
            reward_config=_reward_config)

        # CQ-0215, CQ-0218: per-match heartbeat with policy seat
        eval_runner.set_match_callback(
            lambda mi, ms, seat: sidecar.heartbeat(
                mi, ms, current_policy_seat=seat))
        partial = eval_runner.evaluate_partial(
            num_matches=num_matches,
            policy_seats=policy_seats,
            worker_id=worker_id,
            match_seeds=match_seeds,
        )
        # seed/thread 情報をメタデータに記録
        partial.metadata = {
            "base_seed": base_seed,
            "worker_seed": worker_seed,
            "num_threads": num_threads,
            "torch_num_threads": torch.get_num_threads(),
        }
        save_partial(partial, Path(partials_dir), worker_id=worker_id)
        sidecar.finish()
    except Exception as e:
        if error_queue is not None:
            error_queue.put({
                "worker_id": worker_id,
                "exception_type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            })
        raise


def _selfplay_worker_fn(
    worker_id: int,
    model_path: str,
    config: dict,
    model_config: dict,
    encoder_config: dict,
    obs_mode: str,
    num_matches: int,
    match_seeds: list[int],
    output_dir: str,
    num_threads: int,
    base_seed: int,
    worker_seed: int,
    error_queue: mp.Queue | None = None,
) -> None:
    """self-play worker プロセスのエントリポイント

    subprocess (spawn) として実行される。shard を output_dir に保存し、
    stats.json に統計を書き出す。
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        env_vars = {
            "OMP_NUM_THREADS": str(num_threads),
            "MKL_NUM_THREADS": str(num_threads),
            "OPENBLAS_NUM_THREADS": str(num_threads),
        }
        for key, val in env_vars.items():
            os.environ[key] = val

        # CQ-0212: crash triage sidecar
        sidecar = WorkerSidecar(
            output_dir, worker_id, phase="selfplay",
            base_seed=base_seed, worker_seed=worker_seed,
            num_matches=num_matches,
            match_seed_range=[match_seeds[0], match_seeds[-1]] if match_seeds else [],
        )

        # エンコーダ再構築
        encoder = _rebuild_encoder(encoder_config, obs_mode)

        # モデル再構築 (CQ-0203: _rebuild_model で統一)
        meta = encoder.metadata()
        model = _rebuild_model(model_config, meta)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        worker = SelfPlayWorker(
            config=config,
            model=model,
            encoder=encoder,
            output_dir=Path(output_dir),
            worker_id=f"worker_{worker_id}",
            inference_device=torch.device("cpu"),
        )
        # CQ-0215: per-match heartbeat
        worker.set_match_callback(
            lambda mi, ms: sidecar.heartbeat(mi, ms))
        sp_stats = worker.run(
            num_matches=num_matches,
            match_seeds=match_seeds,
        )
        sidecar.finish()

        # stats を JSON で保存
        sp_stats["base_seed"] = base_seed
        sp_stats["worker_seed"] = worker_seed
        sp_stats["worker_id"] = worker_id
        sp_stats["num_threads"] = num_threads
        if match_seeds is not None and len(match_seeds) > 0:
            sp_stats["match_index_start"] = 0
            sp_stats["match_index_end"] = len(match_seeds) - 1
            sp_stats["first_match_seed"] = match_seeds[0]
            sp_stats["last_match_seed"] = match_seeds[-1]
        # CQ-0142: raw values は別ファイルに保存（stats.json の肥大化防止）
        raw_values = sp_stats.pop("_reward_raw_values", None)
        if raw_values is not None:
            import numpy as _np
            raw_path = Path(output_dir) / "reward_raw_values.npz"
            _np.savez_compressed(
                raw_path,
                **{f"{comp}": _np.array(vals, dtype=_np.float64)
                   for comp, vals in raw_values.items()},
            )

        stats_path = Path(output_dir) / "stats.json"
        import json as _json
        with open(stats_path, "w") as f:
            _json.dump(sp_stats, f, indent=2)

    except Exception as e:
        if error_queue is not None:
            error_queue.put({
                "worker_id": worker_id,
                "exception_type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            })
        raise


logger = logging.getLogger(__name__)


class Stage1Runner:
    """Stage 1 実験の統合ランナー

    config.yaml を入力として以下のフェーズを順に実行する:
      1. run directory 初期化
      2. (optional) imitation warm start
      3. self-play データ生成
      4. learner による学習
      5. evaluator による評価

    各フェーズは experiment.phases 設定で有効/無効を制御できる。
    デフォルト: ["selfplay", "learner", "eval"]
    imitation 付き: ["imitation", "selfplay", "learner", "eval"]
    """

    def __init__(self, config: ExperimentConfig, base_dir: Path = Path("runs"),
                 resume_run_dir: Path | str | None = None,
                 reuse_from: dict | None = None):
        self._config = config
        self._base_dir = base_dir
        self._global_seed: int | None = None
        self._resume_run_dir = Path(resume_run_dir) if resume_run_dir else None
        self._reuse_from = reuse_from  # {"run_dir": str, "phases": list[str]}

    def _get_phases(self) -> list[str]:
        """実行フェーズのリストを取得する"""
        phases = self._config.experiment.get("phases", None)
        if phases is not None:
            return list(phases)
        return ["selfplay", "learner", "eval"]

    def validate_config(self) -> list[str]:
        """config のバリデーションを行い、エラーメッセージのリストを返す

        空リスト = バリデーション成功
        """
        errors: list[str] = []
        cfg = self._config

        # phases の値チェック
        valid_phases = {"imitation", "selfplay", "learner", "eval"}
        phases = self._get_phases()
        for p in phases:
            if p not in valid_phases:
                errors.append(
                    f"不正なフェーズ '{p}' (有効値: {sorted(valid_phases)})")

        # imitation が phases にあるのに selfplay がない場合の警告
        if "imitation" in phases and "selfplay" not in phases:
            errors.append(
                "imitation フェーズがあるのに selfplay フェーズがありません")

        # eval mode チェック
        eval_mode = cfg.evaluation.get("mode", "single")
        if eval_mode not in ("single", "rotation"):
            errors.append(
                f"不正な evaluation.mode '{eval_mode}' (有効値: single, rotation)")

        # observation mode チェック
        obs_mode = cfg.experiment.get("observation_mode", "full")
        if obs_mode not in ("full", "partial"):
            errors.append(
                f"不正な observation_mode '{obs_mode}' (有効値: full, partial)")

        # encoder 名チェック
        enc_name = cfg.feature_encoder.get("name", "FlatFeatureEncoder")
        valid_encoders = {"FlatFeatureEncoder", "ChannelTensorEncoder"}
        if enc_name not in valid_encoders:
            errors.append(
                f"不正な encoder '{enc_name}' (有効値: {sorted(valid_encoders)})")

        # model 名チェック
        model_name = cfg.model.get("name", "MLPPolicyValueModel")
        valid_models = {"MLPPolicyValueModel"}
        if model_name not in valid_models:
            errors.append(
                f"不正な model '{model_name}' (有効値: {sorted(valid_models)})")

        # global_seed 型・値域チェック
        seed = cfg.experiment.get("global_seed", None)
        if seed is not None:
            if not isinstance(seed, (int, float)):
                errors.append(
                    f"global_seed は整数で指定してください (型: {type(seed).__name__})")
            elif isinstance(seed, float) and seed != int(seed):
                errors.append(
                    f"global_seed は整数で指定してください (値: {seed})")
            elif not (0 <= int(seed) <= 2**32 - 1):
                errors.append(
                    f"global_seed は 0 〜 {2**32 - 1} の範囲で指定してください (値: {int(seed)})")

        # seed_start 値域チェック
        for key_path, label in [
            (("selfplay", "seed_start"), "selfplay.seed_start"),
            (("evaluation", "seed_start"), "evaluation.seed_start"),
        ]:
            section = getattr(cfg, key_path[0], {})
            sv = section.get(key_path[1], None)
            if sv is not None:
                if isinstance(sv, (int, float)):
                    if sv < 0:
                        errors.append(
                            f"{label} は 0 以上で指定してください (値: {sv})")

        # デバイス設定チェック
        for device_key, label in [
            (("training", "device"), "training.device"),
            (("selfplay", "inference_device"), "selfplay.inference_device"),
            (("evaluation", "inference_device"), "evaluation.inference_device"),
        ]:
            section = getattr(cfg, device_key[0], {})
            dv = section.get(device_key[1], "auto")
            if dv not in VALID_DEVICES:
                errors.append(
                    f"不正な {label} '{dv}' (有効値: {sorted(VALID_DEVICES)})")

        # num_workers チェック
        for section_name, key in [
            ("selfplay", "num_workers"),
            ("imitation", "num_workers"),
            ("evaluation", "num_workers"),
        ]:
            section = getattr(cfg, section_name, {})
            nw = section.get(key, None)
            if nw is not None:
                if not isinstance(nw, int) or isinstance(nw, bool):
                    errors.append(
                        f"{section_name}.{key} は正の整数で指定してください"
                        f" (型: {type(nw).__name__})")
                elif nw < 1:
                    errors.append(
                        f"{section_name}.{key} は 1 以上で指定してください"
                        f" (値: {nw})")

        # worker_num_threads チェック
        for section_name, key in [
            ("selfplay", "worker_num_threads"),
            ("evaluation", "worker_num_threads"),
        ]:
            section = getattr(cfg, section_name, {})
            nt = section.get(key, None)
            if nt is not None:
                if not isinstance(nt, int) or isinstance(nt, bool):
                    errors.append(
                        f"{section_name}.{key} は正の整数で指定してください"
                        f" (型: {type(nt).__name__})")
                elif nt < 1:
                    errors.append(
                        f"{section_name}.{key} は 1 以上で指定してください"
                        f" (値: {nt})")

        # output_layout チェック
        ol = cfg.selfplay.get("output_layout", None)
        if ol is not None and ol != "worker_subdir":
            errors.append(
                f"不正な selfplay.output_layout '{ol}'"
                f" (有効値: worker_subdir)")

        # seed_strategy チェック
        ss = cfg.experiment.get("seed_strategy", None)
        if ss is not None and ss != "derive":
            errors.append(
                f"不正な experiment.seed_strategy '{ss}'"
                f" (有効値: derive)")

        # profiling チェック (CQ-0098)
        prof_enabled = cfg.profiling.get("enabled", False)
        if not isinstance(prof_enabled, bool):
            errors.append(
                f"profiling.enabled は bool で指定してください: {prof_enabled}")

        # CQ-0257: Stage2a semantic_aux 整合性チェック
        stage = cfg.experiment.get("stage", 1)
        if str(stage) == "stage2a":
            model_sa = cfg.model.get("semantic_aux", {})
            train_sa = cfg.training.get("semantic_aux", {})
            model_sa_enabled = model_sa.get("enabled", False)
            train_sa_enabled = train_sa.get("enabled", False)
            if model_sa_enabled and not train_sa_enabled:
                errors.append(
                    "model.semantic_aux.enabled=true だが"
                    " training.semantic_aux.enabled=false です。"
                    " 両方揃えてください。")
            if not model_sa_enabled and train_sa_enabled:
                errors.append(
                    "training.semantic_aux.enabled=true だが"
                    " model.semantic_aux.enabled=false です。"
                    " 両方揃えてください。")

            # CQ-0265: Stage2a requires shanten_hint / discard_ukeire_hint
            enc_cfg = cfg.feature_encoder
            if not _parse_encoder_flag(enc_cfg, "shanten_hint"):
                errors.append(
                    "Stage2a では feature_encoder.shanten_hint=true が必須です。")
            if not _parse_encoder_flag(enc_cfg, "discard_ukeire_hint"):
                errors.append(
                    "Stage2a では feature_encoder.discard_ukeire_hint=true が必須です。")

        return errors

    def run(self) -> dict:
        """実験を実行して結果を返す

        Returns:
            結果 dict: run_dir, phases, selfplay_stats, train_metrics, eval_metrics
        """
        # バリデーション
        errors = self.validate_config()
        if errors:
            raise ValueError(
                "config バリデーションエラー:\n" + "\n".join(f"  - {e}" for e in errors))

        result = {}
        phases = self._get_phases()
        result["phases"] = phases
        total_phases = len(phases) + 1  # +1 for init
        phase_status: dict[str, str] = {}
        # CQ-0115: phase_action は今回の実行動作を記録（skipped/reused/executed）
        phase_action: dict[str, str] = {}

        # Global seed 固定
        self._global_seed = self._setup_global_seed()
        result["global_seed"] = self._global_seed

        # resume / reuse で完了済み phase を判定 (CQ-0110, CQ-0111)
        completed_phases: set[str] = set()

        # 1. Run directory 初期化
        phase_num = 1
        logger.info(f"[Phase {phase_num}/{total_phases}] run directory 初期化")

        if self._resume_run_dir is not None:
            # CQ-0111: phase 単位 resume
            run_dir = self._resume_run_dir
            manifest = self._load_manifest(run_dir)
            if manifest is None:
                raise ValueError(
                    f"resume 対象の run_dir に artifacts_manifest.json がありません: {run_dir}")
            completed_phases = self._get_completed_phases(manifest)
            self._validate_artifacts(run_dir, manifest, completed_phases)
            # CQ-0115: resume 時は過去の phase_status を復元
            prev_summary_path = run_dir / "summary.json"
            if prev_summary_path.exists():
                try:
                    with open(prev_summary_path) as f:
                        prev_summary = json.load(f)
                    for p, s in prev_summary.get("phase_status", {}).items():
                        phase_status[p] = s
                except (json.JSONDecodeError, OSError):
                    pass
            logger.info(f"  resume モード: 完了済み phase={sorted(completed_phases)}")
        else:
            run_dir = RunDirectory(base_dir=self._base_dir).create(self._config)

        result["run_dir"] = str(run_dir)
        logger.info(f"  run_dir: {run_dir}")

        # run.log 用 FileHandler 追加
        file_handler = self._setup_file_logging(run_dir)

        # モデル・エンコーダ生成
        encoder = self._create_encoder()
        model = self._create_model(encoder)
        obs_mode = self._config.experiment.get("observation_mode", "full")

        # CQ-0121: 入力次元を記録
        import math
        result["input_dim"] = math.prod(encoder.metadata().output_shape)

        # CQ-0110: 成果物再利用
        if self._reuse_from is not None:
            ref_dir = Path(self._reuse_from["run_dir"])
            ref_manifest = self._load_manifest(ref_dir)
            if ref_manifest is None:
                raise ValueError(
                    f"参照元に artifacts_manifest.json がありません: {ref_dir}")
            reuse_phases = set(self._reuse_from.get("phases", []))
            self._validate_artifacts(ref_dir, ref_manifest, reuse_phases)
            self._copy_reused_artifacts(
                run_dir, ref_dir, reuse_phases, ref_manifest, result, phase_status)
            completed_phases = completed_phases | reuse_phases
            # CQ-0115: reuse された phase を phase_action に記録
            for rp in reuse_phases:
                if phase_status.get(rp) == "reused":
                    phase_action[rp] = "reused"
            result["reuse_info"] = {
                "ref_run_dir": str(ref_dir),
                "reused_phases": sorted(reuse_phases),
            }
            logger.info(f"  reuse モード: ref={ref_dir}, phases={sorted(reuse_phases)}")

        # resume/reuse 時に imitation checkpoint をモデルに読み込む
        # CQ-0114: selfplay 再利用時は checkpoint_imitation.pt の存在とロードを必須化
        #   ただし imitation フェーズが実験に含まれない場合はスキップ
        has_imitation_phase = "imitation" in phases
        if "imitation" in completed_phases or \
                ("selfplay" in completed_phases and has_imitation_phase):
            imi_ckpt = run_dir / "checkpoints" / "checkpoint_imitation.pt"
            # selfplay 再利用時は参照元からの checkpoint コピーを試みる (CQ-0114)
            if not imi_ckpt.exists() and self._reuse_from is not None:
                ref_dir = Path(self._reuse_from["run_dir"])
                ref_ckpt = ref_dir / "checkpoints" / "checkpoint_imitation.pt"
                if ref_ckpt.exists():
                    import shutil
                    imi_ckpt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ref_ckpt, imi_ckpt)
                    logger.info(f"  参照元から imitation checkpoint をコピー: {ref_ckpt}")

            if imi_ckpt.exists():
                ckpt_data = torch.load(imi_ckpt, map_location="cpu", weights_only=True)
                # Learner.save_checkpoint は {"model_state_dict": ..., "optimizer_state_dict": ...} 形式
                if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
                    model.load_state_dict(ckpt_data["model_state_dict"])
                else:
                    model.load_state_dict(ckpt_data)
                result["loaded_checkpoint"] = str(imi_ckpt)
                logger.info(f"  imitation checkpoint を読み込みました: {imi_ckpt}")
            elif "selfplay" in completed_phases and "learner" not in completed_phases:
                # selfplay を再利用するのに checkpoint がない場合はエラー (CQ-0114)
                raise ValueError(
                    "selfplay 再利用時に checkpoint_imitation.pt が見つかりません。"
                    " learner 比較には同一初期方策が必要です。"
                    f" 確認先: {imi_ckpt}")

        # デバイス解決と記録
        result["resolved_devices"] = self._resolve_all_devices()
        logger.info(f"  devices: {result['resolved_devices']}")

        # プロファイラ (CQ-0098)
        profiler = Profiler(
            enabled=self._config.profiling.get("enabled", False))
        result["_profiler"] = profiler

        run_start = datetime.now(timezone.utc)
        phase_timing: dict[str, dict] = {}
        result["phase_timing"] = phase_timing
        # CQ-0115: phase_action を result に格納（_save_summary で利用）
        result["_phase_action"] = phase_action

        def _record_start(name: str) -> str:
            ts = _utc_now_str()
            phase_timing[name] = {"start_ts": ts, "end_ts": None, "duration_sec": None}
            return ts

        def _record_end(name: str) -> None:
            ts = _utc_now_str()
            entry = phase_timing[name]
            entry["end_ts"] = ts
            start_dt = datetime.fromisoformat(entry["start_ts"].replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            entry["duration_sec"] = round((end_dt - start_dt).total_seconds(), 3)

        for phase in phases:
            phase_num += 1
            label = f"[Phase {phase_num}/{total_phases}]"

            # CQ-0111: 完了済み phase のスキップ (resume/reuse)
            if phase in completed_phases:
                logger.info(f"{label} {phase} はスキップ（完了済み）")
                # CQ-0115: phase_status は過去の成功を維持、phase_action に今回動作を記録
                if phase not in phase_status:
                    phase_status[phase] = "success"
                # reuse 経由で既に phase_action が設定されている場合はそちらを優先
                if phase not in phase_action:
                    phase_action[phase] = "skipped"
                self._restore_phase_result(run_dir, phase, result)
                continue

            if phase == "imitation":
                logger.info(f"{label} imitation warm start")
                _record_start("imitation")
                profiler.start("imitation_total")
                try:
                    result["imitation_metrics"] = self._run_imitation(
                        run_dir, model, encoder, profiler)
                    phase_status["imitation"] = "success"
                    profiler.stop("imitation_total")
                    _record_end("imitation")
                except Exception as e:
                    logger.error(f"  imitation フェーズで失敗: {e}")
                    result["error"] = f"imitation: {e}"
                    phase_status["imitation"] = "failed"
                    result["total_duration_sec"] = round(
                        (datetime.now(timezone.utc) - run_start).total_seconds(), 3)
                    self._finalize(run_dir, result, phase_status, file_handler)
                    return result

            elif phase == "selfplay":
                # CQ-0232: Stage2a multi-cycle 時は cycle 内で selfplay するのでスキップ
                _stage = self._config.experiment.get("stage", "stage1")
                _mc = self._config.training.get("multi_cycle", {})
                if _stage == "stage2a" and _mc.get("enabled", False):
                    logger.info(f"{label} selfplay スキップ (Stage2a multi-cycle)")
                    phase_status["selfplay"] = "success"
                    phase_action["selfplay"] = "skipped"
                    continue

                logger.info(f"{label} self-play データ生成")
                _record_start("selfplay")
                profiler.start("selfplay_total")
                try:
                    result["selfplay_stats"] = self._run_selfplay(
                        run_dir, model, encoder, profiler)
                    phase_status["selfplay"] = "success"
                    profiler.stop("selfplay_total")
                    _record_end("selfplay")
                except Exception as e:
                    logger.error(f"  self-play フェーズで失敗: {e}")
                    result["error"] = f"selfplay: {e}"
                    phase_status["selfplay"] = "failed"
                    result["total_duration_sec"] = round(
                        (datetime.now(timezone.utc) - run_start).total_seconds(), 3)
                    self._finalize(run_dir, result, phase_status, file_handler)
                    return result

            elif phase == "learner":
                # CQ-0225/0226/0232: Stage2a learner 分岐
                stage = self._config.experiment.get("stage", "stage1")
                if stage == "stage2a":
                    mc_cfg = self._config.training.get("multi_cycle", {})
                    mc_enabled = mc_cfg.get("enabled", False)
                    num_cycles = mc_cfg.get("num_cycles", 1) if mc_enabled else 1

                    if num_cycles > 1:
                        # CQ-0232: Stage2a multi-cycle
                        logger.info(f"{label} Stage2a multi-cycle ({num_cycles} cycles)")
                        _record_start("learner")
                        try:
                            last_tm, cycles_list = \
                                self._run_stage2a_multi_cycle(
                                    run_dir, encoder, mc_cfg, num_cycles)
                            result["train_metrics"] = last_tm
                            result["cycles"] = cycles_list
                            # top-level に最終 cycle を昇格
                            if cycles_list:
                                last = cycles_list[-1]
                                result["selfplay_stats"] = last.get("selfplay_stats", {})
                                if last.get("eval_metrics"):
                                    result["eval_metrics"] = last["eval_metrics"]
                            phase_status["learner"] = "success"
                            _record_end("learner")
                            phase_action["learner"] = "executed"
                        except Exception as e:
                            logger.error(f"  Stage2a multi-cycle で失敗: {e}")
                            result["error"] = f"learner: {e}"
                            phase_status["learner"] = "failed"
                            result["total_duration_sec"] = round(
                                (datetime.now(timezone.utc) - run_start).total_seconds(), 3)
                            self._finalize(run_dir, result, phase_status, file_handler)
                            return result
                    else:
                        # single cycle
                        logger.info(f"{label} Stage2a learner")
                        _record_start("learner")
                        try:
                            result["train_metrics"] = self._run_learner_stage2a(
                                run_dir, encoder)
                            phase_status["learner"] = "success"
                            _record_end("learner")
                            phase_action["learner"] = "executed"
                        except Exception as e:
                            logger.error(f"  Stage2a learner で失敗: {e}")
                            result["error"] = f"learner: {e}"
                            phase_status["learner"] = "failed"
                            result["total_duration_sec"] = round(
                                (datetime.now(timezone.utc) - run_start).total_seconds(), 3)
                            self._finalize(run_dir, result, phase_status, file_handler)
                            return result
                    continue

                # CQ-0179: multi-cycle 判定
                mc_cfg = self._config.training.get("multi_cycle", {})
                mc_enabled = mc_cfg.get("enabled", False)
                num_cycles = mc_cfg.get("num_cycles", 1) if mc_enabled else 1

                if num_cycles > 1:
                    # --- multi-cycle 反復 (CQ-0179, CQ-0181, CQ-0184/0185/0186) ---
                    logger.info(f"{label} multi-cycle 学習 ({num_cycles} cycles)")
                    _record_start("learner")
                    profiler.start("learner_total")
                    cycles_data: list[dict] = []
                    _mc_eval_done = False  # CQ-0181: eval 実施有無を追跡
                    try:
                        sp_matches = mc_cfg.get(
                            "selfplay_matches_per_cycle",
                            self._config.selfplay.get("num_matches", 10))
                        eval_each = mc_cfg.get("eval_each_cycle", True)
                        orig_num_matches = self._config.selfplay.get("num_matches", 10)
                        orig_seed_start = self._config.selfplay.get("seed_start", 0)

                        # CQ-0184: rule_mix 設定
                        rm_cfg = self._config.training.get("rule_mix", {})
                        rm_enabled = rm_cfg.get("enabled", False)
                        rm_policy_ratio = rm_cfg.get("policy_ratio", 1.0)
                        rm_save_baseline = rm_cfg.get("save_baseline_actions", False)
                        orig_policy_ratio = self._config.selfplay.get("policy_ratio", 0.5)
                        orig_save_baseline = self._config.selfplay.get("save_baseline_actions", False)

                        # CQ-0185: rule_mix_learner 設定
                        rml_cfg = self._config.training.get("rule_mix_learner", {})
                        rml_enabled = rml_cfg.get("enabled", False)
                        rml_bl_epochs = rml_cfg.get("baseline_imitation_epochs", 0)
                        rml_ppo_epochs = rml_cfg.get("policy_ppo_epochs",
                                                      self._config.training.get("epochs", 4))
                        rml_order = rml_cfg.get("order", "baseline_then_policy")
                        rml_ppo_mode = rml_cfg.get("ppo_mode", "separated")
                        if rml_enabled and rml_order != "baseline_then_policy" and rml_ppo_mode == "separated":
                            raise ValueError(
                                f"training.rule_mix_learner.order は"
                                f" 'baseline_then_policy' のみ対応: {rml_order!r}")
                        if rml_ppo_mode not in ("separated", "mixed"):
                            raise ValueError(
                                f"training.rule_mix_learner.ppo_mode は"
                                f" 'separated' or 'mixed': {rml_ppo_mode!r}")

                        for ci in range(num_cycles):
                            cyc_label = f"cycle_{ci:02d}"
                            cyc_dir = run_dir / cyc_label
                            cyc_dir.mkdir(parents=True, exist_ok=True)
                            logger.info(f"  === {cyc_label} ({ci+1}/{num_cycles}) ===")
                            cycle_entry: dict = {"cycle_index": ci}

                            # cycle eval_before
                            if eval_each and "eval" in phases:
                                try:
                                    eb_dir = cyc_dir / "eval_before"
                                    eb = self._run_eval(
                                        run_dir, model, encoder, obs_mode,
                                        eval_dir_override=eb_dir)
                                    cycle_entry["eval_before"] = eb
                                    logger.info(
                                        "    eval_before avg_rank: %s, avg_score: %s",
                                        eb.get("avg_rank", "?"),
                                        eb.get("avg_score", "?"),
                                    )
                                except Exception as e:
                                    logger.warning(f"    cycle eval_before をスキップ: {e}")

                            # CQ-0184: cycle selfplay — rule_mix 適用
                            cyc_sp_dir = cyc_dir / "selfplay"
                            self._config.selfplay["num_matches"] = sp_matches
                            self._config.selfplay["seed_start"] = orig_seed_start + ci * sp_matches
                            if rm_enabled:
                                self._config.selfplay["policy_ratio"] = rm_policy_ratio
                                self._config.selfplay["save_baseline_actions"] = rm_save_baseline
                            try:
                                sp_stats = self._run_selfplay(
                                    cyc_dir, model, encoder, profiler)
                            finally:
                                self._config.selfplay["num_matches"] = orig_num_matches
                                self._config.selfplay["seed_start"] = orig_seed_start
                                if rm_enabled:
                                    self._config.selfplay["policy_ratio"] = orig_policy_ratio
                                    self._config.selfplay["save_baseline_actions"] = orig_save_baseline
                            cycle_entry["selfplay_stats"] = {
                                "total_steps": sp_stats.get("total_steps", 0),
                                "num_matches": sp_stats.get("num_matches", 0),
                            }
                            # CQ-0186, CQ-0190: actor_type_counts (shard から集計)
                            atc = self._count_actor_types(cyc_dir)
                            if atc:
                                cycle_entry["actor_type_counts"] = atc
                            logger.info(f"    selfplay steps: {sp_stats.get('total_steps', 0)}")

                            # CQ-0185, CQ-0187, CQ-0188, CQ-0192: cycle learner
                            learner_stages: dict = {}
                            if rml_enabled and rm_enabled and rml_ppo_mode == "mixed":
                                # CQ-0192: mixed PPO — baseline/policy 混合で1段学習
                                logger.info(f"    mixed PPO ({rml_ppo_epochs} epochs)")
                                tm = self._run_learner(
                                    run_dir, cyc_sp_dir, model, profiler,
                                    checkpoint_tag=f"cycle_{ci:02d}",
                                    override_epochs=rml_ppo_epochs,
                                    filter_actor_type=None)  # 混合: filter なし
                                learner_stages["mixed_ppo"] = {
                                    "executed": True,
                                    "used_samples": tm.get("total_steps", 0),
                                    "policy_loss": tm.get("policy_loss", 0.0),
                                    "mode": "mixed",
                                }
                                ppo_diag = tm.get("ppo_diag")
                                if ppo_diag is not None:
                                    learner_stages["mixed_ppo"]["ppo_diag"] = ppo_diag
                            elif rml_enabled and rm_enabled:
                                # separated: baseline BC → policy PPO
                                if rml_bl_epochs > 0:
                                    logger.info(f"    baseline BC ({rml_bl_epochs} epochs)")
                                    _bl_count = cycle_entry.get("actor_type_counts", {}).get("baseline", 0)
                                    if _bl_count == 0:
                                        logger.info("    baseline サンプル 0 件 → BC スキップ")
                                        learner_stages["baseline_imitation"] = {
                                            "executed": False,
                                            "skipped_reason": "no_baseline_samples",
                                        }
                                    else:
                                        bl_tm = self._run_learner(
                                            run_dir, cyc_sp_dir, model, profiler,
                                            checkpoint_tag=f"cycle_{ci:02d}_bl",
                                            override_algorithm="imitation",
                                            override_epochs=rml_bl_epochs,
                                            filter_actor_type="baseline")
                                        learner_stages["baseline_imitation"] = {
                                            "executed": True,
                                            "used_samples": bl_tm.get("total_steps", 0),
                                            "policy_loss": bl_tm.get("policy_loss", 0.0),
                                            "teacher_top1_match_rate": bl_tm.get("teacher_top1_match_rate"),
                                            "teacher_best_set_hit_rate": bl_tm.get("teacher_best_set_hit_rate"),
                                        }

                                logger.info(f"    policy PPO ({rml_ppo_epochs} epochs)")
                                tm = self._run_learner(
                                    run_dir, cyc_sp_dir, model, profiler,
                                    checkpoint_tag=f"cycle_{ci:02d}",
                                    override_epochs=rml_ppo_epochs,
                                    filter_actor_type="policy")
                                learner_stages["policy_ppo"] = {
                                    "executed": True,
                                    "used_samples": tm.get("total_steps", 0),
                                    "policy_loss": tm.get("policy_loss", 0.0),
                                }
                                ppo_diag = tm.get("ppo_diag")
                                if ppo_diag is not None:
                                    learner_stages["policy_ppo"]["ppo_diag"] = ppo_diag
                            else:
                                # CQ-0187: rule_mix ON 時は PPO を policy-only に強制
                                _filter = "policy" if rm_enabled else None
                                tm = self._run_learner(
                                    run_dir, cyc_sp_dir, model, profiler,
                                    checkpoint_tag=f"cycle_{ci:02d}",
                                    filter_actor_type=_filter)

                            cycle_entry["train_metrics"] = {
                                "policy_loss": tm.get("policy_loss", 0.0),
                                "value_loss": tm.get("value_loss", 0.0),
                                "total_steps": tm.get("total_steps", 0),
                                "num_updates": tm.get("num_updates", 0),
                            }
                            ppo_diag = tm.get("ppo_diag")
                            if ppo_diag is not None:
                                cycle_entry["learner_diag"] = ppo_diag
                            if learner_stages:
                                cycle_entry["learner_stages"] = learner_stages
                            logger.info(f"    policy_loss: {tm['policy_loss']:.4f}")

                            # cycle eval_after
                            if eval_each and "eval" in phases:
                                try:
                                    ea_dir = cyc_dir / "eval"
                                    ea = self._run_eval(
                                        run_dir, model, encoder, obs_mode,
                                        eval_dir_override=ea_dir)
                                    cycle_entry["eval"] = ea
                                    _mc_eval_done = True
                                    logger.info(
                                        "    eval avg_rank: %s, avg_score: %s",
                                        ea.get("avg_rank", "?"),
                                        ea.get("avg_score", "?"),
                                    )
                                    # eval_diff
                                    if "eval_before" in cycle_entry:
                                        cycle_entry["eval_diff"] = compute_eval_diff(
                                            cycle_entry["eval_before"], ea)
                                except Exception as e:
                                    logger.warning(f"    cycle eval をスキップ: {e}")

                            cycles_data.append(cycle_entry)

                        # CQ-0181: 最終 cycle 基準で result に反映 (single-cycle 互換)
                        last = cycles_data[-1]
                        result["train_metrics"] = tm
                        # eval_before/eval_diff は最終 cycle 基準
                        if "eval_before" in last:
                            result["eval_before"] = last["eval_before"]
                        if "eval" in last:
                            result["eval_metrics"] = last["eval"]
                        if "eval_before" in last and "eval" in last:
                            result["eval_diff"] = compute_eval_diff(
                                last["eval_before"], last["eval"])
                        result["cycles"] = cycles_data
                        phase_status["learner"] = "success"
                        # CQ-0181: eval の phase_status は実際に eval を実行した場合のみ
                        if _mc_eval_done:
                            phase_status["eval"] = "success"
                        profiler.stop("learner_total")
                        _record_end("learner")
                    except Exception as e:
                        logger.error(f"  multi-cycle で失敗: {e}")
                        result["error"] = f"multi-cycle: {e}"
                        result["cycles"] = cycles_data
                        phase_status["learner"] = "failed"
                        result["total_duration_sec"] = round(
                            (datetime.now(timezone.utc) - run_start).total_seconds(), 3)
                        self._finalize(run_dir, result, phase_status, file_handler)
                        return result
                else:
                    # --- single-cycle (既存動作) ---
                    # 学習前評価 (eval も phases に含まれる場合のみ)
                    if "eval" in phases:
                        if "eval_before" in completed_phases:
                            logger.info(f"{label} eval_before はスキップ（完了済み）")
                            phase_action["eval_before"] = "skipped"
                            self._restore_phase_result(run_dir, "eval_before", result)
                        else:
                            _record_start("eval_before")
                            try:
                                logger.info(f"{label} 学習前評価 (eval_before)")
                                eval_before_dir = run_dir / "eval_before"
                                result["eval_before"] = self._run_eval(
                                    run_dir, model, encoder, obs_mode,
                                    eval_dir_override=eval_before_dir)
                                logger.info(
                                    "  eval_before avg_rank: %s, avg_score: %s",
                                    result["eval_before"].get("avg_rank", "?"),
                                    result["eval_before"].get("avg_score", "?"),
                                )
                                _record_end("eval_before")
                            except Exception as e:
                                logger.warning(f"  学習前評価をスキップ: {e}")
                                _record_end("eval_before")

                    logger.info(f"{label} learner 学習")
                    _record_start("learner")
                    profiler.start("learner_total")
                    try:
                        selfplay_dir = run_dir / "selfplay"
                        result["train_metrics"] = self._run_learner(
                            run_dir, selfplay_dir, model, profiler)
                        phase_status["learner"] = "success"
                        profiler.stop("learner_total")
                        _record_end("learner")
                    except Exception as e:
                        logger.error(f"  learner フェーズで失敗: {e}")
                        result["error"] = f"learner: {e}"
                        phase_status["learner"] = "failed"
                        result["total_duration_sec"] = round(
                            (datetime.now(timezone.utc) - run_start).total_seconds(), 3)
                        self._finalize(run_dir, result, phase_status, file_handler)
                        return result

            elif phase == "eval":
                # multi-cycle 時は learner phase 内で eval 済み
                if result.get("cycles"):
                    if "eval_metrics" in result:
                        phase_status["eval"] = "success"
                    continue

                # CQ-0230: Stage2a eval 分岐
                stage = self._config.experiment.get("stage", "stage1")
                if stage == "stage2a":
                    logger.info(f"{label} Stage2a evaluator 評価")
                    _record_start("eval")
                    try:
                        result["eval_metrics"] = self._run_eval_stage2a(
                            run_dir, encoder)
                        phase_status["eval"] = "success"
                        _record_end("eval")
                        phase_action["eval"] = "executed"
                    except Exception as e:
                        logger.error(f"  Stage2a eval フェーズで失敗: {e}")
                        result["error"] = f"eval: {e}"
                        phase_status["eval"] = "failed"
                        result["total_duration_sec"] = round(
                            (datetime.now(timezone.utc) - run_start).total_seconds(), 3)
                        self._finalize(run_dir, result, phase_status, file_handler)
                        return result
                    continue

                logger.info(f"{label} evaluator 評価")
                _record_start("eval")
                profiler.start("eval_total")
                try:
                    result["eval_metrics"] = self._run_eval(
                        run_dir, model, encoder, obs_mode)
                    phase_status["eval"] = "success"
                    profiler.stop("eval_total")
                    _record_end("eval")

                    # 学習前後差分レポート生成
                    if "eval_before" in result:
                        diff = compute_eval_diff(
                            result["eval_before"], result["eval_metrics"])
                        result["eval_diff"] = diff
                        diff_path = run_dir / "eval" / "eval_diff.json"
                        diff_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(diff_path, "w") as f:
                            json.dump(diff, f, indent=2, ensure_ascii=False)
                        logger.info(f"  eval_diff: avg_rank {diff['avg_rank']['delta']:+.3f}")
                except Exception as e:
                    logger.error(f"  evaluator フェーズで失敗: {e}")
                    result["error"] = f"evaluator: {e}"
                    phase_status["eval"] = "failed"
                    result["total_duration_sec"] = round(
                        (datetime.now(timezone.utc) - run_start).total_seconds(), 3)
                    self._finalize(run_dir, result, phase_status, file_handler)
                    return result

        result["total_duration_sec"] = round(
            (datetime.now(timezone.utc) - run_start).total_seconds(), 3)
        logger.info("実験完了")
        self._finalize(run_dir, result, phase_status, file_handler)
        return result

    def _finalize(self, run_dir: Path, result: dict,
                  phase_status: dict[str, str],
                  file_handler: logging.FileHandler) -> None:
        """run 終了時の共通処理: summary 保存・notes 追記・プロファイル保存・ログ後始末"""
        # プロファイル保存 (CQ-0098)
        profiler: Profiler | None = result.pop("_profiler", None)
        if profiler is not None:
            profiler.save(run_dir / "profile.json")
            if profiler.enabled:
                result["profiling"] = profiler.to_dict()
        self._save_summary(run_dir, result, phase_status)
        self._save_manifest(run_dir, result, phase_status)  # CQ-0109
        self._append_notes(run_dir, result, phase_status)
        self._teardown_file_logging(file_handler)

    def _run_imitation(self, run_dir: Path, model, encoder,
                       profiler=None) -> dict:
        """imitation warm start フェーズ (CQ-0206: multi-chunk 対応, CQ-0224: stage2a)"""
        stage = self._config.experiment.get("stage", "stage1")
        if stage == "stage2a":
            return self._run_imitation_stage2a(run_dir, encoder, profiler)

        mci = self._config.training.get("multi_chunk_imitation", {})
        if mci.get("enabled", False):
            return self._run_imitation_multi_chunk(
                run_dir, model, encoder, mci, profiler)
        return self._run_imitation_single(run_dir, model, encoder, profiler)

    def _run_imitation_single(self, run_dir: Path, model, encoder,
                              profiler=None) -> dict:
        """単発 imitation (従来互換)"""
        sp_cfg = self._config.selfplay
        imitation_dir = run_dir / "imitation"
        num_workers = self._config.imitation.get("num_workers", 1)

        imi_matches = sp_cfg.get("imitation_matches",
                                 sp_cfg.get("num_matches", 10))
        sp_stats = self._generate_imitation_data(
            imitation_dir, model, encoder, imi_matches, num_workers, profiler)

        logger.info(f"  imitation data: {sp_stats['total_steps']} steps")

        metrics, learner_obj = self._train_imitation(run_dir, model, imitation_dir, profiler)
        learner_obj.save_checkpoint(tag="imitation")
        self._log_imitation_metrics(metrics)
        metrics["data_generation"] = {
            "total_steps": sp_stats.get("total_steps", 0),
            "num_matches": sp_stats.get("num_matches", 0),
            "num_workers": sp_stats.get("num_workers", 1),
            "seed_strategy": sp_stats.get("seed_strategy"),
        }
        return metrics

    def _run_imitation_multi_chunk(self, run_dir: Path, model, encoder,
                                   mci: dict, profiler=None) -> dict:
        """CQ-0206: multi-chunk imitation"""
        num_chunks = mci.get("num_chunks", 1)
        matches_per_chunk = mci.get("imitation_matches_per_chunk", 10)
        if num_chunks < 1:
            raise ValueError(f"multi_chunk_imitation.num_chunks は 1 以上: {num_chunks}")
        if matches_per_chunk < 1:
            raise ValueError(f"multi_chunk_imitation.imitation_matches_per_chunk は 1 以上: {matches_per_chunk}")

        sp_cfg = self._config.selfplay
        num_workers = self._config.imitation.get("num_workers", 1)
        base_seed = sp_cfg.get("imitation_seed_start",
                               sp_cfg.get("seed_start", 0))

        chunk_results: list[dict] = []
        final_metrics: dict = {}

        for ci in range(num_chunks):
            chunk_label = f"chunk_{ci:02d}"
            logger.info(f"  [imitation {chunk_label}] データ生成開始 ({matches_per_chunk} matches)")
            chunk_dir = run_dir / "imitation" / chunk_label

            # seed を chunk ごとにずらす
            chunk_seed_start = base_seed + ci * matches_per_chunk

            sp_stats = self._generate_imitation_data(
                chunk_dir, model, encoder, matches_per_chunk,
                num_workers, profiler, seed_start=chunk_seed_start)

            logger.info(f"  [{chunk_label}] data: {sp_stats['total_steps']} steps")

            # imitation 学習
            metrics, last_learner = self._train_imitation(
                run_dir, model, chunk_dir, profiler)
            self._log_imitation_metrics(metrics, prefix=f"  [{chunk_label}]")

            chunk_results.append({
                "chunk_index": ci,
                "num_matches": sp_stats.get("num_matches", 0),
                "total_steps": sp_stats.get("total_steps", 0),
                "policy_loss": metrics.get("policy_loss"),
                "teacher_top1_match_rate": metrics.get("teacher_top1_match_rate"),
                "teacher_best_set_hit_rate": metrics.get("teacher_best_set_hit_rate"),
                "value_loss": metrics.get("value_loss"),
            })
            final_metrics = metrics

        # 最終 checkpoint 保存 (学習済み Learner から保存)
        last_learner.save_checkpoint(tag="imitation")

        # 最終 chunk の metrics をベースに multi-chunk 情報を付加
        final_metrics["multi_chunk_imitation"] = {
            "enabled": True,
            "num_chunks": num_chunks,
            "imitation_matches_per_chunk": matches_per_chunk,
        }
        final_metrics["chunks"] = chunk_results
        final_metrics["data_generation"] = {
            "total_steps": sum(c["total_steps"] for c in chunk_results),
            "num_matches": sum(c["num_matches"] for c in chunk_results),
            "num_workers": num_workers,
        }
        return final_metrics

    def _generate_imitation_data(self, output_dir: Path, model, encoder,
                                 num_matches: int, num_workers: int,
                                 profiler=None, seed_start: int | None = None) -> dict:
        """baseline 教師データ生成 (single/multi-process)"""
        sp_cfg = self._config.selfplay
        imi_config = dict(self._as_dict())
        imi_sp = dict(sp_cfg)
        imi_sp["save_baseline_actions"] = True
        imi_sp["policy_ratio"] = 0.0
        imi_config["selfplay"] = imi_sp

        if seed_start is None:
            seed_start = sp_cfg.get("imitation_seed_start",
                                    sp_cfg.get("seed_start", 0))

        if num_workers > 1:
            return self._run_imitation_parallel(
                output_dir, model, imi_config, num_matches, num_workers,
                base_seed=seed_start)
        sp_device = resolve_device(sp_cfg.get("inference_device", "auto"))
        worker = SelfPlayWorker(
            config=imi_config,
            model=model,
            encoder=encoder,
            output_dir=output_dir,
            inference_device=sp_device,
            profiler=profiler,
        )
        return worker.run(num_matches=num_matches, seed_start=seed_start)

    def _train_imitation(self, run_dir: Path, model,
                         shard_dir: Path, profiler=None,
                         ) -> tuple[dict, "Learner"]:
        """imitation 学習 (共通 helper)

        Returns:
            (metrics, learner): 学習結果と学習済み Learner
        """
        training_device = resolve_device(
            self._config.training.get("device", "auto"))
        learner = Learner(
            config=self._make_imitation_train_config(),
            model=model,
            run_dir=run_dir,
            device=training_device,
        )
        # CQ-0216: imitation_optimizer.epochs → imitation_epochs → epochs の優先順
        imi_opt = self._config.training.get("imitation_optimizer", {})
        imi_epochs = imi_opt.get(
            "epochs",
            self._config.training.get("imitation_epochs",
                                       self._config.training.get("epochs", 4)))
        imi_filter = self._config.training.get("imitation_filter", None)
        metrics = learner.train(
            shard_dir,
            num_epochs=imi_epochs,
            filter_actor_type="baseline",
            imitation_filter=imi_filter,
            profiler=profiler,
        )
        return metrics, learner

    def _make_imitation_train_config(self) -> dict:
        """imitation 用 training config を構築する (CQ-0209)

        training.imitation_optimizer が指定されていれば、
        imitation phase の optimizer 設定だけを上書きする。
        未指定なら従来どおり training.* を使う。
        """
        cfg = dict(self._as_dict())
        cfg["training"] = dict(cfg["training"])
        cfg["training"]["algorithm"] = "imitation"
        # CQ-0209: imitation 専用 optimizer 設定で上書き
        imi_opt = cfg["training"].get("imitation_optimizer", {})
        if imi_opt:
            for key in ("lr", "batch_size", "epochs", "max_grad_norm"):
                if key in imi_opt:
                    cfg["training"][key] = imi_opt[key]
        return cfg

    @staticmethod
    def _log_imitation_metrics(metrics: dict, prefix: str = " ") -> None:
        """imitation 学習結果をログ出力する"""
        logger.info(f"{prefix} imitation loss: {metrics.get('policy_loss', 0):.4f}")
        top1 = metrics.get("teacher_top1_match_rate")
        best_set = metrics.get("teacher_best_set_hit_rate")
        if top1 is not None:
            msg = f"{prefix} teacher_top1_match_rate: {top1:.4f}"
            if best_set is not None:
                msg += f", teacher_best_set_hit_rate: {best_set:.4f}"
            logger.info(msg)
        tbm_status = metrics.get("teacher_best_set_status")
        if tbm_status is not None:
            logger.info(f"{prefix} teacher_best_set_status: {tbm_status}")
        loss_mode = metrics.get("imitation_loss_mode")
        if loss_mode is not None:
            logger.info(f"{prefix} imitation_loss_mode: {loss_mode}")

    def _run_imitation_parallel(
        self, imitation_dir: Path, model, imi_config: dict,
        num_matches: int, num_workers: int,
        base_seed: int | None = None,
    ) -> dict:
        """multi-process imitation 教師データ生成 (CQ-0207: base_seed 引き渡し)

        _run_selfplay_parallel と同じパターンで worker を起動し、
        imitation/worker_<id>/shard_*.parquet に保存する。
        """
        sp_cfg = self._config.selfplay
        imitation_dir.mkdir(parents=True, exist_ok=True)
        num_threads = sp_cfg.get("worker_num_threads", 1)
        if base_seed is None:
            base_seed = self._global_seed or 0
        obs_mode = self._config.experiment.get("observation_mode", "full")

        # model を一時ファイルに保存
        model_path = imitation_dir / "_imitation_model.pt"
        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict_cpu, model_path)

        model_config = dict(self._config.model)
        encoder_config = dict(self._config.feature_encoder)

        matches_per_worker = self._distribute_matches(num_matches, num_workers)

        ctx = mp.get_context("spawn")
        error_queue = ctx.Queue()
        processes = []

        try:
            for i, wm in enumerate(matches_per_worker):
                if wm == 0:
                    continue
                worker_seed = derive_worker_seed(base_seed, i)
                match_seeds = [derive_match_seed(worker_seed, j) for j in range(wm)]
                worker_output_dir = imitation_dir / f"worker_{i}"

                p = ctx.Process(
                    target=_selfplay_worker_fn,
                    args=(
                        i, str(model_path), imi_config, model_config,
                        encoder_config, obs_mode, wm, match_seeds,
                        str(worker_output_dir), num_threads, base_seed,
                        worker_seed, error_queue,
                    ),
                )
                p.start()
                processes.append(p)

            self._wait_and_check_workers(
                processes, error_queue=error_queue,
                worker_label="imitation worker")

            # 統計集約
            aggregated = self._aggregate_selfplay_stats(imitation_dir, num_workers)
            aggregated["num_workers"] = num_workers
            aggregated["seed_strategy"] = {
                "base_seed": base_seed,
                "method": "derive_worker_seed + derive_match_seed",
            }
            logger.info(
                f"  imitation data generated: {aggregated['total_steps']} steps "
                f"({num_workers} workers)")
            return aggregated
        finally:
            if model_path.exists():
                model_path.unlink()

    def _run_selfplay(self, run_dir: Path, model, encoder,
                      profiler=None) -> dict:
        """self-play フェーズ (CQ-0224: stage2a 対応)"""
        stage = self._config.experiment.get("stage", "stage1")
        if stage == "stage2a":
            return self._run_selfplay_stage2a(run_dir, encoder, profiler)

        sp_cfg = self._config.selfplay
        num_workers = sp_cfg.get("num_workers", 1)

        if num_workers > 1:
            return self._run_selfplay_parallel(run_dir, model, num_workers)

        # 単一 worker 経路
        selfplay_dir = run_dir / "selfplay"
        worker_output_dir = selfplay_dir / "worker_0"
        sp_device = resolve_device(
            sp_cfg.get("inference_device", "auto"))
        worker = SelfPlayWorker(
            config=self._as_dict(),
            model=model,
            encoder=encoder,
            output_dir=worker_output_dir,
            worker_id="worker_0",
            inference_device=sp_device,
            profiler=profiler,
        )
        sp_stats = worker.run(
            num_matches=sp_cfg.get("num_matches", 10),
            seed_start=sp_cfg.get("seed_start", 0),
        )
        # CQ-0142: 内部キーを除去（single worker は quantile 計算済み）
        sp_stats.pop("_reward_raw_values", None)
        logger.info(f"  total_steps: {sp_stats['total_steps']}")
        return sp_stats

    def _create_stage2a_model(self, encoder):
        """Stage2a model factory"""
        from mahjong_rl.models.stage2a_model import Stage2aModel
        meta = encoder.metadata()
        input_dim = int(np.prod(meta.output_shape))
        mc = self._config.model
        # CQ-0265: extract direct hint ranges for discard branch
        hint_ranges = {}
        for src in ("shanten_hint", "discard_ukeire_hint"):
            if src in meta.feature_ranges:
                hint_ranges[src] = meta.feature_ranges[src]
        return Stage2aModel(
            input_dim=input_dim,
            discard_hidden_dims=mc.get("discard_hidden_dims", [256, 128]),
            optional_hidden_dims=mc.get("optional_hidden_dims",
                mc.get("call_hidden_dims", [128, 64])),
            value_hidden_dims=mc.get("value_hidden_dims", [128, 64]),
            candidate_dim=mc.get("candidate_dim", 16),
            optional_scorer_hidden=mc.get("optional_scorer_hidden", 32),
            semantic_aux_config=mc.get("semantic_aux"),
            direct_hint_ranges=hint_ranges if hint_ranges else None,
        )

    def _run_imitation_stage2a(self, run_dir: Path, encoder, profiler=None) -> dict:
        """CQ-0236: Stage2a imitation (multi-chunk 対応)"""
        mci = self._config.training.get("multi_chunk_imitation", {})
        if mci.get("enabled", False):
            return self._run_imitation_stage2a_multi_chunk(
                run_dir, encoder, mci, profiler)
        return self._run_imitation_stage2a_single(run_dir, encoder, profiler)

    def _run_imitation_stage2a_single(self, run_dir, encoder, profiler=None):
        """Stage2a single imitation"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.models.stage2a_model import Stage2aModel
        from mahjong_rl.stage2a_learner import Stage2aLearner

        sp_cfg = self._config.selfplay
        imitation_dir = run_dir / "imitation"
        obs_mode = self._config.experiment.get("observation_mode", "full")
        num_matches = sp_cfg.get("imitation_matches",
                                  self._config.imitation.get("num_matches", 100))

        # 1. Data generation (single or parallel)
        imi_workers = self._config.imitation.get("num_workers", 1)
        if imi_workers > 1:
            from mahjong_rl.stage2a_parallel import run_stage2a_selfplay_parallel
            gen_stats = run_stage2a_selfplay_parallel(
                output_dir=imitation_dir,
                num_workers=imi_workers,
                num_matches=num_matches,
                base_seed=self._global_seed,
                obs_mode=obs_mode,
                encoder_config=dict(self._config.feature_encoder),
                model_state_path=None,  # imitation は baseline actor
                model_config=dict(self._config.model),
                experiment_id=self._config.experiment.get("name", "stage2a"),
                run_id=str(run_dir),
                inference_device=sp_cfg.get("inference_device", "cpu"),
                num_threads=sp_cfg.get("worker_num_threads", 1),
            )
        else:
            worker = Stage2SelfPlayWorker(
                config=self._as_dict(),
                output_dir=imitation_dir,
                observation_mode=obs_mode,
                encoder=encoder,
            )
            gen_stats = worker.generate(
                num_matches=num_matches,
                base_seed=self._global_seed,
                experiment_id=self._config.experiment.get("name", "stage2a"),
                run_id=str(run_dir),
            )
        logger.info(f"  Stage2a imitation data: {gen_stats}")

        # 2. Learner
        input_dim = int(np.prod(encoder.metadata().output_shape))
        model_cfg = self._config.model
        s2_model = self._create_stage2a_model(encoder)
        learner_config = self._as_dict()
        learner_config["training"]["algorithm"] = "imitation"
        # CQ-0231: imitation epoch 分離
        tc = self._config.training
        imi_epochs = tc.get("imitation_epochs", tc.get("epochs", 4))
        learner = Stage2aLearner(
            config=learner_config,
            model=s2_model,
            run_dir=run_dir,
        )
        train_metrics = learner.train(imitation_dir, num_epochs=imi_epochs)
        logger.info(f"  Stage2a imitation learner (epochs={imi_epochs}): {train_metrics}")

        # 3. Checkpoint
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(s2_model.state_dict(), ckpt_dir / "checkpoint_imitation.pt")

        # Store model for later phases
        self._stage2a_model = s2_model

        # CQ-0254: imitation eval
        imi_eval = self._config.training.get("imitation_eval", {})
        imitation_eval_metrics = None
        if imi_eval.get("enabled", False):
            logger.info("  Stage2a imitation eval start")
            imitation_eval_metrics = self._run_eval_stage2a(
                run_dir, encoder,
                num_matches_override=imi_eval.get("num_matches"),
                num_workers_override=imi_eval.get("num_workers"),
                seed_start_override=imi_eval.get("seed_start"),
                eval_subdir="imitation_eval",
            )
            logger.info(f"  Stage2a imitation eval: "
                         f"avg_rank={imitation_eval_metrics.get('avg_rank', 0):.2f}")

        return {
            "stage": "stage2a",
            "total_steps": gen_stats["total_steps"],
            "discard_count": gen_stats["discard_count"],
            "call_count": gen_stats["call_count"],
            "train_metrics": train_metrics,
            "imitation_epochs": imi_epochs,
            "imitation_eval": imitation_eval_metrics,
        }

    def _run_imitation_stage2a_multi_chunk(self, run_dir, encoder, mci, profiler=None):
        """CQ-0236: Stage2a multi-chunk imitation"""
        from mahjong_rl.models.stage2a_model import Stage2aModel
        from mahjong_rl.stage2a_learner import Stage2aLearner

        num_chunks = mci.get("num_chunks", 1)
        matches_per = mci.get("imitation_matches_per_chunk", 100)
        obs_mode = self._config.experiment.get("observation_mode", "full")
        tc = self._config.training
        imi_epochs = tc.get("imitation_epochs", tc.get("epochs", 4))

        input_dim = int(np.prod(encoder.metadata().output_shape))
        model_cfg = self._config.model
        s2_model = self._create_stage2a_model(encoder)

        import time as _time
        chunks_data = []
        for ci in range(num_chunks):
            chunk_dir = run_dir / "imitation" / f"chunk_{ci:02d}"
            chunk_t0 = _time.perf_counter()
            # data gen (single or parallel)
            logger.info(f"  chunk {ci}: data generation start")
            dg_t0 = _time.perf_counter()
            imi_workers = self._config.imitation.get("num_workers", 1)
            chunk_seed = self._global_seed + ci * matches_per
            if imi_workers > 1:
                from mahjong_rl.stage2a_parallel import run_stage2a_selfplay_parallel
                gen = run_stage2a_selfplay_parallel(
                    output_dir=chunk_dir,
                    num_workers=imi_workers,
                    num_matches=matches_per,
                    base_seed=chunk_seed,
                    obs_mode=obs_mode,
                    encoder_config=dict(self._config.feature_encoder),
                    model_state_path=None,
                    model_config=dict(self._config.model),
                    experiment_id=self._config.experiment.get("name", "stage2a"),
                    run_id=str(run_dir),
                    inference_device=self._config.selfplay.get("inference_device", "cpu"),
                    num_threads=self._config.selfplay.get("worker_num_threads", 1),
                )
            else:
                from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
                worker = Stage2SelfPlayWorker(
                    config=self._as_dict(), output_dir=chunk_dir,
                    observation_mode=obs_mode, encoder=encoder,
                )
                gen = worker.generate(
                    num_matches=matches_per,
                    base_seed=chunk_seed,
                    experiment_id=self._config.experiment.get("name", "stage2a"),
                    run_id=str(run_dir),
                )
            dg_sec = _time.perf_counter() - dg_t0
            logger.info(f"  chunk {ci}: data gen done ({dg_sec:.1f}s)")

            # learner
            logger.info(f"  chunk {ci}: learner start")
            lr_t0 = _time.perf_counter()
            lc = self._as_dict()
            lc["training"]["algorithm"] = "imitation"
            learner = Stage2aLearner(config=lc, model=s2_model, run_dir=run_dir)
            tm = learner.train(chunk_dir, num_epochs=imi_epochs)
            lr_wall = _time.perf_counter() - lr_t0
            chunk_sec = _time.perf_counter() - chunk_t0
            # CQ-0248: learner 内訳 (train + diagnostics)
            tm_timing = tm.get("timing", {})
            train_sec = tm_timing.get("train_sec", lr_wall)
            diag_sec = tm_timing.get("diagnostics_sec", 0)
            logger.info(f"  chunk {ci}: learner done "
                         f"(train={train_sec:.1f}s diag={diag_sec:.1f}s) "
                         f"steps={gen['total_steps']} loss={tm.get('policy_loss', 0):.4f} "
                         f"total={chunk_sec:.1f}s")
            # CQ-0254: chunk eval
            chunk_eval = None
            imi_eval = self._config.training.get("imitation_eval", {})
            if imi_eval.get("enabled", False) and imi_eval.get("eval_each_chunk", False):
                self._stage2a_model = s2_model
                logger.info(f"  chunk {ci}: eval start")
                chunk_eval = self._run_eval_stage2a(
                    run_dir, encoder,
                    seed_offset=ci * 200,
                    num_matches_override=imi_eval.get("num_matches"),
                    num_workers_override=imi_eval.get("num_workers"),
                    seed_start_override=imi_eval.get("seed_start"),
                    eval_subdir=f"imitation/chunk_{ci:02d}/eval",
                )
                logger.info(f"  chunk {ci}: eval avg_rank="
                             f"{chunk_eval.get('avg_rank', 0):.2f}")

            chunk_sec = _time.perf_counter() - chunk_t0
            chunks_data.append({
                "chunk_index": ci,
                "gen_stats": gen,
                "train_metrics": tm,
                "imitation_eval": chunk_eval,
                "timing": {
                    "data_generation_sec": round(dg_sec, 3),
                    "learner_sec": round(train_sec, 3),
                    "diagnostics_sec": round(diag_sec, 3),
                    "chunk_total_sec": round(chunk_sec, 3),
                },
            })

        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(s2_model.state_dict(), ckpt_dir / "checkpoint_imitation.pt")
        self._stage2a_model = s2_model

        # CQ-0254: final imitation eval (multi-chunk 完了後)
        imitation_eval_metrics = None
        imi_eval = self._config.training.get("imitation_eval", {})
        if imi_eval.get("enabled", False):
            self._stage2a_model = s2_model
            logger.info("  Stage2a imitation eval (final) start")
            imitation_eval_metrics = self._run_eval_stage2a(
                run_dir, encoder,
                num_matches_override=imi_eval.get("num_matches"),
                num_workers_override=imi_eval.get("num_workers"),
                seed_start_override=imi_eval.get("seed_start"),
                eval_subdir="imitation_eval",
            )
            logger.info(f"  Stage2a imitation eval (final): "
                         f"avg_rank={imitation_eval_metrics.get('avg_rank', 0):.2f}")

        last_tm = chunks_data[-1]["train_metrics"] if chunks_data else {}
        return {
            "stage": "stage2a",
            "total_steps": sum(c["gen_stats"]["total_steps"] for c in chunks_data),
            "discard_count": sum(c["gen_stats"]["discard_count"] for c in chunks_data),
            "call_count": sum(c["gen_stats"]["call_count"] for c in chunks_data),
            "train_metrics": last_tm,
            "imitation_epochs": imi_epochs,
            "imitation_eval": imitation_eval_metrics,
            "multi_chunk_imitation": {"enabled": True, "num_chunks": num_chunks,
                                       "chunks": chunks_data},
        }

    def _run_selfplay_stage2a(self, run_dir: Path, encoder,
                              profiler=None, output_dir: Path | None = None,
                              num_matches: int | None = None,
                              base_seed: int | None = None) -> dict:
        """CQ-0234: Stage2a selfplay (single / multi-process)"""
        sp_cfg = self._config.selfplay
        selfplay_dir = output_dir or (run_dir / "selfplay")
        obs_mode = self._config.experiment.get("observation_mode", "full")
        n_matches = num_matches or sp_cfg.get("num_matches", 10)
        b_seed = base_seed if base_seed is not None else sp_cfg.get("seed_start", 0)
        num_workers = sp_cfg.get("num_workers", 1)
        s2_model = getattr(self, "_stage2a_model", None)

        if num_workers > 1:
            from mahjong_rl.stage2a_parallel import run_stage2a_selfplay_parallel
            # model を一時保存
            model_path = None
            if s2_model is not None:
                selfplay_dir.mkdir(parents=True, exist_ok=True)
                model_path = str(selfplay_dir / "_model.pt")
                sd = {k: v.cpu() for k, v in s2_model.state_dict().items()}
                torch.save(sd, model_path)
            stats = run_stage2a_selfplay_parallel(
                output_dir=selfplay_dir,
                num_workers=num_workers,
                num_matches=n_matches,
                base_seed=b_seed,
                obs_mode=obs_mode,
                encoder_config=dict(self._config.feature_encoder),
                model_state_path=model_path,
                model_config=dict(self._config.model),
                experiment_id=self._config.experiment.get("name", "stage2a"),
                run_id=str(run_dir),
                inference_device=sp_cfg.get("inference_device", "cpu"),
                num_threads=sp_cfg.get("worker_num_threads", 1),
                policy_ratio=sp_cfg.get("policy_ratio", 1.0),
                save_baseline_actions=sp_cfg.get("save_baseline_actions", False),
            )
        else:
            from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
            worker = Stage2SelfPlayWorker(
                config=self._as_dict(),
                output_dir=selfplay_dir,
                observation_mode=obs_mode,
                encoder=encoder,
                model=s2_model,
                policy_ratio=sp_cfg.get("policy_ratio", 1.0),
                save_baseline_actions=sp_cfg.get("save_baseline_actions", False),
            )
            stats = worker.generate(
                num_matches=n_matches,
                base_seed=b_seed,
                experiment_id=self._config.experiment.get("name", "stage2a"),
                run_id=str(run_dir),
            )

        logger.info(f"  Stage2a selfplay: {stats}")
        stats["stage"] = "stage2a"
        return stats

    def _run_stage2a_multi_cycle(
        self, run_dir: Path, encoder, mc_cfg: dict, num_cycles: int,
    ) -> tuple[dict, list[dict]]:
        """CQ-0232: Stage2a multi-cycle (selfplay → learner → eval) × N"""
        from mahjong_rl.stage2_selfplay_worker import Stage2SelfPlayWorker
        from mahjong_rl.models.stage2a_model import Stage2aModel
        from mahjong_rl.stage2a_learner import Stage2aLearner

        sp_matches = mc_cfg.get(
            "selfplay_matches_per_cycle",
            self._config.selfplay.get("num_matches", 10))
        eval_each = mc_cfg.get("eval_each_cycle", True)
        eval_matches = self._config.evaluation.get("num_matches", 0)
        obs_mode = self._config.experiment.get("observation_mode", "full")
        tc = self._config.training

        s2_model = getattr(self, "_stage2a_model", None)
        if s2_model is None:
            input_dim = int(np.prod(encoder.metadata().output_shape))
            model_cfg = self._config.model
            s2_model = self._create_stage2a_model(encoder)

        # CQ-0240: policy anchor 設定
        pa_cfg = tc.get("policy_anchor", {})
        pa_enabled = pa_cfg.get("enabled", False)
        pa_reference = pa_cfg.get("reference", "imitation_fixed")
        pa_warmup = pa_cfg.get("warmup_cycles", 0)
        pa_interval = pa_cfg.get("update_interval_cycles", 1)

        # CQ-0236: rule_mix 設定
        rm_cfg = tc.get("rule_mix", {})
        rm_enabled = rm_cfg.get("enabled", False)
        rm_ratio = rm_cfg.get("policy_ratio", 1.0)
        rm_save_bl = rm_cfg.get("save_baseline_actions", False)
        rml_cfg = tc.get("rule_mix_learner", {})
        rml_enabled = rml_cfg.get("enabled", False)
        rml_bl_epochs = rml_cfg.get("baseline_imitation_epochs", 0)
        rml_ppo_epochs = rml_cfg.get("policy_ppo_epochs", tc.get("epochs", 4))
        rml_ppo_mode = rml_cfg.get("ppo_mode", "separated")

        cycles_data: list[dict] = []
        last_train_metrics: dict = {}
        last_sp_stats: dict = {}
        last_eval_metrics: dict = {}
        # CQ-0240: anchor path を cycle 間で保持
        _last_anchor_path: str | None = None

        for ci in range(num_cycles):
            cyc_label = f"cycle_{ci:02d}"
            cyc_dir = run_dir / cyc_label
            cyc_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  --- {cyc_label} ---")

            # 1. Selfplay (rule_mix → 一時上書き)
            sp_dir = cyc_dir / "selfplay"
            sp_seed = self._global_seed + ci * sp_matches
            self._stage2a_model = s2_model
            if rm_enabled:
                orig_pr = self._config.selfplay.get("policy_ratio", 1.0)
                orig_sb = self._config.selfplay.get("save_baseline_actions", False)
                self._config.selfplay["policy_ratio"] = rm_ratio
                self._config.selfplay["save_baseline_actions"] = rm_save_bl
            try:
                sp_stats = self._run_selfplay_stage2a(
                    run_dir, encoder, output_dir=sp_dir,
                    num_matches=sp_matches, base_seed=sp_seed)
            finally:
                if rm_enabled:
                    self._config.selfplay["policy_ratio"] = orig_pr
                    self._config.selfplay["save_baseline_actions"] = orig_sb
            last_sp_stats = sp_stats
            logger.info(f"    selfplay: steps={sp_stats['total_steps']} "
                         f"rounds={sp_stats.get('num_rounds', 0)}")

            # CQ-0240: anchor loading for PPO
            def _setup_anchor(learner_obj):
                nonlocal _last_anchor_path
                if not pa_enabled or ci < pa_warmup:
                    return
                ckpt_dir = run_dir / "checkpoints"
                if pa_reference == "imitation_fixed":
                    imi_ckpt = ckpt_dir / "checkpoint_imitation.pt"
                    if imi_ckpt.exists():
                        learner_obj.load_anchor(str(imi_ckpt))
                elif pa_reference == "lagged_policy":
                    effective_ci = ci - pa_warmup
                    should_update = (effective_ci % pa_interval == 0
                                      if pa_interval > 0 else True)
                    if effective_ci == 0 or _last_anchor_path is None:
                        # warmup 後最初: 必ず imitation checkpoint
                        imi = ckpt_dir / "checkpoint_imitation.pt"
                        if imi.exists():
                            learner_obj.load_anchor(str(imi))
                            _last_anchor_path = str(imi)
                    elif should_update:
                        # interval に従って lagged checkpoint に更新
                        prev = ckpt_dir / f"checkpoint_cycle_{ci-1:02d}.pt"
                        if prev.exists():
                            learner_obj.load_anchor(str(prev))
                            _last_anchor_path = str(prev)
                        elif _last_anchor_path is not None:
                            learner_obj.load_anchor(_last_anchor_path)
                    elif _last_anchor_path is not None:
                        learner_obj.load_anchor(_last_anchor_path)

            # 2. Learner (rule_mix_learner → baseline BC + policy PPO or mixed PPO)
            learner_stages: dict = {}
            if rml_enabled and rml_ppo_mode == "separated":
                # baseline imitation stage
                if rml_bl_epochs > 0:
                    bl_lc = self._as_dict()
                    bl_lc["training"]["algorithm"] = "imitation"
                    bl_learner = Stage2aLearner(
                        config=bl_lc, model=s2_model, run_dir=cyc_dir)
                    bl_tm = bl_learner.train(
                        sp_dir, num_epochs=rml_bl_epochs,
                        filter_actor_type="baseline")
                    learner_stages["baseline_imitation"] = bl_tm
                    logger.info(f"    baseline BC: {bl_tm.get('num_updates', 0)} updates")
                # policy PPO stage
                ppo_lc = self._as_dict()
                ppo_lc["training"]["algorithm"] = "ppo"
                ppo_learner = Stage2aLearner(
                    config=ppo_lc, model=s2_model, run_dir=cyc_dir)
                _setup_anchor(ppo_learner)
                train_metrics = ppo_learner.train(
                    sp_dir, num_epochs=rml_ppo_epochs,
                    filter_actor_type="policy")
                learner_stages["policy_ppo"] = train_metrics
            else:
                # mixed or simple PPO
                ppo_epochs = rml_ppo_epochs if rml_enabled else tc.get("epochs", 4)
                learner_config = self._as_dict()
                learner_config["training"]["algorithm"] = "ppo"
                learner = Stage2aLearner(
                    config=learner_config, model=s2_model, run_dir=cyc_dir)
                _setup_anchor(learner)
                train_metrics = learner.train(sp_dir, num_epochs=ppo_epochs)
                if rml_enabled:
                    learner_stages["mixed_ppo"] = train_metrics
                else:
                    learner_stages["policy_ppo"] = train_metrics
            last_train_metrics = train_metrics
            logger.info(f"    learner: loss={train_metrics.get('policy_loss', 0):.4f} "
                         f"updates={train_metrics.get('num_updates', 0)}")

            # 3. Checkpoint
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(s2_model.state_dict(),
                        ckpt_dir / f"checkpoint_{cyc_label}.pt")

            # 4. Eval (optional, single or parallel)
            eval_metrics: dict = {}
            if eval_each and eval_matches > 0:
                self._stage2a_model = s2_model
                eval_metrics = self._run_eval_stage2a(
                    run_dir, encoder, seed_offset=ci * 400)
                last_eval_metrics = eval_metrics
                logger.info(f"    eval: avg_rank={eval_metrics.get('avg_rank', 0):.2f}")

            cycles_data.append({
                "cycle_index": ci,
                "selfplay_stats": sp_stats,
                "learner_metrics": train_metrics,
                "learner_stages": learner_stages,
                "eval_metrics": eval_metrics if eval_metrics else None,
            })

        # Final checkpoint
        torch.save(s2_model.state_dict(),
                    ckpt_dir / "checkpoint_learner.pt")
        self._stage2a_model = s2_model

        # last cycle eval → top-level
        if last_eval_metrics:
            # Store for summary
            pass

        return last_train_metrics, cycles_data

    def _run_eval_stage2a(self, run_dir: Path, encoder,
                          seed_offset: int = 0,
                          num_matches_override: int | None = None,
                          num_workers_override: int | None = None,
                          seed_start_override: int | None = None,
                          eval_subdir: str = "eval",
                          ) -> dict:
        """CQ-0230/0235/0254: Stage2a deterministic evaluation"""
        eval_cfg = self._config.evaluation
        num_matches = (num_matches_override if num_matches_override is not None
                       else eval_cfg.get("num_matches", 100))
        if num_matches <= 0:
            return {"skipped": True}

        obs_mode = self._config.experiment.get("observation_mode", "full")
        eval_mode = eval_cfg.get("mode", eval_cfg.get("eval_mode", "single"))
        seed_start = ((seed_start_override if seed_start_override is not None
                        else eval_cfg.get("seed_start", 10000)) + seed_offset)
        num_workers = (num_workers_override if num_workers_override is not None
                       else eval_cfg.get("num_workers", 1))
        s2_model = getattr(self, "_stage2a_model", None)

        if num_workers > 1 and s2_model is not None:
            from mahjong_rl.stage2a_parallel import run_stage2a_eval_parallel
            # model を一時保存
            eval_dir = run_dir / eval_subdir
            eval_dir.mkdir(parents=True, exist_ok=True)
            model_path = str(eval_dir / "_eval_model.pt")
            sd = {k: v.cpu() for k, v in s2_model.state_dict().items()}
            torch.save(sd, model_path)
            metrics = run_stage2a_eval_parallel(
                num_workers=num_workers,
                num_matches=num_matches,
                seed_start=seed_start,
                obs_mode=obs_mode,
                encoder_config=dict(self._config.feature_encoder),
                model_state_path=model_path,
                model_config=dict(self._config.model),
                eval_mode=eval_mode,
                policy_seat=eval_cfg.get("policy_seat", 0),
                inference_device=eval_cfg.get("inference_device", "cpu"),
                num_threads=eval_cfg.get("worker_num_threads", 1),
            )
        else:
            from mahjong_rl.stage2a_evaluator import Stage2aEvaluator
            from mahjong_rl.models.stage2a_model import Stage2aModel
            if s2_model is None:
                input_dim = int(np.prod(encoder.metadata().output_shape))
                model_cfg = self._config.model
                s2_model = self._create_stage2a_model(encoder)
            evaluator = Stage2aEvaluator(
                model=s2_model, encoder=encoder,
                observation_mode=obs_mode,
                device=torch.device("cpu"),
            )

            if eval_mode == "rotation":
                metrics = evaluator.evaluate_rotation(
                    num_matches=num_matches,
                    seed_start=seed_start,
                )
            else:
                metrics = evaluator.evaluate(
                    num_matches=num_matches,
                    seed_start=seed_start,
                    policy_seat=eval_cfg.get("policy_seat", 0),
                )

        # eval 成果物を保存
        eval_dir = run_dir / eval_subdir
        eval_dir.mkdir(parents=True, exist_ok=True)
        import json as _json
        with open(eval_dir / "metrics.json", "w") as f:
            _json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"  Stage2a eval: avg_rank={metrics['avg_rank']:.2f}, "
                     f"win_rate={metrics['win_rate']:.3f}")
        return metrics

    def _run_learner_stage2a(self, run_dir: Path, encoder) -> dict:
        """CQ-0225/0226: Stage2a learner (PPO on selfplay shard)"""
        from mahjong_rl.models.stage2a_model import Stage2aModel
        from mahjong_rl.stage2a_learner import Stage2aLearner

        selfplay_dir = run_dir / "selfplay"
        s2_model = getattr(self, "_stage2a_model", None)
        if s2_model is None:
            input_dim = int(np.prod(encoder.metadata().output_shape))
            model_cfg = self._config.model
            s2_model = self._create_stage2a_model(encoder)

        learner_config = self._as_dict()
        learner_config["training"]["algorithm"] = "ppo"
        learner = Stage2aLearner(
            config=learner_config,
            model=s2_model,
            run_dir=run_dir,
        )
        # CQ-0240: single-cycle anchor
        pa = self._config.training.get("policy_anchor", {})
        if pa.get("enabled", False):
            ref = pa.get("reference", "imitation_fixed")
            ckpt_dir = run_dir / "checkpoints"
            imi_ckpt = ckpt_dir / "checkpoint_imitation.pt"
            if ref == "imitation_fixed" and imi_ckpt.exists():
                learner.load_anchor(str(imi_ckpt))
            elif ref == "lagged_policy" and imi_ckpt.exists():
                learner.load_anchor(str(imi_ckpt))
        metrics = learner.train(selfplay_dir)
        logger.info(f"  Stage2a PPO learner: {metrics}")

        # Checkpoint
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(s2_model.state_dict(), ckpt_dir / "checkpoint_learner.pt")

        self._stage2a_model = s2_model
        return metrics

    def _run_selfplay_parallel(
        self, run_dir: Path, model, num_workers: int,
    ) -> dict:
        """multi-process self-play を実行する

        各 worker に matches を分配し、worker_*/shard_*.parquet に保存後に統計を集約する。
        """
        sp_cfg = self._config.selfplay
        selfplay_dir = run_dir / "selfplay"
        selfplay_dir.mkdir(parents=True, exist_ok=True)
        num_matches = sp_cfg.get("num_matches", 10)
        num_threads = sp_cfg.get("worker_num_threads", 1)
        base_seed = self._global_seed or 0
        obs_mode = self._config.experiment.get("observation_mode", "full")

        # model を一時ファイルに保存
        model_path = selfplay_dir / "_selfplay_model.pt"
        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict_cpu, model_path)

        model_config = dict(self._config.model)
        encoder_config = dict(self._config.feature_encoder)
        config_dict = self._as_dict()

        matches_per_worker = self._distribute_matches(num_matches, num_workers)

        ctx = mp.get_context("spawn")
        error_queue = ctx.Queue()
        processes = []

        try:
            for i, wm in enumerate(matches_per_worker):
                if wm == 0:
                    continue
                worker_seed = derive_worker_seed(base_seed, i)
                match_seeds = [derive_match_seed(worker_seed, j) for j in range(wm)]
                worker_output_dir = selfplay_dir / f"worker_{i}"

                p = ctx.Process(
                    target=_selfplay_worker_fn,
                    args=(
                        i, str(model_path), config_dict, model_config,
                        encoder_config, obs_mode, wm, match_seeds,
                        str(worker_output_dir), num_threads, base_seed,
                        worker_seed, error_queue,
                    ),
                )
                p.start()
                processes.append(p)

            self._wait_and_check_workers(
                processes, error_queue=error_queue,
                worker_label="selfplay worker")

            # 統計集約
            aggregated = self._aggregate_selfplay_stats(selfplay_dir, num_workers)
            aggregated["num_workers"] = num_workers
            aggregated["seed_strategy"] = {
                "base_seed": base_seed,
                "method": "derive_worker_seed + derive_match_seed",
            }
            logger.info(
                f"  total_steps: {aggregated['total_steps']} "
                f"({num_workers} workers)")
            return aggregated
        finally:
            if model_path.exists():
                model_path.unlink()

    @staticmethod
    def _aggregate_selfplay_stats(selfplay_dir: Path, num_workers: int) -> dict:
        """各 worker の stats.json を読んで集約する"""
        total_steps = 0
        total_rounds = 0
        total_matches = 0
        worker_stats_list = []

        # CQ-0108: 局結果集計キー
        _round_stat_keys = [
            "num_rounds", "tsumo_count", "ron_count", "ryukyoku_count",
            "policy_wins", "policy_deal_ins", "policy_draws",
            "policy_win_by_tsumo", "policy_win_by_ron",
        ]
        round_totals = {k: 0 for k in _round_stat_keys}

        for i in range(num_workers):
            stats_path = selfplay_dir / f"worker_{i}" / "stats.json"
            if not stats_path.exists():
                continue
            with open(stats_path) as f:
                ws = json.load(f)
            total_steps += ws.get("total_steps", 0)
            total_rounds += ws.get("total_rounds", 0)
            total_matches += ws.get("num_matches", 0)
            for k in _round_stat_keys:
                round_totals[k] += ws.get(k, 0)
            worker_stats_list.append(ws)

        # CQ-0139, CQ-0142: reward_composition 集約（raw values からの正確な quantile 計算）
        reward_composition = None
        rc_list = [ws.get("reward_composition") for ws in worker_stats_list
                   if ws.get("reward_composition") is not None]
        if rc_list:
            reward_composition = {"shanten_delta_enabled": rc_list[0].get("shanten_delta_enabled", False)}
            # raw values を読み込んで結合
            merged_raw: dict[str, list] = {"point_delta": [], "shanten_delta": [], "total": []}
            for i in range(num_workers):
                raw_path = selfplay_dir / f"worker_{i}" / "reward_raw_values.npz"
                if raw_path.exists():
                    data = np.load(raw_path)
                    for comp in merged_raw:
                        if comp in data:
                            merged_raw[comp].append(data[comp])
                    data.close()

            for comp in ("point_delta", "shanten_delta", "total"):
                arrays = merged_raw.get(comp, [])
                if arrays:
                    arr = np.concatenate(arrays)
                    n = len(arr)
                    s = float(arr.sum())
                    mean = float(arr.mean())
                    std = float(arr.std())
                    nz = int(np.count_nonzero(arr))
                    p50 = float(np.percentile(arr, 50))
                    p90 = float(np.percentile(arr, 90))
                    p99 = float(np.percentile(arr, 99))
                else:
                    # raw values なし → per-worker stats からのフォールバック
                    n = sum(rc[comp]["count"] for rc in rc_list if comp in rc)
                    s = sum(rc[comp]["sum"] for rc in rc_list if comp in rc)
                    mean = s / n if n > 0 else 0.0
                    std = 0.0
                    nz = sum(rc[comp].get("nonzero_count", 0) for rc in rc_list if comp in rc)
                    p50 = 0.0
                    p90 = 0.0
                    p99 = 0.0
                reward_composition[comp] = {
                    "count": n,
                    "sum": s,
                    "mean": mean,
                    "std": std,
                    "nonzero_count": nz,
                    "p50": p50,
                    "p90": p90,
                    "p99": p99,
                }
            # CQ-0143: reward_shaping 設定（最初の worker から取得）
            reward_shaping = worker_stats_list[0].get("reward_shaping") if worker_stats_list else None

        result = {
            "num_matches": total_matches,
            "total_steps": total_steps,
            "total_rounds": total_rounds,
            "output_dir": str(selfplay_dir),
            "worker_stats": worker_stats_list,
        }
        result.update(round_totals)
        if reward_composition is not None:
            result["reward_composition"] = reward_composition
        if reward_composition is not None and reward_shaping is not None:
            result["reward_shaping"] = reward_shaping
        return result

    def _run_learner(self, run_dir: Path, shard_dir: Path, model,
                     profiler=None, checkpoint_tag: str = "final",
                     override_algorithm: str | None = None,
                     override_epochs: int | None = None,
                     filter_actor_type: str | None = None) -> dict:
        """learner フェーズ (CQ-0185: override パラメータ追加)"""
        training_device = resolve_device(
            self._config.training.get("device", "auto"))
        # override_algorithm が指定された場合、一時的に config を差し替え
        config_dict = self._as_dict()
        if override_algorithm is not None:
            config_dict = dict(config_dict)
            config_dict["training"] = dict(config_dict.get("training", {}))
            config_dict["training"]["algorithm"] = override_algorithm
        learner = Learner(
            config=config_dict,
            model=model,
            run_dir=run_dir,
            device=training_device,
        )
        train_metrics = learner.train(
            shard_dir, num_epochs=override_epochs,
            filter_actor_type=filter_actor_type, profiler=profiler)
        learner.save_checkpoint(tag=checkpoint_tag)
        logger.info(f"  policy_loss: {train_metrics['policy_loss']:.4f}")
        return train_metrics

    def _run_eval(self, run_dir: Path, model, encoder, obs_mode: str,
                  eval_dir_override: Path | None = None) -> dict:
        """eval フェーズ

        evaluation.mode で単席 / rotation を切り替え可能。
        evaluation.num_workers > 1 の場合は multi-process 実行。
        - "single" (デフォルト): 単席評価
        - "rotation": 全席ローテーション評価
        """
        eval_cfg = self._config.evaluation
        eval_dir = eval_dir_override or (run_dir / "eval")
        num_workers = eval_cfg.get("num_workers", 1)
        eval_mode = eval_cfg.get("mode", "single")
        num_matches = eval_cfg.get("num_matches", 10)
        seed_start = eval_cfg.get("seed_start", 0)

        if num_workers > 1:
            return self._run_eval_parallel(
                run_dir, model, eval_dir, eval_mode, num_matches,
                seed_start, num_workers, obs_mode)

        # 単一プロセス評価 (既存経路)
        eval_device = resolve_device(
            eval_cfg.get("inference_device", "auto"))
        # CQ-0153: value current_shanten 有効時は evaluator にも渡す
        vf_cfg = self._config.model.get("value_features", {})
        cs_enabled = vf_cfg.get("current_shanten", {}).get("enabled", False)
        eval_runner = EvaluationRunner(
            model=model,
            encoder=encoder,
            observation_mode=obs_mode,
            inference_device=eval_device,
            value_shanten_enabled=cs_enabled,
        )

        if eval_mode == "rotation":
            seats = eval_cfg.get("rotation_seats", [0, 1, 2, 3])
            rotation_result = eval_runner.evaluate_rotation(
                num_matches=num_matches,
                seed_start=seed_start,
                eval_dir=eval_dir,
                seats=seats,
            )
            agg = rotation_result.aggregate
            logger.info(f"  avg_rank (rotation): {agg.avg_rank:.2f}")
            return {
                "eval_mode": "rotation",
                "rotation_seats": seats,
                "avg_rank": agg.avg_rank,
                "avg_score": agg.avg_score,
                "win_rate": agg.win_rate,
                "deal_in_rate": agg.deal_in_rate,
            }
        else:
            policy_seats = eval_cfg.get("policy_seats", None)
            eval_metrics = eval_runner.evaluate(
                num_matches=num_matches,
                seed_start=seed_start,
                eval_dir=eval_dir,
                policy_seats=policy_seats,
            )
            logger.info(f"  avg_rank: {eval_metrics.avg_rank:.2f}")
            return {
                "eval_mode": "single",
                "avg_rank": eval_metrics.avg_rank,
                "avg_score": eval_metrics.avg_score,
                "win_rate": eval_metrics.win_rate,
                "deal_in_rate": eval_metrics.deal_in_rate,
            }

    def _run_eval_parallel(
        self,
        run_dir: Path,
        model,
        eval_dir: Path,
        eval_mode: str,
        num_matches: int,
        seed_start: int,
        num_workers: int,
        obs_mode: str,
    ) -> dict:
        """multi-process evaluation を実行する

        各 worker に matches を分配し、partial 結果を保存後に集約する。
        モデルはファイル経由で受け渡す (shared-memory 非依存)。
        """
        eval_cfg = self._config.evaluation
        partials_dir = eval_dir / "partials"
        partials_dir.mkdir(parents=True, exist_ok=True)
        num_threads = eval_cfg.get("worker_num_threads", 1)
        base_seed = self._global_seed or 0

        # state_dict をファイルに保存 (shared-memory 非依存)
        model_path = eval_dir / "_eval_model.pt"
        eval_dir.mkdir(parents=True, exist_ok=True)
        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict_cpu, model_path)

        model_config = dict(self._config.model)
        encoder_config = dict(self._config.feature_encoder)

        try:
            if eval_mode == "rotation":
                seats = eval_cfg.get("rotation_seats", [0, 1, 2, 3])
                return self._run_eval_parallel_rotation(
                    partials_dir, eval_dir, str(model_path), model_config,
                    encoder_config, obs_mode, num_matches,
                    num_workers, num_threads, base_seed, seats)
            else:
                policy_seats = eval_cfg.get("policy_seats", None) or [0]
                return self._run_eval_parallel_single(
                    partials_dir, eval_dir, str(model_path), model_config,
                    encoder_config, obs_mode, num_matches,
                    num_workers, num_threads, base_seed, policy_seats)
        finally:
            # 一時モデルファイルを削除
            if model_path.exists():
                model_path.unlink()

    def _run_eval_parallel_single(
        self,
        partials_dir: Path,
        eval_dir: Path,
        model_path: str,
        model_config: dict,
        encoder_config: dict,
        obs_mode: str,
        num_matches: int,
        num_workers: int,
        num_threads: int,
        base_seed: int,
        policy_seats: list[int],
    ) -> dict:
        """single モードの parallel eval"""
        # matches を worker に分配
        matches_per_worker = self._distribute_matches(num_matches, num_workers)

        ctx = mp.get_context("spawn")
        error_queue = ctx.Queue()
        processes = self._spawn_eval_workers(
            matches_per_worker, model_path, model_config, encoder_config,
            obs_mode, policy_seats, str(partials_dir), num_threads, base_seed,
            error_queue=error_queue,
            reward_config_dict=dict(self._config.reward))

        self._wait_and_check_workers(processes, error_queue=error_queue)

        # 集約
        metrics = aggregate_and_save(partials_dir, eval_dir)
        logger.info(f"  avg_rank (parallel, {num_workers} workers): {metrics.avg_rank:.2f}")
        return {
            "eval_mode": "single",
            "avg_rank": metrics.avg_rank,
            "avg_score": metrics.avg_score,
            "win_rate": metrics.win_rate,
            "deal_in_rate": metrics.deal_in_rate,
            "num_workers": num_workers,
        }

    def _run_eval_parallel_rotation(
        self,
        partials_dir: Path,
        eval_dir: Path,
        model_path: str,
        model_config: dict,
        encoder_config: dict,
        obs_mode: str,
        num_matches: int,
        num_workers: int,
        num_threads: int,
        base_seed: int,
        seats: list[int],
    ) -> dict:
        """rotation モードの parallel eval

        各席を各 worker に割り当てる。worker 数が席数より多い場合、
        席ごとの matches も分割する。
        """
        ctx = mp.get_context("spawn")
        error_queue = ctx.Queue()
        all_processes = []
        worker_id_offset = 0

        for seat in seats:
            if num_workers >= len(seats):
                workers_for_seat = max(1, num_workers // len(seats))
            else:
                workers_for_seat = 1
            matches_per_worker = self._distribute_matches(
                num_matches, workers_for_seat)

            processes = self._spawn_eval_workers(
                matches_per_worker, model_path, model_config,
                encoder_config, obs_mode, [seat],
                str(partials_dir), num_threads, base_seed,
                worker_id_offset=worker_id_offset,
                error_queue=error_queue,
                reward_config_dict=dict(self._config.reward))
            all_processes.extend(processes)
            worker_id_offset += len(matches_per_worker)

        self._wait_and_check_workers(all_processes, error_queue=error_queue)

        # 席別集約
        result = aggregate_rotation_partials(partials_dir, eval_dir, seats)
        agg = result.aggregate
        logger.info(f"  avg_rank (rotation parallel, {num_workers} workers): {agg.avg_rank:.2f}")
        return {
            "eval_mode": "rotation",
            "rotation_seats": seats,
            "avg_rank": agg.avg_rank,
            "avg_score": agg.avg_score,
            "win_rate": agg.win_rate,
            "deal_in_rate": agg.deal_in_rate,
            "num_workers": num_workers,
        }

    @staticmethod
    def _spawn_eval_workers(
        matches_per_worker: list[int],
        model_path: str,
        model_config: dict,
        encoder_config: dict,
        obs_mode: str,
        policy_seats: list[int],
        partials_dir: str,
        num_threads: int,
        base_seed: int,
        worker_id_offset: int = 0,
        error_queue: mp.Queue | None = None,
        reward_config_dict: dict | None = None,
    ) -> list[mp.Process]:
        """eval worker プロセスを生成・起動する

        Returns:
            起動済みの Process リスト
        """
        ctx = mp.get_context("spawn")
        processes = []
        for i, wm in enumerate(matches_per_worker):
            if wm == 0:
                continue
            wid = worker_id_offset + i
            p = ctx.Process(
                target=_eval_worker_fn,
                args=(
                    wid, model_path, model_config, encoder_config,
                    obs_mode, wm, policy_seats,
                    partials_dir, num_threads, base_seed,
                    error_queue, reward_config_dict,
                ),
            )
            p.start()
            processes.append(p)
        return processes

    @staticmethod
    def _wait_and_check_workers(
        processes: list[mp.Process],
        error_queue: mp.Queue | None = None,
        worker_label: str = "eval worker",
    ) -> None:
        """全 worker の完了を待ち、エラーを検知する

        error_queue が渡された場合、worker 側の例外詳細を取得してログに記録する。
        """
        for p in processes:
            p.join()

        # error_queue からエラー詳細を収集
        # (Queue.empty() は信頼性が低いため get_nowait + Empty 例外で終了)
        worker_errors: list[dict] = []
        if error_queue is not None:
            import queue
            while True:
                try:
                    worker_errors.append(error_queue.get_nowait())
                except queue.Empty:
                    break

        failed = [p for p in processes if p.exitcode != 0]
        if failed:
            exit_codes = [p.exitcode for p in failed]

            # worker 側エラー詳細をログ出力
            for err in worker_errors:
                logger.error(
                    f"{worker_label} {err['worker_id']} 例外: "
                    f"[{err['exception_type']}] {err['message']}")
                logger.error(f"{worker_label} {err['worker_id']} traceback:\n{err['traceback']}")

            # エラーメッセージ組み立て
            msg_parts = [f"{worker_label} {len(failed)}/{len(processes)} 件が失敗 (exit codes: {exit_codes})"]
            for err in worker_errors:
                msg_parts.append(
                    f"  worker {err['worker_id']}: [{err['exception_type']}] {err['message']}")
            msg = "\n".join(msg_parts)
            logger.error(msg)
            raise RuntimeError(msg)

    @staticmethod
    def _distribute_matches(num_matches: int, num_workers: int) -> list[int]:
        """matches を worker に均等分配する

        Returns:
            各 worker の match 数リスト
        """
        base = num_matches // num_workers
        remainder = num_matches % num_workers
        return [base + (1 if i < remainder else 0) for i in range(num_workers)]

    def _create_encoder(self):
        """設定からエンコーダを生成する (CQ-0119, CQ-0171, CQ-0217)"""
        enc_cfg = self._config.feature_encoder
        name = enc_cfg.get("name", "FlatFeatureEncoder")
        obs_mode = enc_cfg.get(
            "observation_mode",
            self._config.experiment.get("observation_mode", "full"),
        )
        if name == "ChannelTensorEncoder":
            return ChannelTensorEncoder(observation_mode=obs_mode)
        return FlatFeatureEncoder(
            observation_mode=obs_mode,
            shanten_hint=_parse_encoder_flag(enc_cfg, "shanten_hint"),
            discard_ukeire_hint=_parse_encoder_flag(enc_cfg, "discard_ukeire_hint"),
            current_shanten_input=_parse_encoder_flag(enc_cfg, "current_shanten"),
            shape_hint=_parse_encoder_flag(enc_cfg, "shape_hint"),
            turn_context=_parse_encoder_flag(enc_cfg, "turn_context"),
            opponent_current_shanten=_parse_encoder_flag(enc_cfg, "opponent_current_shanten"),
            opponent_tenpai_flag=_parse_encoder_flag(enc_cfg, "opponent_tenpai_flag"),
            danger_mask=_parse_encoder_flag(enc_cfg, "danger_mask"),
        )

    def _create_model(self, encoder):
        """設定からモデルを生成する (CQ-0157, CQ-0203)"""
        model_cfg = self._config.model
        hidden_dims = model_cfg.get("hidden_dims", [256, 128])
        value_heads = model_cfg.get("value_heads", ["round_delta"])

        # CQ-0151: value head 専用補助特徴の次元
        vf_cfg = model_cfg.get("value_features", {})
        cs_cfg = vf_cfg.get("current_shanten", {})
        value_aux_dim = 1 if cs_cfg.get("enabled", False) else 0

        # エンコーダの出力次元を取得
        meta = encoder.metadata()
        import math
        input_dim = math.prod(meta.output_shape)

        # CQ-0157: tower config
        pt_cfg = model_cfg.get("policy_tower", {})
        vt_cfg = model_cfg.get("value_tower", {})

        # CQ-0203, CQ-0204, CQ-0214, CQ-0217: policy_direct_hints 整合検証
        pdh_cfg = model_cfg.get("policy_direct_hints", {})
        if pdh_cfg.get("enabled", False):
            enc_cfg = self._config.feature_encoder
            obs_mode = enc_cfg.get(
                "observation_mode",
                self._config.experiment.get("observation_mode", "full"))
            is_partial = (obs_mode == "partial")
            fr = meta.feature_ranges or {}
            for src in pdh_cfg.get("sources", []):
                if src not in fr:
                    if src in _FULL_ONLY_SOURCES and is_partial:
                        continue  # Partial mode auto-off
                    # Full mode で fr にない = 対応する encoder feature が off
                    raise ValueError(
                        f"policy_direct_hints.sources に '{src}' があるが、"
                        f"encoder feature_ranges に見つかりません。"
                        f"対応する feature_encoder flag を有効にしてください")
        direct_hint_ranges = _resolve_direct_hint_ranges(
            pdh_cfg, meta.feature_ranges)

        return MLPPolicyValueModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            value_heads=value_heads,
            value_aux_dim=value_aux_dim,
            policy_tower_config=pt_cfg if pt_cfg.get("enabled", False) else None,
            value_tower_config=vt_cfg if vt_cfg.get("enabled", False) else None,
            policy_direct_hints_config=pdh_cfg if pdh_cfg.get("enabled", False) else None,
            direct_hint_ranges=direct_hint_ranges,
        )

    def _setup_file_logging(self, run_dir: Path) -> logging.FileHandler:
        """run.log 用の FileHandler を設定する"""
        handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        # mahjong_rl ロガーにアタッチ（ルートロガーのレベルに依存しない）
        ml_logger = logging.getLogger("mahjong_rl")
        ml_logger.addHandler(handler)
        if ml_logger.getEffectiveLevel() > logging.DEBUG:
            ml_logger.setLevel(logging.DEBUG)
        return handler

    def _teardown_file_logging(self, handler: logging.FileHandler) -> None:
        """FileHandler を除去する"""
        handler.flush()
        handler.close()
        logging.getLogger("mahjong_rl").removeHandler(handler)

    def _save_summary(self, run_dir: Path, result: dict,
                      phase_status: dict[str, str]) -> None:
        """summary.json を保存する"""
        # shard 数カウント (平坦 + worker サブディレクトリ)
        selfplay_dir = run_dir / "selfplay"
        if selfplay_dir.exists():
            flat = list(selfplay_dir.glob("shard_*.parquet"))
            nested = list(selfplay_dir.glob("worker_*/shard_*.parquet"))
            shard_count = len(set(flat) | set(nested))
        else:
            shard_count = 0

        # checkpoint 有無
        ckpt_dir = run_dir / "checkpoints"
        has_checkpoint = any(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else False

        # eval 有無
        has_eval = "eval_metrics" in result

        # フェーズ別統計 (CQ-0053)
        phase_stats = {}
        if "selfplay_stats" in result:
            sp = result["selfplay_stats"]
            phase_stats["selfplay"] = {
                "total_steps": sp.get("total_steps", 0),
                "total_matches": sp.get("total_matches",
                                         sp.get("num_matches", 0)),
                "shard_count": shard_count,
                "num_workers": sp.get("num_workers", 1),
                "seed_strategy": sp.get("seed_strategy"),
                # CQ-0106: 局結果集計
                "num_rounds": sp.get("num_rounds", 0),
                "tsumo_count": sp.get("tsumo_count", 0),
                "ron_count": sp.get("ron_count", 0),
                "ryukyoku_count": sp.get("ryukyoku_count", 0),
                "policy_wins": sp.get("policy_wins", 0),
                "policy_deal_ins": sp.get("policy_deal_ins", 0),
                "policy_draws": sp.get("policy_draws", 0),
                "policy_win_by_tsumo": sp.get("policy_win_by_tsumo", 0),
                "policy_win_by_ron": sp.get("policy_win_by_ron", 0),
                # Stage2a 追加
                "discard_count": sp.get("discard_count"),
                "call_count": sp.get("call_count"),
                "stage": sp.get("stage"),
            }
            # CQ-0139: reward composition
            rc = sp.get("reward_composition")
            if rc is not None:
                phase_stats["selfplay"]["reward_composition"] = rc
            # CQ-0143: reward shaping 設定
            rs = sp.get("reward_shaping")
            if rs is not None:
                phase_stats["selfplay"]["reward_shaping"] = rs
        if "imitation_metrics" in result:
            imi = result["imitation_metrics"]
            # imitation shard 数
            imi_dir = run_dir / "imitation"
            if imi_dir.exists():
                imi_flat = list(imi_dir.glob("shard_*.parquet"))
                imi_nested = list(imi_dir.glob("worker_*/shard_*.parquet"))
                imi_chunk_flat = list(imi_dir.glob("chunk_*/shard_*.parquet"))
                imi_chunk_nested = list(imi_dir.glob("chunk_*/worker_*/shard_*.parquet"))
                imi_shard_count = len(set(imi_flat) | set(imi_nested)
                                      | set(imi_chunk_flat) | set(imi_chunk_nested))
            else:
                imi_shard_count = 0
            # Stage2a: train_metrics がネストされている場合
            tm = imi.get("train_metrics", {})
            dg = imi.get("data_generation", {})
            imi_stats: dict = {
                "total_steps": dg.get("total_steps", imi.get("total_steps", 0)),
                "num_updates": tm.get("num_updates", imi.get("num_updates", 0)),
                "shard_count": imi_shard_count,
                "policy_loss": tm.get("policy_loss", imi.get("policy_loss")),
                "num_workers": dg.get("num_workers", 1),
                "seed_strategy": dg.get("seed_strategy"),
                # Stage2a 追加
                "discard_count": tm.get("discard_count", imi.get("discard_count")),
                "call_count": tm.get("call_count", imi.get("call_count")),
                "discard_loss": tm.get("discard_loss"),
                "call_loss": tm.get("call_loss"),
                "stage": imi.get("stage"),
                # 教師再現メトリクス (Stage1 / Stage2a 共通)
                # is not None ベースで値解決 (0.0 を欠損扱いしない)
                "teacher_top1_match_rate": _first_not_none(
                    tm.get("teacher_top1_match_rate_discard"),
                    imi.get("teacher_top1_match_rate")),
                "teacher_best_set_hit_rate": _first_not_none(
                    tm.get("teacher_best_set_hit_rate_discard"),
                    imi.get("teacher_best_set_hit_rate")),
                "teacher_best_set_status": imi.get("teacher_best_set_status"),
                "teacher_best_mask_shard_info": imi.get("teacher_best_mask_shard_info"),
                # Stage2a teacher diagnostics
                "teacher_top1_match_rate_discard": tm.get("teacher_top1_match_rate_discard"),
                "teacher_best_set_hit_rate_discard": tm.get("teacher_best_set_hit_rate_discard"),
                "teacher_top1_match_rate_optional": tm.get("teacher_top1_match_rate_optional"),
                # loss mode 追跡
                "imitation_loss_mode": tm.get("imitation_loss_mode", imi.get("imitation_loss_mode")),
                # value warm start 追跡
                "value_loss": tm.get("value_loss", imi.get("value_loss")),
                "imitation_value_warmstart": tm.get("imitation_value_warmstart",
                                                      imi.get("imitation_value_warmstart")),
            }
            # CQ-0209: imitation_optimizer 追跡
            imi_opt = self._config.training.get("imitation_optimizer", {})
            if imi_opt:
                imi_stats["imitation_optimizer"] = dict(imi_opt)
            # CQ-0206: multi-chunk imitation
            mci_info = imi.get("multi_chunk_imitation")
            if mci_info is not None:
                imi_stats["multi_chunk_imitation"] = mci_info
            chunks = imi.get("chunks")
            if chunks is not None:
                imi_stats["chunks"] = chunks
            # CQ-0254: imitation eval
            imi_eval = imi.get("imitation_eval")
            if imi_eval is not None:
                imi_stats["imitation_eval"] = imi_eval
            phase_stats["imitation"] = imi_stats
        if "train_metrics" in result:
            tm = result["train_metrics"]
            phase_stats["learner"] = {
                "total_steps": tm.get("total_steps", 0),
                "num_updates": tm.get("num_updates", 0),
                "policy_loss": tm.get("policy_loss"),
                "value_loss": tm.get("value_loss"),
                "mode": tm.get("mode"),
            }
            # CQ-0135: PPO 診断統計を phase_stats に転記
            ppo_diag = tm.get("ppo_diag")
            if ppo_diag is not None:
                phase_stats["learner"]["ppo_diag"] = ppo_diag
            # CQ-0166: learner 補助統計を summary に転送
            pre = tm.get("post_riichi_exclusion")
            if pre is not None:
                phase_stats["learner"]["post_riichi_exclusion"] = pre
            fs = tm.get("filter_stats")
            if fs is not None:
                phase_stats["learner"]["filter_stats"] = fs
        # CQ-0174: eval_before を phase_stats に保存
        if "eval_before" in result:
            eb = result["eval_before"]
            phase_stats["eval_before"] = {
                "eval_mode": eb.get("eval_mode"),
                "avg_rank": eb.get("avg_rank"),
                "avg_score": eb.get("avg_score"),
                "win_rate": eb.get("win_rate"),
                "deal_in_rate": eb.get("deal_in_rate"),
            }
        if "eval_metrics" in result:
            em = result["eval_metrics"]
            phase_stats["eval"] = {
                "eval_mode": em.get("eval_mode"),
                "avg_rank": em.get("avg_rank"),
                "avg_score": em.get("avg_score"),
                "win_rate": em.get("win_rate"),
                "deal_in_rate": em.get("deal_in_rate"),
            }

        # actor_type 内訳 (shard から集計, summary 報告用なので失敗時は空)
        try:
            actor_type_counts = self._count_actor_types(run_dir)
        except Exception:
            actor_type_counts = {}

        # device 情報
        resolved = result.get("resolved_devices", {})
        device_info = {
            "training": {
                "requested": self._config.training.get("device", "auto"),
                "resolved": resolved.get("training", "cpu"),
            },
            "selfplay": {
                "requested": self._config.selfplay.get("inference_device", "auto"),
                "resolved": resolved.get("selfplay", "cpu"),
            },
            "evaluation": {
                "requested": self._config.evaluation.get("inference_device", "auto"),
                "resolved": resolved.get("evaluation", "cpu"),
            },
        }

        # CQ-0179/CQ-0180: cycle 別メトリクスを phase_stats に追加
        cycles = result.get("cycles")
        if cycles:
            phase_stats["cycles"] = cycles

        # CQ-0115: phase_action を取り出し（内部キーなので pop）
        phase_action = result.pop("_phase_action", {})

        summary = {
            "global_seed": result.get("global_seed"),
            "phases": result.get("phases", []),
            "phase_status": phase_status,
            "success": "error" not in result,
            "error": result.get("error"),
            "shard_count": shard_count,
            "has_checkpoint": has_checkpoint,
            "has_eval": has_eval,
            "phase_stats": phase_stats,
            "phase_timing": result.get("phase_timing", {}),
            "total_duration_sec": result.get("total_duration_sec"),
            "actor_type_counts": actor_type_counts,
            "device_info": device_info,
            "env_info": self._collect_env_info(),
        }

        # CQ-0115: phase_action（今回の実行動作）を記録
        if phase_action:
            summary["phase_action"] = phase_action

        # CQ-0121, CQ-0171: encoder_features を記録
        enc_cfg = self._config.feature_encoder
        summary["encoder_features"] = {
            "name": enc_cfg.get("name", "FlatFeatureEncoder"),
            "observation_mode": enc_cfg.get(
                "observation_mode",
                self._config.experiment.get("observation_mode", "?")),
            "shanten_hint": _parse_encoder_flag(enc_cfg, "shanten_hint"),
            "discard_ukeire_hint": _parse_encoder_flag(enc_cfg, "discard_ukeire_hint"),
            "current_shanten": _parse_encoder_flag(enc_cfg, "current_shanten"),
            "shape_hint": _parse_encoder_flag(enc_cfg, "shape_hint"),
            "turn_context": _parse_encoder_flag(enc_cfg, "turn_context"),
            "opponent_current_shanten": _parse_encoder_flag(enc_cfg, "opponent_current_shanten"),
            "opponent_tenpai_flag": _parse_encoder_flag(enc_cfg, "opponent_tenpai_flag"),
            "danger_mask": _parse_encoder_flag(enc_cfg, "danger_mask"),
            "input_dim": result.get("input_dim"),
        }

        # CQ-0151, CQ-0152, CQ-0157, CQ-0203: model_features を記録
        model_cfg = self._config.model
        vf_cfg = model_cfg.get("value_features", {})
        cs_cfg = vf_cfg.get("current_shanten", {})
        pt_cfg = model_cfg.get("policy_tower", {})
        vt_cfg = model_cfg.get("value_tower", {})
        pdh_cfg = model_cfg.get("policy_direct_hints", {})
        summary["model_features"] = {
            "value_features": {
                "current_shanten": {
                    "enabled": cs_cfg.get("enabled", False),
                },
            },
            "policy_tower": {
                "enabled": pt_cfg.get("enabled", False),
                "hidden_dim": pt_cfg.get("hidden_dim"),
            },
            "value_tower": {
                "enabled": vt_cfg.get("enabled", False),
                "hidden_dim": vt_cfg.get("hidden_dim"),
            },
            # CQ-0203: policy_direct_hints
            "policy_direct_hints": {
                "enabled": pdh_cfg.get("enabled", False),
                "sources": pdh_cfg.get("sources", []),
                "local_hidden_dim": pdh_cfg.get("local_hidden_dim"),
                "tile_embedding_dim": pdh_cfg.get("tile_embedding_dim"),
                "context_gate_enabled": pdh_cfg.get(
                    "context_gate", {}).get("enabled", False),
            },
        }

        # プロファイル情報 (CQ-0098)
        profiling = result.get("profiling")
        if profiling is not None:
            summary["profiling"] = profiling

        # CQ-0110: 再利用情報
        reuse_info = result.get("reuse_info")
        if reuse_info is not None:
            summary["reuse_info"] = reuse_info

        # CQ-0114: ロード元 checkpoint パス
        loaded_checkpoint = result.get("loaded_checkpoint")
        if loaded_checkpoint is not None:
            summary["loaded_checkpoint"] = loaded_checkpoint

        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _save_manifest(self, run_dir: Path, result: dict,
                       phase_status: dict[str, str]) -> None:
        """artifacts_manifest.json を保存する (CQ-0109)

        phase 完了状態・成果物パス・config fingerprint・再利用メタデータを記録し、
        再利用判定に必要な情報を機械可読に提供する。
        """
        # phase_completion
        phase_completion = dict(phase_status)

        # eval_before は phase_status に含まれない場合がある
        if "eval_before" in result:
            if "eval_before" not in phase_completion:
                phase_completion["eval_before"] = "success"

        # artifacts 検出
        artifacts: dict[str, dict] = {}

        # imitation checkpoint
        imi_ckpt = run_dir / "checkpoints" / "checkpoint_imitation.pt"
        artifacts["imitation_checkpoint"] = {
            "exists": imi_ckpt.exists(),
            "path": "checkpoints/checkpoint_imitation.pt",
        }

        # imitation shard (CQ-0207: chunk 配下も集計)
        imi_dir = run_dir / "imitation"
        if imi_dir.exists():
            imi_flat = list(imi_dir.glob("shard_*.parquet"))
            imi_nested = list(imi_dir.glob("worker_*/shard_*.parquet"))
            imi_chunk_flat = list(imi_dir.glob("chunk_*/shard_*.parquet"))
            imi_chunk_nested = list(imi_dir.glob("chunk_*/worker_*/shard_*.parquet"))
            imi_shard_count = len(set(imi_flat) | set(imi_nested)
                                  | set(imi_chunk_flat) | set(imi_chunk_nested))
        else:
            imi_shard_count = 0
        artifacts["imitation_shards"] = {
            "exists": imi_shard_count > 0,
            "path": "imitation",
            "shard_count": imi_shard_count,
        }

        # selfplay shard
        sp_dir = run_dir / "selfplay"
        if sp_dir.exists():
            sp_flat = list(sp_dir.glob("shard_*.parquet"))
            sp_nested = list(sp_dir.glob("worker_*/shard_*.parquet"))
            sp_shard_count = len(set(sp_flat) | set(sp_nested))
        else:
            sp_shard_count = 0
        artifacts["selfplay_shards"] = {
            "exists": sp_shard_count > 0,
            "path": "selfplay",
            "shard_count": sp_shard_count,
        }

        # eval_before
        eval_before_dir = run_dir / "eval_before"
        eb_result = result.get("eval_before", {})
        artifacts["eval_before"] = {
            "exists": eval_before_dir.exists() and any(eval_before_dir.iterdir()) if eval_before_dir.exists() else False,
            "path": "eval_before",
            "avg_rank": eb_result.get("avg_rank"),
            "avg_score": eb_result.get("avg_score"),
            "win_rate": eb_result.get("win_rate"),
            "deal_in_rate": eb_result.get("deal_in_rate"),
        }

        # learner checkpoint (CQ-0228: Stage2a は checkpoint_learner.pt)
        stage = self._config.experiment.get("stage", "stage1")
        if stage == "stage2a":
            ckpt_name = "checkpoint_learner.pt"
        else:
            ckpt_name = "checkpoint_final.pt"
        learner_ckpt = run_dir / "checkpoints" / ckpt_name
        artifacts["learner_checkpoint"] = {
            "exists": learner_ckpt.exists(),
            "path": f"checkpoints/{ckpt_name}",
            "stage": stage,
        }

        # eval
        eval_dir = run_dir / "eval"
        artifacts["eval"] = {
            "exists": eval_dir.exists() and any(eval_dir.iterdir()) if eval_dir.exists() else False,
            "path": "eval",
        }

        # CQ-0254: imitation eval
        imi_eval_dir = run_dir / "imitation_eval"
        artifacts["imitation_eval"] = {
            "exists": (imi_eval_dir.exists()
                       and any(imi_eval_dir.iterdir()) if imi_eval_dir.exists() else False),
            "path": "imitation_eval",
        }

        # config fingerprint
        config_fingerprint = self._compute_config_fingerprint(run_dir)

        # reuse_metadata
        sp_cfg = self._config.selfplay
        eval_cfg = self._config.evaluation
        reuse_metadata = {
            "global_seed": result.get("global_seed"),
            "num_workers": sp_cfg.get("num_workers", 1),
            "policy_ratio": sp_cfg.get("policy_ratio", 0.5),
            "save_baseline_actions": sp_cfg.get("save_baseline_actions", False),
            "selfplay_num_matches": sp_cfg.get("num_matches", 10),
            "eval_mode": eval_cfg.get("mode", "fixed"),
            "eval_rotation_seats": eval_cfg.get("rotation_seats"),
            "eval_num_matches": eval_cfg.get("num_matches", 10),
            "imitation_matches": sp_cfg.get("imitation_matches",
                                            sp_cfg.get("num_matches", 10)),
        }

        manifest = {
            "manifest_version": 1,
            "phase_completion": phase_completion,
            "artifacts": artifacts,
            "config_fingerprint": config_fingerprint,
            "reuse_metadata": reuse_metadata,
        }

        # CQ-0114: ロード元 checkpoint パス
        loaded_checkpoint = result.get("loaded_checkpoint")
        if loaded_checkpoint is not None:
            manifest["loaded_checkpoint"] = loaded_checkpoint

        with open(run_dir / "artifacts_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _compute_config_fingerprint(run_dir: Path) -> str:
        """config.yaml の SHA-256 ハッシュを算出する (CQ-0109)"""
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            return hashlib.sha256(config_path.read_bytes()).hexdigest()
        return ""

    @staticmethod
    def _load_manifest(run_dir: Path) -> dict | None:
        """artifacts_manifest.json を読み込む (CQ-0111)"""
        path = run_dir / "artifacts_manifest.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _get_completed_phases(manifest: dict) -> set[str]:
        """manifest から完了済み phase の set を返す (CQ-0111)"""
        pc = manifest.get("phase_completion", {})
        completed = set()
        for phase, status in pc.items():
            if status in ("success", "skipped", "reused"):
                completed.add(phase)
        return completed

    @staticmethod
    def _validate_artifacts(run_dir: Path, manifest: dict,
                            phases: set[str]) -> None:
        """完了 phase の成果物が実際に存在するか検証する (CQ-0111, CQ-0113)

        CQ-0113: eval_before / selfplay の検証を厳密化。
        - selfplay: ディレクトリ存在 + shard ファイル実在まで確認
        - eval_before: ディレクトリまたは復元可能な指標ファイル存在を必須化

        Raises:
            ValueError: 成果物が不足している場合
        """
        artifacts = manifest.get("artifacts", {})
        missing = []

        if "imitation" in phases:
            # imitation checkpoint または shard が必要
            imi_ckpt = artifacts.get("imitation_checkpoint", {})
            imi_shards = artifacts.get("imitation_shards", {})
            ckpt_path = run_dir / imi_ckpt.get("path", "checkpoints/checkpoint_imitation.pt")
            shard_path = run_dir / imi_shards.get("path", "imitation")
            if not ckpt_path.exists() and not shard_path.exists():
                missing.append("imitation: checkpoint も shard も見つかりません")

        if "selfplay" in phases:
            sp_shards = artifacts.get("selfplay_shards", {})
            sp_path = run_dir / sp_shards.get("path", "selfplay")
            if not sp_path.exists():
                missing.append(f"selfplay: ディレクトリ {sp_path} が見つかりません")
            else:
                # CQ-0113: shard ファイルが実在するか確認
                flat = list(sp_path.glob("shard_*.parquet"))
                nested = list(sp_path.glob("worker_*/shard_*.parquet"))
                if not flat and not nested:
                    missing.append(
                        f"selfplay: {sp_path} に shard ファイルがありません")

        if "eval_before" in phases:
            # CQ-0113: eval_before はディレクトリまたは復元可能な指標が必要
            eb = artifacts.get("eval_before", {})
            eb_path = run_dir / eb.get("path", "eval_before")
            eb_has_dir = eb_path.exists() and eb_path.is_dir() and any(eb_path.iterdir())
            # CQ-0117: rotation は eval_rotation.json, single は eval_metrics.json
            eb_has_results = False
            if eb_path.exists():
                eb_has_results = (eb_path / "eval_rotation.json").exists() or \
                    (eb_path / "eval_metrics.json").exists()
            # summary から復元可能かもチェック
            summary_path = run_dir / "summary.json"
            eb_has_summary = False
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                    eb_has_summary = "eval_before" in summary.get("phase_stats", {}) or \
                        eb.get("avg_rank") is not None
                except (json.JSONDecodeError, OSError):
                    pass
            if not eb_has_dir and not eb_has_results and not eb_has_summary:
                missing.append(
                    "eval_before: ディレクトリ・結果ファイル・復元可能な指標のいずれも見つかりません")

        if missing:
            raise ValueError(
                "成果物整合エラー:\n" + "\n".join(f"  - {m}" for m in missing))

    def _restore_phase_result(self, run_dir: Path, phase: str,
                              result: dict) -> None:
        """スキップされた phase の結果を復元する (CQ-0111, CQ-0117)

        後続 phase で必要な値（eval_before の avg_rank など）を result に設定する。
        eval_before はファイルベース復元を優先する（summary.json 不在でも動作）。
        """
        # eval_before はファイルベース復元を優先 (CQ-0117)
        if phase == "eval_before" and "eval_before" not in result:
            # 優先順: eval_rotation.json → eval_metrics.json → manifest fallback
            eb_dir = run_dir / "eval_before"
            restored = False

            # 1. eval_rotation.json (rotation モード出力)
            eb_rotation_path = eb_dir / "eval_rotation.json"
            if eb_rotation_path.exists():
                try:
                    with open(eb_rotation_path) as f:
                        data = json.load(f)
                    data.setdefault("eval_mode", "rotation")
                    result["eval_before"] = data
                    restored = True
                except (json.JSONDecodeError, OSError):
                    pass

            # 2. eval_metrics.json (single モード出力)
            if not restored:
                eb_metrics_path = eb_dir / "eval_metrics.json"
                if eb_metrics_path.exists():
                    try:
                        with open(eb_metrics_path) as f:
                            data = json.load(f)
                        data.setdefault("eval_mode", "single")
                        result["eval_before"] = data
                        restored = True
                    except (json.JSONDecodeError, OSError):
                        pass

            # 3. manifest fallback (主要4指標)
            if not restored:
                manifest = self._load_manifest(run_dir)
                if manifest:
                    eb = manifest.get("artifacts", {}).get("eval_before", {})
                    fb: dict = {}
                    for key in ("avg_rank", "avg_score", "win_rate",
                                "deal_in_rate"):
                        val = eb.get(key)
                        if val is not None:
                            fb[key] = val
                    if fb:
                        result["eval_before"] = fb
            return

        # その他の phase は summary.json から復元
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            return
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        ps = summary.get("phase_stats", {})

        if phase == "imitation" and "imitation_metrics" not in result:
            imi_stats = ps.get("imitation", {})
            if imi_stats:
                result["imitation_metrics"] = imi_stats

        elif phase == "selfplay" and "selfplay_stats" not in result:
            sp_stats = ps.get("selfplay", {})
            if sp_stats:
                result["selfplay_stats"] = sp_stats

        elif phase == "learner" and "train_metrics" not in result:
            tm = ps.get("learner", {})
            if tm:
                result["train_metrics"] = tm

    def _copy_reused_artifacts(
        self, run_dir: Path, ref_dir: Path, reuse_phases: set[str],
        ref_manifest: dict, result: dict, phase_status: dict[str, str],
    ) -> None:
        """参照元 run_dir から成果物をコピーする (CQ-0110, CQ-0113)

        CQ-0113: コピー成功した phase のみ reused を設定する。
        """
        import shutil

        if "imitation" in reuse_phases:
            copied = False
            # imitation checkpoint コピー
            src_ckpt = ref_dir / "checkpoints" / "checkpoint_imitation.pt"
            if src_ckpt.exists():
                dst_ckpt = run_dir / "checkpoints" / "checkpoint_imitation.pt"
                dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_ckpt, dst_ckpt)
                copied = True
            # imitation shard コピー
            src_imi = ref_dir / "imitation"
            if src_imi.exists():
                dst_imi = run_dir / "imitation"
                shutil.copytree(src_imi, dst_imi, dirs_exist_ok=True)
                copied = True
            if copied:
                phase_status["imitation"] = "reused"

        if "selfplay" in reuse_phases:
            copied = False
            src_sp = ref_dir / "selfplay"
            if src_sp.exists():
                dst_sp = run_dir / "selfplay"
                shutil.copytree(src_sp, dst_sp, dirs_exist_ok=True)
                copied = True
            if copied:
                phase_status["selfplay"] = "reused"

        if "eval_before" in reuse_phases:
            copied = False
            src_eb = ref_dir / "eval_before"
            if src_eb.exists():
                dst_eb = run_dir / "eval_before"
                shutil.copytree(src_eb, dst_eb, dirs_exist_ok=True)
                copied = True
            if copied:
                phase_status["eval_before"] = "reused"

    def _count_actor_types(self, run_dir: Path) -> dict[str, int]:
        """shard ファイルから actor_type ごとの件数を集計する (CQ-0190)

        shard 読み取り失敗時は RuntimeError を送出する。
        shard が存在しないディレクトリはスキップする。
        """
        from mahjong_rl.shard import ShardReader
        counts: dict[str, int] = {}
        for subdir_name in ["selfplay", "imitation"]:
            subdir = run_dir / subdir_name
            if not subdir.exists():
                continue
            try:
                reader = ShardReader(subdir)
                if not reader._find_shards():
                    continue
                tensors = reader.read_as_tensors()
                for at in tensors.get("actor_types", []):
                    counts[at] = counts.get(at, 0) + 1
            except Exception as e:
                raise RuntimeError(
                    f"actor_type 集計に失敗しました ({subdir}): {e}") from e
        return counts

    def _append_notes(self, run_dir: Path, result: dict,
                      phase_status: dict[str, str]) -> None:
        """notes.md に実行結果の概要を追記する"""
        notes_path = run_dir / "notes.md"
        lines = [
            "",
            "## 実行結果",
            f"- 状態: {'成功' if 'error' not in result else '失敗'}",
            f"- global_seed: {result.get('global_seed')}",
            f"- phases: {result.get('phases', [])}",
        ]
        # フェーズ別ステータス
        for phase, status in phase_status.items():
            lines.append(f"  - {phase}: {status}")

        # CQ-0121, CQ-0171: encoder 情報
        enc_cfg = self._config.feature_encoder
        _flags = {
            "shanten_hint": _parse_encoder_flag(enc_cfg, "shanten_hint"),
            "discard_ukeire_hint": _parse_encoder_flag(enc_cfg, "discard_ukeire_hint"),
            "current_shanten": _parse_encoder_flag(enc_cfg, "current_shanten"),
            "shape_hint": _parse_encoder_flag(enc_cfg, "shape_hint"),
            "turn_context": _parse_encoder_flag(enc_cfg, "turn_context"),
            "opponent_current_shanten": _parse_encoder_flag(enc_cfg, "opponent_current_shanten"),
            "opponent_tenpai_flag": _parse_encoder_flag(enc_cfg, "opponent_tenpai_flag"),
            "danger_mask": _parse_encoder_flag(enc_cfg, "danger_mask"),
        }
        _flag_str = ", ".join(f"{k}={'on' if v else 'off'}" for k, v in _flags.items())
        lines.append(f"- encoder: {enc_cfg.get('name', '?')} "
                     f"({_flag_str}, "
                     f"input_dim={result.get('input_dim', '?')})")

        # デバイス情報
        resolved = result.get("resolved_devices", {})
        if resolved:
            lines.append(f"- devices: training={resolved.get('training', '?')}, "
                         f"selfplay={resolved.get('selfplay', '?')}, "
                         f"eval={resolved.get('evaluation', '?')}")

        # Python 実行環境情報
        lines.append(f"- python: {sys.version.split()[0]} ({sys.executable})")

        # imitation 並列情報
        imi = result.get("imitation_metrics", {})
        dg = imi.get("data_generation", {})
        if dg.get("num_workers", 1) > 1:
            lines.append(f"- imitation: num_workers={dg['num_workers']}, "
                         f"seed_strategy={dg.get('seed_strategy', {}).get('method', '?')}")

        # imitation 教師再現メトリクス (CQ-0127, CQ-0128)
        imi_top1 = imi.get("teacher_top1_match_rate")
        imi_best_set = imi.get("teacher_best_set_hit_rate")
        if imi_top1 is not None:
            line = f"- imitation teacher_top1_match_rate: {imi_top1:.4f}"
            if imi_best_set is not None:
                line += f", teacher_best_set_hit_rate: {imi_best_set:.4f}"
            lines.append(line)
        # CQ-0128: teacher_best_set_status
        tbm_status = imi.get("teacher_best_set_status")
        if tbm_status is not None:
            tbm_info = imi.get("teacher_best_mask_shard_info", {})
            line = f"- imitation teacher_best_set_status: {tbm_status}"
            if tbm_info:
                line += f" (shards: {tbm_info.get('available', 0)}/{tbm_info.get('total', 0)})"
            lines.append(line)
        # CQ-0132: loss mode
        imi_loss_mode = imi.get("imitation_loss_mode")
        if imi_loss_mode is not None:
            lines.append(f"- imitation loss_mode: {imi_loss_mode}")

        # CQ-0139: reward shaping 情報
        sp_stats = result.get("selfplay_stats", {})
        rc = sp_stats.get("reward_composition")
        if rc and rc.get("shanten_delta_enabled"):
            sd = rc.get("shanten_delta", {})
            lines.append(f"- reward_shaping: shanten_delta enabled, "
                         f"mean={sd.get('mean', 0):.6f}, "
                         f"nonzero={sd.get('nonzero_count', 0)}/{sd.get('count', 0)}")

        # CQ-0135: PPO 診断統計サマリ
        tm = result.get("train_metrics", {})
        ppo_diag = tm.get("ppo_diag")
        if ppo_diag:
            lines.append(f"- ppo_diag: advantage_mean={ppo_diag.get('advantage_mean', '?'):.4f}, "
                         f"clip_fraction={ppo_diag.get('clip_fraction', '?'):.4f}, "
                         f"ratio_mean={ppo_diag.get('ratio_mean', '?'):.4f}")

        # phase duration
        pt = result.get("phase_timing", {})
        if pt:
            lines.append("- phase duration:")
            for pname, pinfo in pt.items():
                dur = pinfo.get("duration_sec")
                if dur is not None:
                    lines.append(f"  - {pname}: {dur:.1f}s")
                else:
                    lines.append(f"  - {pname}: (未完了)")
        td = result.get("total_duration_sec")
        if td is not None:
            lines.append(f"- total_duration: {td:.1f}s")

        # エラー情報
        if "error" in result:
            lines.append(f"- エラー: {result['error']}")

        # 主要指標
        if "eval_metrics" in result:
            em = result["eval_metrics"]
            lines.append("")
            lines.append("## 主要指標")
            lines.append(f"- avg_rank: {em.get('avg_rank', '?'):.2f}"
                         if isinstance(em.get('avg_rank'), (int, float))
                         else f"- avg_rank: {em.get('avg_rank', '?')}")
            lines.append(f"- avg_score: {em.get('avg_score', '?'):.1f}"
                         if isinstance(em.get('avg_score'), (int, float))
                         else f"- avg_score: {em.get('avg_score', '?')}")
            lines.append(f"- win_rate: {em.get('win_rate', '?'):.3f}"
                         if isinstance(em.get('win_rate'), (int, float))
                         else f"- win_rate: {em.get('win_rate', '?')}")
            lines.append(f"- deal_in_rate: {em.get('deal_in_rate', '?'):.3f}"
                         if isinstance(em.get('deal_in_rate'), (int, float))
                         else f"- deal_in_rate: {em.get('deal_in_rate', '?')}")

        # checkpoint
        ckpt_dir = run_dir / "checkpoints"
        has_ckpt = any(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else False
        lines.append(f"- checkpoint: {'あり' if has_ckpt else 'なし'}")
        lines.append("")

        with open(notes_path, "a") as f:
            f.write("\n".join(lines))

    def _setup_global_seed(self) -> int:
        """global seed を設定する

        experiment.global_seed が指定されていればそれを使い、
        未指定なら乱数で生成する。
        """
        seed = self._config.experiment.get("global_seed", None)
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info(f"  global_seed: {seed}")
        return seed

    def _resolve_all_devices(self) -> dict:
        """全フェーズのデバイスを解決し、解決結果を返す"""
        cfg = self._config
        training_dev = resolve_device(cfg.training.get("device", "auto"))
        sp_dev = resolve_device(cfg.selfplay.get("inference_device", "auto"))
        eval_dev = resolve_device(cfg.evaluation.get("inference_device", "auto"))
        return {
            "training": str(training_dev),
            "selfplay": str(sp_dev),
            "evaluation": str(eval_dev),
        }

    @staticmethod
    def _collect_env_info() -> dict:
        """PyTorch/CUDA/Python 環境情報を収集する"""
        info: dict = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "venv": os.environ.get("VIRTUAL_ENV"),
        }
        if torch.cuda.is_available():
            try:
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
                info["cuda_device_count"] = torch.cuda.device_count()
            except Exception:
                pass
        return info

    def _as_dict(self) -> dict:
        """ExperimentConfig を dict として返す"""
        return {
            "experiment": self._config.experiment,
            "feature_encoder": self._config.feature_encoder,
            "model": self._config.model,
            "reward": self._config.reward,
            "selfplay": self._config.selfplay,
            "training": self._config.training,
            "evaluation": self._config.evaluation,
            "imitation": self._config.imitation,
            "export": self._config.export,
            "profiling": self._config.profiling,
        }
