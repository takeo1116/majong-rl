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

        # モデル・エンコーダ再構築 (ファイルから state_dict を読み込み)
        enc_name = encoder_config.get("name", "FlatFeatureEncoder")
        enc_obs = encoder_config.get("observation_mode", obs_mode)
        if enc_name == "ChannelTensorEncoder":
            encoder = ChannelTensorEncoder(observation_mode=enc_obs)
        else:
            encoder = FlatFeatureEncoder(observation_mode=enc_obs)

        import math
        meta = encoder.metadata()
        input_dim = math.prod(meta.output_shape)
        model = MLPPolicyValueModel(
            input_dim=input_dim,
            hidden_dims=model_config.get("hidden_dims", [256, 128]),
            value_heads=model_config.get("value_heads", ["round_delta"]),
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        eval_runner = EvaluationRunner(
            model=model, encoder=encoder, observation_mode=obs_mode)

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

        # エンコーダ再構築
        enc_name = encoder_config.get("name", "FlatFeatureEncoder")
        enc_obs = encoder_config.get("observation_mode", obs_mode)
        if enc_name == "ChannelTensorEncoder":
            encoder = ChannelTensorEncoder(observation_mode=enc_obs)
        else:
            encoder = FlatFeatureEncoder(observation_mode=enc_obs)

        # モデル再構築
        import math
        meta = encoder.metadata()
        input_dim = math.prod(meta.output_shape)
        model = MLPPolicyValueModel(
            input_dim=input_dim,
            hidden_dims=model_config.get("hidden_dims", [256, 128]),
            value_heads=model_config.get("value_heads", ["round_delta"]),
        )
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
        sp_stats = worker.run(
            num_matches=num_matches,
            match_seeds=match_seeds,
        )

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
                # 学習前評価 (eval も phases に含まれる場合のみ)
                if "eval" in phases:
                    if "eval_before" in completed_phases:
                        # CQ-0111: eval_before スキップ
                        logger.info(f"{label} eval_before はスキップ（完了済み）")
                        # CQ-0115: phase_action に記録
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
                            logger.info(f"  eval_before avg_rank: {result['eval_before'].get('avg_rank', '?')}")
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
        """imitation warm start フェーズ

        imitation.num_workers > 1 の場合は multi-process で教師データを生成する。
        """
        sp_cfg = self._config.selfplay
        imitation_dir = run_dir / "imitation"
        num_workers = self._config.imitation.get("num_workers", 1)

        # baseline 教師データ生成
        imi_config = dict(self._as_dict())
        imi_sp = dict(sp_cfg)
        imi_sp["save_baseline_actions"] = True
        imi_sp["policy_ratio"] = 0.0  # 全席 baseline
        imi_config["selfplay"] = imi_sp

        imi_matches = sp_cfg.get("imitation_matches",
                                 sp_cfg.get("num_matches", 10))

        if num_workers > 1:
            sp_stats = self._run_imitation_parallel(
                imitation_dir, model, imi_config, imi_matches, num_workers)
        else:
            sp_device = resolve_device(
                sp_cfg.get("inference_device", "auto"))
            worker = SelfPlayWorker(
                config=imi_config,
                model=model,
                encoder=encoder,
                output_dir=imitation_dir,
                inference_device=sp_device,
                profiler=profiler,
            )
            sp_stats = worker.run(
                num_matches=imi_matches,
                seed_start=sp_cfg.get("imitation_seed_start",
                                      sp_cfg.get("seed_start", 0)),
            )

        logger.info(f"  imitation data: {sp_stats['total_steps']} steps")

        # imitation 学習
        imi_train_config = dict(self._as_dict())
        imi_train_config["training"] = dict(imi_train_config["training"])
        imi_train_config["training"]["algorithm"] = "imitation"

        training_device = resolve_device(
            self._config.training.get("device", "auto"))
        learner = Learner(
            config=imi_train_config,
            model=model,
            run_dir=run_dir,
            device=training_device,
        )
        imi_epochs = self._config.training.get("imitation_epochs",
                                                self._config.training.get("epochs", 4))
        imi_filter = self._config.training.get("imitation_filter", None)
        metrics = learner.train(
            imitation_dir,
            num_epochs=imi_epochs,
            filter_actor_type="baseline",
            imitation_filter=imi_filter,
            profiler=profiler,
        )
        learner.save_checkpoint(tag="imitation")
        logger.info(f"  imitation loss: {metrics['policy_loss']:.4f}")
        # データ生成統計を学習 metrics に付加
        metrics["data_generation"] = {
            "total_steps": sp_stats.get("total_steps", 0),
            "num_matches": sp_stats.get("num_matches", 0),
            "num_workers": sp_stats.get("num_workers", 1),
            "seed_strategy": sp_stats.get("seed_strategy"),
        }
        return metrics

    def _run_imitation_parallel(
        self, imitation_dir: Path, model, imi_config: dict,
        num_matches: int, num_workers: int,
    ) -> dict:
        """multi-process imitation 教師データ生成

        _run_selfplay_parallel と同じパターンで worker を起動し、
        imitation/worker_<id>/shard_*.parquet に保存する。
        """
        sp_cfg = self._config.selfplay
        imitation_dir.mkdir(parents=True, exist_ok=True)
        num_threads = sp_cfg.get("worker_num_threads", 1)
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
        """self-play フェーズ

        selfplay.num_workers > 1 の場合は multi-process 実行。
        """
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
        logger.info(f"  total_steps: {sp_stats['total_steps']}")
        return sp_stats

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

        result = {
            "num_matches": total_matches,
            "total_steps": total_steps,
            "total_rounds": total_rounds,
            "output_dir": str(selfplay_dir),
            "worker_stats": worker_stats_list,
        }
        result.update(round_totals)
        return result

    def _run_learner(self, run_dir: Path, shard_dir: Path, model,
                     profiler=None) -> dict:
        """learner フェーズ"""
        training_device = resolve_device(
            self._config.training.get("device", "auto"))
        learner = Learner(
            config=self._as_dict(),
            model=model,
            run_dir=run_dir,
            device=training_device,
        )
        train_metrics = learner.train(shard_dir, profiler=profiler)
        learner.save_checkpoint(tag="final")
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
        eval_runner = EvaluationRunner(
            model=model,
            encoder=encoder,
            observation_mode=obs_mode,
            inference_device=eval_device,
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
            error_queue=error_queue)

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
                error_queue=error_queue)
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
                    error_queue,
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
        """設定からエンコーダを生成する"""
        enc_cfg = self._config.feature_encoder
        name = enc_cfg.get("name", "FlatFeatureEncoder")
        obs_mode = enc_cfg.get(
            "observation_mode",
            self._config.experiment.get("observation_mode", "full"),
        )
        if name == "ChannelTensorEncoder":
            return ChannelTensorEncoder(observation_mode=obs_mode)
        return FlatFeatureEncoder(observation_mode=obs_mode)

    def _create_model(self, encoder):
        """設定からモデルを生成する"""
        model_cfg = self._config.model
        hidden_dims = model_cfg.get("hidden_dims", [256, 128])
        value_heads = model_cfg.get("value_heads", ["round_delta"])

        # エンコーダの出力次元を取得
        meta = encoder.metadata()
        import math
        input_dim = math.prod(meta.output_shape)

        return MLPPolicyValueModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            value_heads=value_heads,
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
                "total_matches": sp.get("num_matches", 0),
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
            }
        if "imitation_metrics" in result:
            imi = result["imitation_metrics"]
            # imitation shard 数 (平坦 + worker サブディレクトリ)
            imi_dir = run_dir / "imitation"
            if imi_dir.exists():
                imi_flat = list(imi_dir.glob("shard_*.parquet"))
                imi_nested = list(imi_dir.glob("worker_*/shard_*.parquet"))
                imi_shard_count = len(set(imi_flat) | set(imi_nested))
            else:
                imi_shard_count = 0
            dg = imi.get("data_generation", {})
            phase_stats["imitation"] = {
                "total_steps": dg.get("total_steps", imi.get("total_steps", 0)),
                "num_updates": imi.get("num_updates", 0),
                "shard_count": imi_shard_count,
                "policy_loss": imi.get("policy_loss"),
                "num_workers": dg.get("num_workers", 1),
                "seed_strategy": dg.get("seed_strategy"),
            }
        if "train_metrics" in result:
            tm = result["train_metrics"]
            phase_stats["learner"] = {
                "total_steps": tm.get("total_steps", 0),
                "num_updates": tm.get("num_updates", 0),
                "policy_loss": tm.get("policy_loss"),
                "value_loss": tm.get("value_loss"),
                "mode": tm.get("mode"),
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

        # actor_type 内訳 (shard から集計)
        actor_type_counts = self._count_actor_types(run_dir)

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

        # imitation shard
        imi_dir = run_dir / "imitation"
        if imi_dir.exists():
            imi_flat = list(imi_dir.glob("shard_*.parquet"))
            imi_nested = list(imi_dir.glob("worker_*/shard_*.parquet"))
            imi_shard_count = len(set(imi_flat) | set(imi_nested))
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

        # learner checkpoint
        learner_ckpt = run_dir / "checkpoints" / "checkpoint_final.pt"
        artifacts["learner_checkpoint"] = {
            "exists": learner_ckpt.exists(),
            "path": "checkpoints/checkpoint_final.pt",
        }

        # eval
        eval_dir = run_dir / "eval"
        artifacts["eval"] = {
            "exists": eval_dir.exists() and any(eval_dir.iterdir()) if eval_dir.exists() else False,
            "path": "eval",
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
        """shard ファイルから actor_type ごとの件数を集計する"""
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
            except Exception:
                pass
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
