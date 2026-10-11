#!/usr/bin/env python3
"""Retrain the selected three-model combo through 2025 and evaluate January 2026.

This experiment intentionally locks every model hyperparameter to the selected
2025H1 release. Only the training end date and OOS date range change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/kemove/anaconda3/envs/rainbow/bin/python")
DATASET = Path(
    "/data/features/alpha/bar_5m/alpha_all_1111_strict_cszscore_clip20/"
    "2024-11-28_2026-01-31"
)
BASELINE_RELEASE = Path(
    "/data/features/alpha/releases/lgbm1h_mlp4h8h_selected_2025H1"
)
BASELINE_MODEL_ROOT = BASELINE_RELEASE / "models"
BASELINE_ENSEMBLE = BASELINE_RELEASE / "ensemble/signal_manifest.json"
OUTPUT_ROOT = Path(
    "/data/features/alpha/models/"
    "lgbm1h_mlp4h8h_rolling_train20241130_20251231_"
    "oos20260101_20260131_v1"
)

TRAIN_START = 20241130
TRAIN_END = 20251231
EVAL_START = 20260101
EVAL_END = 20260131
MAX_ALLOWED_MEMORY_BYTES = 55 * 1024**3
TARGETS = [
    "ret_15m",
    "ret_1h",
    "ret_2h",
    "ret_4h",
    "ret_8h",
    "retcombo1",
    "retcombo2",
]
SELECTED_BOOKS = [
    {
        "name": "book_500k",
        "book_size_usd": 500_000,
        "candidate_pool_each_side": 40,
        "max_name_weight": 0.02,
        "turnover_cost": 0.8,
        "risk_budget_4h_bps": 450.0,
    },
    {
        "name": "book_5m",
        "book_size_usd": 5_000_000,
        "candidate_pool_each_side": 75,
        "max_name_weight": 0.01,
        "turnover_cost": 0.65,
        "risk_budget_4h_bps": 350.0,
    },
    {
        "name": "book_20m",
        "book_size_usd": 20_000_000,
        "candidate_pool_each_side": 100,
        "max_name_weight": 0.015,
        "turnover_cost": 0.65,
        "risk_budget_4h_bps": 0.0,
    },
]

MLP_TARGETS = {
    "ret_4h": {
        "purge_bars": 48,
        "baseline": BASELINE_MODEL_ROOT / "mlp_ret4h/training_meta.json",
        "target_cache": Path(
            "/data/features/alpha/models/"
            "mlp_raw_all_targets_csdemean_bs16_lr5e-4_epoch1_"
            "train20241130_20250630_oos20250701_20260131_v1/"
            "models/ret_4h/target_aligned.npy"
        ),
    },
    "ret_8h": {
        "purge_bars": 96,
        "baseline": BASELINE_MODEL_ROOT / "mlp_ret8h/training_meta.json",
        "target_cache": Path(
            "/data/features/alpha/models/"
            "mlp_raw_all_targets_csdemean_bs16_lr5e-4_epoch1_"
            "train20241130_20250630_oos20250701_20260131_v1/"
            "models/ret_8h/target_aligned.npy"
        ),
    },
}
LGB_TARGET_CACHE = Path(
    "/data/features/alpha/models/"
    "lgbm_ret1h_csdemean_l512_lr0p05_bin63_r50_"
    "train20241130_20250630_eval2025q3_v1/target_aligned.npy"
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, dry_run: bool) -> None:
    print(json.dumps({"event": "run", "command": command}), flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": "/tmp/mpl",
            "XDG_CACHE_HOME": "/tmp",
            "OMP_NUM_THREADS": "16",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _require_bounded_memory() -> int:
    unified_path = None
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            unified_path = relative.lstrip("/")
            break
    if unified_path is None:
        raise RuntimeError("cgroup v2 process path is unavailable")
    memory_max_path = Path("/sys/fs/cgroup") / unified_path / "memory.max"
    if not memory_max_path.is_file():
        raise RuntimeError("cgroup v2 memory.max is unavailable")
    raw_limit = memory_max_path.read_text().strip()
    if raw_limit == "max":
        raise RuntimeError(
            "refusing an unbounded run; launch through systemd-run with MemoryMax=55G"
        )
    limit = int(raw_limit)
    if limit > MAX_ALLOWED_MEMORY_BYTES:
        raise RuntimeError(
            f"cgroup memory.max={limit} exceeds the 55 GiB safety limit"
        )
    return limit


def _check_equal(payload: dict[str, Any], expected: dict[str, Any], source: Path) -> None:
    mismatches = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"baseline parameter mismatch in {source}: {mismatches}")


def _validate_baseline() -> dict[str, Any]:
    required = [
        DATASET / "factors.npy",
        DATASET / "valid.npy",
        DATASET / "timeline.npy",
        BASELINE_ENSEMBLE,
        LGB_TARGET_CACHE,
        *[Path(value["target_cache"]) for value in MLP_TARGETS.values()],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required inputs are missing: {missing}")

    for target, config in MLP_TARGETS.items():
        source = Path(config["baseline"])
        payload = json.loads(source.read_text())
        _check_equal(
            payload,
            {
                "batch_size_t": 16,
                "hidden_dims": [256, 128],
                "dropout": 0.3,
                "ema_decay": 0.999,
                "learning_rate": 0.0005,
                "weight_decay": 0.0,
                "epochs": 1,
                "seed": 20260916,
                "target_transform": "csdemean",
                "purge_bars": config["purge_bars"],
            },
            source,
        )
        if Path(payload["target"]).name != f"{target}.parquet":
            raise ValueError(f"baseline target mismatch in {source}")

    lgb_source = BASELINE_MODEL_ROOT / "lgbm_ret1h/training_meta.json"
    lgb_payload = json.loads(lgb_source.read_text())
    _check_equal(lgb_payload, {"num_boost_round": 50}, lgb_source)
    expected_lgb = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 512,
        "max_depth": -1,
        "min_data_in_leaf": 5000,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 10.0,
        "min_gain_to_split": 0.0,
        "max_bin": 63,
        "bin_construct_sample_cnt": 500000,
        "histogram_pool_size": 2048,
        "num_threads": 16,
        "seed": 20260917,
    }
    _check_equal(lgb_payload["training_params"], expected_lgb, lgb_source)
    return {
        "baseline_release": str(BASELINE_RELEASE),
        "baseline_ensemble_sha256": _sha256(BASELINE_ENSEMBLE),
        "factor_order_sha256": json.loads((DATASET / "manifest.json").read_text())[
            "factor_order_sha256"
        ],
        "locked_model_parameters": {
            "lgbm_ret1h": expected_lgb | {"num_boost_round": 50},
            "mlp": {
                "hidden_dims": [256, 128],
                "batch_size_t": 16,
                "dropout": 0.3,
                "ema_decay": 0.999,
                "learning_rate": 0.0005,
                "weight_decay": 0.0,
                "epochs": 1,
                "seed": 20260916,
                "target_transform": "csdemean",
            },
        },
    }


def _lgb_command(output_root: Path) -> list[str]:
    output = output_root / "models/lgbm_ret1h"
    stage = "all"
    extra: list[str] = []
    if (output / "train_dataset.bin").is_file() and not (output / "model.txt").is_file():
        stage = "train-predict"
        extra = ["--prepared-dir", str(output)]
    return [
        str(PYTHON),
        str(ROOT / "scripts/train_lgbm_npy.py"),
        "--stage",
        stage,
        "--dataset-dir",
        str(DATASET),
        "--target",
        "/data/targets/bar/5m_wide/ret_1h.parquet",
        "--target-cache",
        str(LGB_TARGET_CACHE),
        "--output-dir",
        str(output),
        "--train-start-date",
        "2024-11-30",
        "--train-end-date",
        "2025-12-31",
        "--eval-start-date",
        "2026-01-01",
        "--eval-end-date",
        "2026-01-31",
        "--monitor-start-date",
        "2026-01-01",
        "--monitor-end-date",
        "2026-01-31",
        "--monitor-sample-time-rows",
        "0",
        "--purge-bars",
        "48",
        "--target-winsor-quantiles",
        "0.005",
        "0.995",
        "--num-threads",
        "16",
        "--max-bin",
        "63",
        "--bin-construct-sample-cnt",
        "500000",
        "--histogram-pool-size-mb",
        "2048",
        "--memory-budget-gib",
        "50",
        "--objective",
        "regression",
        "--num-boost-round",
        "50",
        "--learning-rate",
        "0.05",
        "--num-leaves",
        "512",
        "--max-depth",
        "-1",
        "--min-data-in-leaf",
        "5000",
        "--feature-fraction",
        "0.9",
        "--bagging-fraction",
        "0.9",
        "--bagging-freq",
        "1",
        "--lambda-l1",
        "0",
        "--lambda-l2",
        "10",
        "--min-gain-to-split",
        "0",
        "--seed",
        "20260917",
        "--log-interval",
        "10",
        *extra,
    ]


def _mlp_command(output_root: Path, target: str, config: dict[str, Any]) -> list[str]:
    output = output_root / f"models/mlp_{target}"
    command = [
        str(PYTHON),
        str(ROOT / "scripts/train_mlp_npy.py"),
        "--dataset-dir",
        str(DATASET),
        "--target",
        f"/data/targets/bar/5m_wide/{target}.parquet",
        "--target-cache",
        str(config["target_cache"]),
        "--output-dir",
        str(output),
        "--epochs",
        "1",
        "--batch-size",
        "16",
        "--hidden-dims",
        "256",
        "128",
        "--dropout",
        "0.3",
        "--ema-decay",
        "0.999",
        "--learning-rate",
        "0.0005",
        "--weight-decay",
        "0",
        "--train-start-date",
        "20241130",
        "--train-end-date",
        "20251231",
        "--eval-start-date",
        "20260101",
        "--eval-end-date",
        "20260131",
        "--purge-bars",
        str(config["purge_bars"]),
        "--top-k",
        "10",
        "--target-winsor-quantiles",
        "0.005",
        "0.995",
        "--target-transform",
        "csdemean",
        "--seed",
        "20260916",
        "--device",
        "cuda",
        "--log-interval",
        "250",
    ]
    checkpoint = output / "training_state_last.pt"
    if checkpoint.is_file() and not (output / "signal_manifest.json").is_file():
        command.extend(["--resume", str(checkpoint)])
    return command


def _train(output_root: Path, *, dry_run: bool) -> None:
    lgb_manifest = output_root / "models/lgbm_ret1h/signal_manifest.json"
    if not lgb_manifest.is_file():
        _run(_lgb_command(output_root), dry_run=dry_run)
    for target, config in MLP_TARGETS.items():
        manifest = output_root / f"models/mlp_{target}/signal_manifest.json"
        if not manifest.is_file():
            _run(_mlp_command(output_root, target, config), dry_run=dry_run)


def _ensemble(output_root: Path, *, dry_run: bool) -> Path:
    output = output_root / "ensemble"
    manifest = output / "signal_manifest.json"
    if not manifest.is_file():
        _run(
            [
                str(PYTHON),
                str(ROOT / "scripts/ensemble_mlp_signals.py"),
                "--signal-manifests",
                str(output_root / "models/lgbm_ret1h/signal_manifest.json"),
                str(output_root / "models/mlp_ret_4h/signal_manifest.json"),
                str(output_root / "models/mlp_ret_8h/signal_manifest.json"),
                "--output-dir",
                str(output),
                "--valid-mask",
                str(DATASET / "valid.npy"),
                "--valid-timeline",
                str(DATASET / "timeline.npy"),
                "--weights",
                "1",
                "1",
                "1",
                "--names",
                "lgbm_ret1h",
                "mlp_ret4h",
                "mlp_ret8h",
                "--allow-factor-hash-mismatch",
            ],
            dry_run=dry_run,
        )
    return manifest


def _resolve(manifest_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _slice_baseline_january(output_root: Path, *, dry_run: bool) -> Path:
    output = output_root / "baseline_2025H1_january_2026"
    output_manifest = output / "signal_manifest.json"
    if output_manifest.is_file() or dry_run:
        return output_manifest
    source = json.loads(BASELINE_ENSEMBLE.read_text())
    signal_entry = source["signal"]
    signal_value = signal_entry["path"] if isinstance(signal_entry, dict) else signal_entry
    source_signal = np.load(_resolve(BASELINE_ENSEMBLE, signal_value), mmap_mode="r")
    source_timeline = np.load(_resolve(BASELINE_ENSEMBLE, source["timeline"]), mmap_mode="r")
    selected = np.flatnonzero(
        (source_timeline[:, 0] >= EVAL_START) & (source_timeline[:, 0] <= EVAL_END)
    )
    if not len(selected) or np.any(np.diff(selected) != 1):
        raise ValueError("baseline January 2026 range is missing or non-contiguous")
    rows = slice(int(selected[0]), int(selected[-1]) + 1)
    output.mkdir(parents=True, exist_ok=True)
    signal_path = output / "oos_signal.npy"
    partial = output / ".oos_signal.npy.partial"
    destination = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=source_signal[rows].shape
    )
    destination[:] = source_signal[rows]
    destination.flush()
    del destination
    os.replace(partial, signal_path)
    timeline_path = output / "oos_timeline.npy"
    np.save(timeline_path, np.asarray(source_timeline[rows]))
    sliced = dict(source)
    sliced["kind"] = "date_sliced_weighted_signal_ensemble"
    sliced["source_manifest"] = str(BASELINE_ENSEMBLE)
    sliced["eval_dates"] = [EVAL_START, EVAL_END]
    sliced["signal"] = {
        "path": str(signal_path),
        "shape": list(source_signal[rows].shape),
        "dtype": "float32",
        "sha256": _sha256(signal_path),
        "valid_prediction_count": int(np.isfinite(source_signal[rows]).sum()),
        "mask": "preserved from source ensemble",
    }
    sliced["timeline"] = str(timeline_path)
    _atomic_json(output_manifest, sliced)
    return output_manifest


def _evaluate_signal(manifest: Path, output: Path, *, dry_run: bool) -> None:
    if output.is_file():
        return
    _run(
        [
            str(PYTHON),
            str(ROOT / "scripts/evaluate_mlp_signal.py"),
            "--signal-manifest",
            str(manifest),
            "--targets",
            *TARGETS,
            "--top-k",
            "10",
            "--top-ks",
            "20",
            "30",
            "--winsor-quantiles",
            "0.005",
            "0.995",
            "--output",
            str(output),
        ],
        dry_run=dry_run,
    )


def _metric(payload: dict[str, Any], target: str, variant: str) -> dict[str, float]:
    result = payload["results"][target][variant]
    return {
        "global_ic": float(result["ic"]["global"]),
        "mean_cs_ic": float(result["ic"]["cross_section_mean"]),
        "daily_icir_sqrt365": float(result["ic"]["daily_icir_sqrt365"]),
        "top10_ls_bps": float(result["top_return"]["top10"]["long_short_bps"]),
        "top20_ls_bps": float(result["top_return"]["top20"]["long_short_bps"]),
        "top30_ls_bps": float(result["top_return"]["top30"]["long_short_bps"]),
    }


def _compare(output_root: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    evaluation_root = output_root / "evaluation"
    rolling_path = evaluation_root / "rolling_full_oos.json"
    baseline_path = evaluation_root / "baseline_2025H1_january_2026.json"
    rolling = json.loads(rolling_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    comparison: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "fixed-parameter rolling train through 2025",
        "train_dates": [TRAIN_START, TRAIN_END],
        "oos_dates": [EVAL_START, EVAL_END],
        "selection_policy": "no OOS tuning; fixed model parameters and equal weights",
        "results": {},
        "artifacts": {
            "rolling_evaluation": str(rolling_path),
            "baseline_evaluation": str(baseline_path),
        },
    }
    for target in TARGETS:
        comparison["results"][target] = {}
        for variant in ("raw", "funding_adjusted"):
            candidate = _metric(rolling, target, variant)
            reference = _metric(baseline, target, variant)
            comparison["results"][target][variant] = {
                "rolling": candidate,
                "baseline_2025H1": reference,
                "delta": {key: candidate[key] - reference[key] for key in candidate},
            }
    _atomic_json(evaluation_root / "rolling_vs_2025H1_comparison.json", comparison)


def _evaluate(output_root: Path, *, dry_run: bool) -> None:
    rolling_manifest = _ensemble(output_root, dry_run=dry_run)
    baseline_manifest = _slice_baseline_january(output_root, dry_run=dry_run)
    evaluation_root = output_root / "evaluation"
    if not dry_run:
        evaluation_root.mkdir(parents=True, exist_ok=True)
    _evaluate_signal(
        rolling_manifest,
        evaluation_root / "rolling_full_oos.json",
        dry_run=dry_run,
    )
    _evaluate_signal(
        baseline_manifest,
        evaluation_root / "baseline_2025H1_january_2026.json",
        dry_run=dry_run,
    )
    _compare(output_root, dry_run=dry_run)


def _prepare_portfolio_adapter(
    output_root: Path,
    name: str,
    manifest: Path,
    *,
    dry_run: bool,
) -> Path:
    output = output_root / "strategy" / name
    adapter = output / "input_adapter"
    if not (adapter / "trade_amt_aligned.npy").is_file():
        _run(
            [
                str(PYTHON),
                str(ROOT / "scripts/evaluate_mlp_signal_suite.py"),
                "--signal-manifest",
                str(manifest),
                "--feature-dataset",
                str(DATASET),
                "--variant",
                "raw",
                "--mode",
                "quick",
                "--start-date",
                str(EVAL_START),
                "--end-date",
                str(EVAL_END),
                "--quick-targets",
                "ret_1h",
                "--output-dir",
                str(output),
            ],
            dry_run=dry_run,
        )
    return adapter


def _portfolio_command(adapter: Path, output: Path, book: dict[str, Any]) -> list[str]:
    return [
        str(PYTHON),
        str(ROOT / "scripts/backtest_mlp_long_short_portfolio.py"),
        "--baseline-dir",
        str(adapter),
        "--variant",
        "raw",
        "--start-date",
        str(EVAL_START),
        "--end-date",
        str(EVAL_END),
        "--book-size-usd",
        str(book["book_size_usd"]),
        "--rebalance-bars",
        "1",
        "--signal-horizon-bars",
        "48",
        "--volume-lookback-bars",
        "48",
        "--position-volume-lookback-bars",
        "48",
        "--planning-participation-rate",
        "0.02",
        "--posterior-participation-rate",
        "0.05",
        "--realized-cost-bps",
        "10",
        "--turnover-cost",
        str(book["turnover_cost"]),
        "--funding-mode",
        "apply",
        "--candidate-pool-each-side",
        str(book["candidate_pool_each_side"]),
        "--max-name-weight",
        str(book["max_name_weight"]),
        "--max-net-weight",
        "0.10",
        "--max-turnover-weight",
        "1.0",
        "--net-penalty",
        "5.0",
        "--max-position-volume-participation-rate",
        "0.05",
        "--invalid-position-policy",
        "instant-last-valid",
        "--invalid-grace-bars",
        "3",
        "--invalid-exit-bars",
        "48",
        "--invalid-hold-signal-threshold",
        "0.5",
        "--forced-liquidation-extra-cost-bps",
        "10",
        "--volatility-lookback-bars",
        "288",
        "--volatility-horizon-bars",
        "48",
        "--min-volatility-observations",
        "96",
        "--missing-volatility-policy",
        "budget",
        "--risk-slack-penalty",
        "100",
        "--risk-budget-4h-bps",
        str(book["risk_budget_4h_bps"]),
        "--output",
        str(output),
    ]


def _portfolio(output_root: Path, *, dry_run: bool) -> None:
    manifests = {
        "rolling": output_root / "ensemble/signal_manifest.json",
        "baseline_2025H1": output_root
        / "baseline_2025H1_january_2026/signal_manifest.json",
    }
    adapters = {
        name: _prepare_portfolio_adapter(
            output_root, name, manifest, dry_run=dry_run
        )
        for name, manifest in manifests.items()
    }
    for name, adapter in adapters.items():
        for book in SELECTED_BOOKS:
            output = output_root / "strategy" / name / "portfolio" / f"{book['name']}.json"
            if output.is_file():
                continue
            if not dry_run:
                output.parent.mkdir(parents=True, exist_ok=True)
            _run(_portfolio_command(adapter, output, book), dry_run=dry_run)

    if dry_run:
        return
    metrics = [
        "total_net_pnl_usd",
        "mean_daily_net_pnl_usd",
        "daily_net_sharpe_sqrt365",
        "calmar_pnl_ratio_annualized",
        "max_drawdown_usd",
        "max_drawdown_over_book",
        "positive_day_fraction",
        "mean_daily_turnover_usd",
        "mean_daily_turnover_over_book",
        "mean_gross_exposure_usd",
        "mean_gross_utilization",
        "planning_failures",
        "execution_failures",
    ]
    comparison: dict[str, Any] = {
        "schema_version": 1,
        "date_range": [EVAL_START, EVAL_END],
        "selection_policy": "fixed 2025H1 release portfolio parameters; no tuning",
        "books": {},
    }
    for book in SELECTED_BOOKS:
        results = {}
        for name in manifests:
            path = output_root / "strategy" / name / "portfolio" / f"{book['name']}.json"
            payload = json.loads(path.read_text())["result"]
            results[name] = {key: payload[key] for key in metrics if key in payload}
        comparison["books"][book["name"]] = {
            "parameters": book,
            **results,
            "delta_rolling_minus_baseline": {
                key: results["rolling"][key] - results["baseline_2025H1"][key]
                for key in metrics
                if key in results["rolling"]
                and key in results["baseline_2025H1"]
                and isinstance(results["rolling"][key], (int, float))
            },
        }
    _atomic_json(output_root / "strategy/rolling_vs_2025H1_comparison.json", comparison)


def _finalize_artifacts(output_root: Path) -> None:
    source_strategy = BASELINE_RELEASE / "release_configs/portfolio_selected_2025H1.yaml"
    strategy = yaml.safe_load(source_strategy.read_text())
    strategy["name"] = "lgbm1h_mlp4h8h_equal_r50_rolling_train2025_candidate"
    strategy["signal"]["manifest"] = str(output_root / "ensemble/signal_manifest.json")
    strategy["backtest_contract"]["start_date"] = EVAL_START
    strategy["backtest_contract"]["end_date"] = EVAL_END
    strategy["selection_result"] = str(
        output_root / "strategy/rolling_vs_2025H1_comparison.json"
    )
    strategy["backup"] = str(output_root)
    strategy["training_contract"] = {
        "train_start_date": TRAIN_START,
        "train_end_date": TRAIN_END,
        "oos_start_date": EVAL_START,
        "oos_end_date": EVAL_END,
        "hyperparameter_policy": "copied from 2025H1 release; no OOS tuning",
    }
    strategy_path = output_root / "configs/portfolio_rolling_train2025_candidate.yaml"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_partial = strategy_path.with_name(f".{strategy_path.name}.partial")
    strategy_partial.write_text(yaml.safe_dump(strategy, sort_keys=False))
    os.replace(strategy_partial, strategy_path)

    relative_paths = [
        Path("experiment_spec.json"),
        Path("configs/portfolio_rolling_train2025_candidate.yaml"),
        Path("models/lgbm_ret1h/model.txt"),
        Path("models/lgbm_ret1h/training_meta.json"),
        Path("models/lgbm_ret1h/signal_manifest.json"),
        Path("models/mlp_ret_4h/model_ema.pt"),
        Path("models/mlp_ret_4h/model_ema.onnx"),
        Path("models/mlp_ret_4h/training_meta.json"),
        Path("models/mlp_ret_4h/signal_manifest.json"),
        Path("models/mlp_ret_8h/model_ema.pt"),
        Path("models/mlp_ret_8h/model_ema.onnx"),
        Path("models/mlp_ret_8h/training_meta.json"),
        Path("models/mlp_ret_8h/signal_manifest.json"),
        Path("ensemble/oos_signal.npy"),
        Path("ensemble/signal_manifest.json"),
        Path("evaluation/rolling_full_oos.json"),
        Path("evaluation/baseline_2025H1_january_2026.json"),
        Path("evaluation/rolling_vs_2025H1_comparison.json"),
        Path("strategy/rolling_vs_2025H1_comparison.json"),
        *[
            Path("strategy") / signal / "portfolio" / f"{book['name']}.json"
            for signal in ("rolling", "baseline_2025H1")
            for book in SELECTED_BOOKS
        ],
    ]
    missing = [str(path) for path in relative_paths if not (output_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot finalize incomplete experiment: {missing}")
    predictive = json.loads(
        (output_root / "evaluation/rolling_vs_2025H1_comparison.json").read_text()
    )
    deltas = [
        value
        for target in predictive["results"].values()
        for variant in target.values()
        for value in variant["delta"].values()
    ]
    artifact_manifest = {
        "schema_version": 1,
        "experiment": "rolling_train_2025_fixed_selected_combo",
        "train_dates": [TRAIN_START, TRAIN_END],
        "oos_dates": [EVAL_START, EVAL_END],
        "deployable_models": {
            "lgbm_ret1h": "models/lgbm_ret1h/model.txt",
            "mlp_ret4h_ema_onnx": "models/mlp_ret_4h/model_ema.onnx",
            "mlp_ret8h_ema_onnx": "models/mlp_ret_8h/model_ema.onnx",
        },
        "combo_manifest": "ensemble/signal_manifest.json",
        "predictive_comparison": "evaluation/rolling_vs_2025H1_comparison.json",
        "strategy_comparison": "strategy/rolling_vs_2025H1_comparison.json",
        "candidate_strategy_config": "configs/portfolio_rolling_train2025_candidate.yaml",
        "predictive_delta_count": len(deltas),
        "predictive_positive_delta_count": sum(value > 0 for value in deltas),
        "checksums": "DEPLOYABLE_SHA256SUMS",
    }
    manifest_path = output_root / "ARTIFACT_MANIFEST.json"
    _atomic_json(manifest_path, artifact_manifest)
    checksum_paths = [*relative_paths, Path("ARTIFACT_MANIFEST.json")]
    lines = [
        f"{_sha256(output_root / path)}  ./{path.as_posix()}"
        for path in sorted(checksum_paths)
    ]
    checksum_path = output_root / "DEPLOYABLE_SHA256SUMS"
    partial = output_root / ".DEPLOYABLE_SHA256SUMS.partial"
    partial.write_text("\n".join(lines) + "\n")
    os.replace(partial, checksum_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--stage", choices=["train", "evaluate", "portfolio", "all"], default="all"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    baseline = _validate_baseline()
    cgroup_memory_max = (
        MAX_ALLOWED_MEMORY_BYTES if args.dry_run else _require_bounded_memory()
    )
    spec = {
        "schema_version": 1,
        "experiment": "rolling_train_2025_fixed_selected_combo",
        "created_at_unix": time.time(),
        "dataset": str(DATASET),
        "train_dates": [TRAIN_START, TRAIN_END],
        "oos_dates": [EVAL_START, EVAL_END],
        "targets": {"lgbm": "ret_1h", "mlp": ["ret_4h", "ret_8h"]},
        "ensemble": "valid-only CS-zscore each component, equal-weight mean",
        "oos_policy": "fixed before OOS; no early stopping or OOS parameter selection",
        "resource_limits": {
            "cpu_threads": 16,
            "lgbm_memory_budget_gib": 50,
            "cgroup_memory_high_gib": 50,
            "cgroup_memory_max_bytes": cgroup_memory_max,
        },
        **baseline,
    }
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(output_root / "experiment_spec.json", spec)
    else:
        print(json.dumps(spec, indent=2, sort_keys=True), flush=True)

    if args.stage in {"train", "all"}:
        _train(output_root, dry_run=args.dry_run)
    if args.stage in {"evaluate", "all"}:
        _evaluate(output_root, dry_run=args.dry_run)
    if args.stage in {"portfolio", "all"}:
        _portfolio(output_root, dry_run=args.dry_run)
        if not args.dry_run:
            _finalize_artifacts(output_root)
    print(json.dumps({"event": "complete", "output_root": str(output_root)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
