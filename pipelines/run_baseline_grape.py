#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run baseline GRAPE (imputation +/or label), normalize outputs into a unified artifact layout,
and (optionally) copy already-exported intermediates.

Features
- Supports BOTH GRAPE tasks:
    • train_mdi  → imputation  → baseline/<ds>/seedX/impute/
    • train_y    → label       → baseline/<ds>/seedX/label/
- Works with GRAPE CLI style where the data "domain" (e.g., `uci`) is a subcommand.
  Example:
     python train_mdi.py --seed 0 uci --data yacht
     python train_y.py  --seed 0 uci --data yacht
- Optional flag injection: `--inject_artifact_flags` inserts
  `--artifact_dir <ART> --dump_intermediate` **before** the domain token (e.g., `uci`).
- Robust result.pkl parsing (handles common fields and nested `curves`).

Outputs
- {artifact_dir}/baseline/{dataset}/seed{seed}/
    X_norm.npy, mask.npy, split_idx.json, omega_test_idx.npz, ... (if GRAPE exported them)
    impute/{result.pkl, metrics.json}
    label/{result.pkl,  metrics.json}

Examples
========
A) Train both tasks and export intermediates from GRAPE itself:
python pipelines/run_baseline_grape.py \
  --grape_root /content/GRAPE \
  --dataset yacht \
  --seed 0 \
  --artifact_dir /content/grapt_artifacts \
  --task both \
  --inject_artifact_flags \
  --mdi_cmd "python /content/GRAPE/train_mdi.py --seed 0 uci --data yacht" \
  --y_cmd   "python /content/GRAPE/train_y.py  --seed 0 uci --data yacht"

B) Provide full commands manually (flags placed before domain):
python pipelines/run_baseline_grape.py \
  --grape_root /content/GRAPE \
  --dataset yacht \
  --seed 0 \
  --artifact_dir /content/grapt_artifacts \
  --task both \
  --mdi_cmd "python /content/GRAPE/train_mdi.py --seed 0 --artifact_dir /content/grapt_artifacts --dump_intermediate uci --data yacht" \
  --y_cmd   "python /content/GRAPE/train_y.py  --seed 0 --artifact_dir /content/grapt_artifacts --dump_intermediate uci --data yacht"

C) Skip training; just collect existing PKLs:
python pipelines/run_baseline_grape.py \
  --grape_root /content/GRAPE \
  --dataset yacht \
  --seed 0 \
  --artifact_dir /content/grapt_artifacts \
  --task both \
  --collect_from_mdi /content/GRAPE/uci/test/yacht/m0/result.pkl \
  --collect_from_y   /content/GRAPE/uci/test/yacht/y0/result.pkl
"""

import os
import json
import glob
import shutil
import argparse
import subprocess
import time
from pathlib import Path

try:
    import joblib
except Exception:
    joblib = None

# ----------------------------- utils -----------------------------

def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# run_baseline_grape.py
def run_cmd(cmd, cwd=None, env=None):
    import os, shlex, subprocess, textwrap, time
    t0 = time.time()
    args = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    print("\n[run_cmd] CWD:", cwd or os.getcwd())
    print("[run_cmd] CMD:", " ".join(args))
    p = subprocess.run(args, cwd=cwd, env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.stdout:
        print(textwrap.indent(p.stdout, prefix="[child stdout] "))
    if p.stderr:
        print(textwrap.indent(p.stderr, prefix="[child stderr] "))
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with code {p.returncode}")
    return t0

def _glob_result_pkls(grape_root: Path, dataset: str):
    pat = str(grape_root / "uci" / "test" / dataset / "**" / "result.pkl")
    return sorted(glob.glob(pat, recursive=True), key=os.path.getmtime)


def find_latest_result_after(grape_root: Path, dataset: str, t_after: float):
    cands = _glob_result_pkls(grape_root, dataset)
    cands = [p for p in cands if os.path.getmtime(p) >= t_after - 1.0]
    return Path(cands[-1]) if cands else None


def find_latest_result_any(grape_root: Path, dataset: str):
    cands = _glob_result_pkls(grape_root, dataset)
    return Path(cands[-1]) if cands else None


def _to_dict(obj):
    if isinstance(obj, dict):
        return dict(obj)
    d = getattr(obj, "__dict__", None)
    return dict(d) if isinstance(d, dict) else {"_raw": str(obj)}


def parse_result_pkl(pkl_path: Path):
    """Robustly parse GRAPE's result.pkl into a flat metrics dict.
    Tries common keys and also recursively mines curves-like structures.
    """
    if joblib is None:
        raise RuntimeError("joblib 未安裝，無法解析 result.pkl。pip install joblib")
    obj = joblib.load(pkl_path)
    d = _to_dict(obj)

    metrics = {}
    preferred = [
        # imputation
        "impute_mae", "impute_rmse", "impute_nmae", "impute_nrmse",
        "impute_MEAN_MAE", "impute_GRAPE_MAE",
        # label
        "label_mae", "label_rmse", "label_auroc", "label_auc", "auroc", "auc",
        # misc
        "mask_rate", "best_val", "seed",
    ]
    for k in preferred:
        if k in d:
            metrics[k] = d[k]

    # Drill into curves-like structures and grab the last point (test/val)
    def mine_curves(node, prefix=""):
        out = {}
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{prefix}.{k}" if prefix else k
                out.update(mine_curves(v, p))
        elif isinstance(node, list) and node:
            last = node[-1]
            if isinstance(last, dict):
                for kk, vv in last.items():
                    key = f"{prefix}.{kk}".lower()
                    if any(t in key for t in ("mae","rmse","auc","auroc","acc","f1")):
                        out[key] = vv
        return out

    if "curves" in d:
        try:
            metrics.update(mine_curves(d["curves"], "curves"))
        except Exception:
            pass

    # Deep-scan for scalar metrics with metric-like key names
    def walk(node, path=""):
        out = {}
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{path}.{k}" if path else k
                if isinstance(v, (dict, list)):
                    out.update(walk(v, p))
                else:
                    try:
                        from numbers import Number
                        if isinstance(v, Number):
                            kl = k.lower()
                            if any(t in kl for t in ("mae","rmse","auc","auroc","acc","f1")):
                                out[p] = v
                    except Exception:
                        pass
        elif isinstance(node, list):
            for i, v in enumerate(node):
                out.update(walk(v, f"{path}[{i}]"))
        return out

    try:
        metrics.update(walk(d))
    except Exception:
        pass
    return metrics


def write_json(path: Path, obj: dict):
    makedirs(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def copy_if_exists(src: Path, dst: Path):
    if src and src.exists():
        makedirs(dst.parent)
        shutil.copy2(str(src), str(dst))
        print(f"[copy] {src} -> {dst}")
        return True
    return False

def copy_many_if_exist(src_dir: Path, dst_dir: Path, names):
    ok = False
    for n in names:
        if copy_if_exists(src_dir / n, dst_dir / n):
            ok = True
    return ok

def ensure_prep_baseline(save_root: Path) -> None:
    """
    prep-only 模式檢查：確認 baseline 中介物是否齊全。
    你目前只跑 GRAPE → 這四個檔案足夠做後續串接：
      - X_norm.npy
      - mask.npy
      - split_idx.json
      - omega_test_idx.npz
    """
    needed = ["X_norm.npy", "mask.npy", "split_idx.json", "omega_test_idx.npz"]
    missing = [n for n in needed if not (save_root / n).exists()]
    if missing:
        raise FileNotFoundError(f"[prep_only] baseline 中介物未齊全：缺 {', '.join(missing)}（root={save_root}）")

def ensure_prep_stage(save_root: Path, tag: str) -> None:
    """
    prep-only 階段性檢查：
      - impute: 需要 X_norm.npy / mask.npy  （可選再加 bipartite_edges.npz）
      - label : 需要 split_idx.json / omega_test_idx.npz
    """
    if tag == "impute":
        needed = ["X_norm.npy", "mask.npy"]
        # 若你想更嚴格，可解除下一行註解
        # needed += ["bipartite_edges.npz"]
    elif tag == "label":
        needed = ["split_idx.json", "omega_test_idx.npz"]
    else:
        raise ValueError(f"unknown tag: {tag}")

    missing = [n for n in needed if not (save_root / n).exists()]
    if missing:
        raise FileNotFoundError(f"[prep_only:{tag}] 檔案未齊全：缺 {', '.join(missing)}（root={save_root}）")

def _fallback_metrics(tag: str, out_dir: Path):
    """用中介 .npy 後備計算 RMSE/MAE；抓不到就回空 dict。"""
    try:
        import numpy as np
        if tag == "impute":
            ep = np.load(out_dir / "edge_pred.npy")
            el = np.load(out_dir / "edge_label.npy")
            rmse = float(np.sqrt(np.mean((ep - el) ** 2)))
            mae  = float(np.mean(np.abs(ep - el)))
            return {"test": {"rmse": rmse, "mae": mae}}
        if tag == "label":
            yp = np.load(out_dir / "y_pred.npy")
            yl = np.load(out_dir / "label_test.npy")
            rmse = float(np.sqrt(np.mean((yp - yl) ** 2)))
            mae  = float(np.mean(np.abs(yp - yl)))
            return {"test": {"rmse": rmse, "mae": mae}}
    except Exception:
        pass
    return {}

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Run/collect GRAPE baseline (mdi/y/both) and normalize artifacts")
    ap.add_argument("--grape_root", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--artifact_dir", type=str, required=True)

    ap.add_argument("--task", type=str, default="both", choices=["mdi","y","both"], help="which task(s) to run/collect")

    # Commands (optional). If omitted, defaults are composed.
    ap.add_argument("--mdi_cmd", type=str, default="", help="command to run train_mdi")
    ap.add_argument("--y_cmd",   type=str, default="", help="command to run train_y")

    # Direct collection (skip training)
    ap.add_argument("--collect_from_mdi", type=str, default="", help="path to imputation result.pkl")
    ap.add_argument("--collect_from_y",   type=str, default="", help="path to label result.pkl")

    # Env
    ap.add_argument("--env", type=str, default="", help="extra env as KEY=VAL,KEY2=VAL2")

    # Optional: copy already-exported intermediates from this folder
    ap.add_argument("--prep_source_dir", type=str, default="", help="folder containing X_norm/mask/splits, etc.")

    # Convenience: auto-append artifact flags to commands, inserted **before** the domain token (e.g., `uci`).
    ap.add_argument("--inject_artifact_flags", action="store_true",
                    help="insert --artifact_dir <ART> --dump_intermediate before the domain token (e.g., 'uci')")
    ap.add_argument("--allow_fallback",
                    action="store_true",
                    help="Allow fallback to latest existing result if this run fails or produces nothing (default: disabled/strict).",
    )
    ap.add_argument("--prep_only", action="store_true",
                    help="Export baseline intermediates only (no training). Will set GRAPT_PREP_ONLY=1 and pass --prep_only to train_*.")

    args = ap.parse_args()

    grape_root = Path(args.grape_root).resolve()
    if not grape_root.exists():
        raise FileNotFoundError(f"GRAPE 根目錄不存在：{grape_root}")

    save_root = Path(args.artifact_dir) / "baseline" / args.dataset / f"seed{args.seed}"
    impute_dir = save_root / "impute"
    label_dir  = save_root / "label"
    makedirs(impute_dir); makedirs(label_dir)

    # Compose defaults if missing
    def _default_mdi():
        return f"python {grape_root}/train_mdi.py --seed {args.seed} uci --data {args.dataset}"
    def _default_y():
        return f"python {grape_root}/train_y.py --seed {args.seed} uci --data {args.dataset}"

    mdi_cmd = args.mdi_cmd or _default_mdi()
    y_cmd   = args.y_cmd   or _default_y()

    # Inject flags BEFORE the domain token (e.g., 'uci') if requested
    # 既有的 inject_artifact_flags 區塊之外，再補一個保險
    if args.inject_artifact_flags:
        import shlex
        def inject_flags_before_domain(cmd: str, art_dir: str):
            if "--artifact_dir" in cmd or "--dump_intermediate" in cmd:
                return cmd
            flags = ["--artifact_dir", art_dir, "--dump_intermediate"]
            tokens = shlex.split(cmd)
            domains = {"uci", "mimic", "physionet", "eicu"}
            insert_at = None
            for i, tok in enumerate(tokens):
                if tok in domains:
                    insert_at = i
                    break
            if insert_at is None:
                tokens.extend(flags)
            else:
                tokens = tokens[:insert_at] + flags + tokens[insert_at:]
            def q(x):
                return shlex.quote(x) if (" " in x or ";" in x) else x
            return " ".join(q(t) for t in tokens)
        def inject_before_domain(cmd: str, extra_flags: list[str]) -> str:
            """
            將 extra_flags 插入在 domain token（如 uci/mimic/physionet/eicu）之前。
            cmd 是一條字串命令；回傳新字串命令。
            """
            tokens = shlex.split(cmd)
            domains = {"uci", "mimic", "physionet", "eicu"}
            insert_at = None
            for i, tok in enumerate(tokens):
                if tok in domains:
                    insert_at = i
                    break
            if insert_at is None:
                tokens.extend(extra_flags)
            else:
                tokens = tokens[:insert_at] + extra_flags + tokens[insert_at:]
            def q(x):
                return shlex.quote(x) if (" " in x or ";" in x) else x
            return " ".join(q(t) for t in tokens)

        mdi_cmd = inject_flags_before_domain(mdi_cmd, args.artifact_dir)
        y_cmd   = inject_flags_before_domain(y_cmd,   args.artifact_dir)
        if args.prep_only:
            mdi_cmd = inject_before_domain(mdi_cmd, ["--prep_only"])
            y_cmd   = inject_before_domain(y_cmd,   ["--prep_only"])

    # Build env
    env = os.environ.copy()
    if args.prep_only:
        env["GRAPT_PREP_ONLY"] = "1"
    if args.env:
        for pair in args.env.split(","):
            if pair.strip():
                k, v = pair.split("=", 1)
                env[k.strip()] = v.strip()

    # Save run config for reproducibility
    write_json(save_root / "run_args.json", {
        "dataset": args.dataset,
        "seed": args.seed,
        "grape_root": str(grape_root),
        "task": args.task,
        "mdi_cmd": mdi_cmd,
        "y_cmd": y_cmd,
        "collect_from_mdi": args.collect_from_mdi,
        "collect_from_y": args.collect_from_y,
        "prep_source_dir": args.prep_source_dir,
        "inject_artifact_flags": args.inject_artifact_flags,
        "env": args.env,
        "timestamp": now_str(),
    })

    def handle_task(tag: str, cmd: str, collect_from: str, out_dir: Path):
        pkl_path = Path(collect_from) if collect_from else None
        if pkl_path and not pkl_path.exists():
            raise FileNotFoundError(f"指定的 {tag} result.pkl 不存在：{pkl_path}")

        if not pkl_path:
            # 嚴格：用 run_cmd 回傳的 t0 當門檻；失敗預設直接拋錯
            try:
                t0 = run_cmd(cmd, cwd=str(grape_root), env=env)  # 啟動前的時間戳
                # ★ prep-only：不要求新 result.pkl；改驗證 baseline 中介物是否齊全後直接返回
                if args.prep_only:
                    try:
                        ensure_prep_stage(save_root, tag)   # tag 是 "impute" 或 "label"
                    except FileNotFoundError as e:
                        raise FileNotFoundError(
                            f"[prep_only] 本次執行後 baseline 中介物仍不齊全，請檢查 gnn 匯出與路徑。詳情：{e}"
                        )
                    else:
                        print(f"[OK] prep-only：baseline 中介物已就緒 → {save_root}")
                    return
            except Exception as e:
                if getattr(args, "allow_fallback", False):
                    print(f"[warn] {tag} 子程序失敗：{e}；allow_fallback 啟用，改用最近一份舊 result.pkl")
                    pkl_path = find_latest_result_any(grape_root, args.dataset)
                    if not pkl_path:
                        raise FileNotFoundError("allow_fallback 啟用，但仍找不到任何舊的 result.pkl")
                else:
                    raise

            if not getattr(args, "allow_fallback", False):
                pkl_path = find_latest_result_after(grape_root, args.dataset, t0)
                if not pkl_path:
                    raise FileNotFoundError(
                        f"這次執行後沒有新產物（dataset={args.dataset}）。"
                        "若你暫時要接受舊結果，請加 --allow_fallback 後再跑。"
                    )
            else:
                # allow_fallback：保險再找一次
                if not pkl_path:
                    pkl_path = find_latest_result_after(grape_root, args.dataset, t0) \
                            or find_latest_result_any(grape_root, args.dataset)
                if not pkl_path:
                    raise FileNotFoundError("allow_fallback 啟用，但找不到任何 result.pkl")

        shutil.copy2(str(pkl_path), str(out_dir / "result.pkl"))
        print(f"[copy] {pkl_path} -> {out_dir / 'result.pkl'}")

        # —— 安全輸出 metrics（決不寫空檔）——
        try:
            metrics = parse_result_pkl(pkl_path)  # 可能回 {}
            dst = out_dir / "metrics.json"
            if metrics and isinstance(metrics, dict) and len(metrics) > 0:
                write_json(dst, metrics)
                print(f"[ok] {tag} metrics.json 輸出（來源=pkl）：{dst}")
            else:
                alt = pkl_path.parent / "metrics.json"  # 訓練腳本通常會寫在同目錄
                if alt.exists() and alt.stat().st_size > 0:
                    shutil.copy2(str(alt), str(dst))
                    print(f"[copy] {alt} -> {dst}")
                else:
                    fm = _fallback_metrics(tag, out_dir)
                    if fm:
                        write_json(dst, fm)
                        print(f"[ok] {tag} metrics.json 輸出（來源=fallback npy）：{dst}")
                    else:
                        print(f"[warn] {tag} 沒有可寫入的指標（pkl/原檔/npy 都不可用），跳過以避免覆蓋成空檔。")
        except Exception as e:
            print(f"[warn] 解析 {tag} result.pkl 失敗：{e}；不覆蓋原有 metrics.json。")

    if args.task in ("mdi","both"):
        handle_task("impute", mdi_cmd, args.collect_from_mdi, impute_dir)
    if args.task in ("y","both"):
        handle_task("label",  y_cmd,  args.collect_from_y,  label_dir)

    # Copy intermediates if provided separately
    prep_src = Path(args.prep_source_dir) if args.prep_source_dir else None
    if prep_src and prep_src.exists():
        names = [
            "X_norm.npy","mask.npy","y.npy","split_idx.json",
            "feature_names.json","scaler.pkl",
            "bipartite_edges.npz","omega_test_idx.npz",
            "imputed.npy","y_pred.npy"
        ]
        copy_many_if_exist(prep_src, save_root, names)

    print(f"[OK] Baseline artifacts 準備完成：{save_root}")
    print(" - 檢查是否已有：X_norm.npy / mask.npy / split_idx.json / omega_test_idx.npz / (bipartite_edges.npz?)")


if __name__ == "__main__":
    main()
