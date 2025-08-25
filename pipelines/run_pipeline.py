#!/usr/bin/env python3
"""
GRAFT Orchestrator — 完成版（variants + overlay manifest 嚴格模式）
-----------------------------------------------------------------
- 單一入口 `run_pipeline.py`：依 `--modules` / `--order` 串接 T2G / LUNAR / RANDOM / GRAPE。
- 嚴格 manifest：呼叫 GRAPE 前，**必定**產出 `overlay_manifest.json`，並以環境變數
  `GRAFT_OVERLAY_MANIFEST`、`GRAFT_MASK_OP` 傳入；`gnn_y.py` / `gnn_mdi.py` 會據此組合遮罩。
- 變體工作區：每次 run 建立 `{artifact_dir}/baseline/{dataset}/seed{seed}/variants/{run_token}/`，
  把上游 stage 的輸出（mask_* / edge_keep_*）集中在該處，避免互吃。
- Baseline 準備：若 baseline 中介物（X_norm / mask / edges …）尚未存在，會先呼叫
  `pipelines/run_baseline_grape.py --task both --inject_artifact_flags` 產出中介物。

注意：
- 預設 `--grape_script` 指向 `pipelines/run_baseline_grape.py`（請確認位置）。
- **必填 `--grape_root`**（GRAPE 專案根目錄，內含 `train_mdi.py`/`train_y.py`）。
- 預設 `--mask_op=AND`，可用 `--mask_op OR` 或環境變數 `GRAFT_MASK_OP` 覆寫。
- T2G / LUNAR 目前為 TODO 佔位；RANDOM 已能輸出 `mask_random.npy` / `edge_keep_random.npy`。
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import subprocess
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

try:
    import numpy as np
except Exception:
    np = None

# =========================
# 小工具
# =========================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(p: Path, obj: Any) -> None:
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(p: Path, default=None):
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def mtime_safe(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0

# =========================
# 契約：baseline 位置與檔案名
# =========================

@dataclass
class Contract:
    artifact_dir: Path
    dataset: str
    seed: int

    def baseline_dir(self) -> Path:
        return self.artifact_dir / "baseline" / self.dataset / f"seed{self.seed}"

    # baseline 必備檔
    def p_X(self) -> Path: return self.baseline_dir() / "X_norm.npy"
    def p_mask(self) -> Path: return self.baseline_dir() / "mask.npy"
    def p_edges(self) -> Path: return self.baseline_dir() / "bipartite_edges.npz"
    def p_split(self) -> Path: return self.baseline_dir() / "split_idx.json"
    def p_logs(self) -> Path: return self.baseline_dir() / "logs"
    def p_variants_root(self) -> Path: return self.baseline_dir() / "variants"

# =========================
# Stages
# =========================

class Stage:
    name = "base"

    def __init__(self, args: argparse.Namespace, contract: Contract, variant_dir: Path, stage_args: Dict[str, Any]):
        self.args = args
        self.contract = contract
        self.variant_dir = variant_dir
        self.stage_args = stage_args

    def input_fingerprints(self) -> List[Tuple[str, float]]:
        fps = []
        for p in [self.contract.p_X(), self.contract.p_mask(), self.contract.p_edges(), self.contract.p_split()]:
            fps.append((str(p), mtime_safe(p)))
        return fps

    def run(self) -> None:
        raise NotImplementedError

class RandomStage(Stage):
    name = "random"

    def run(self) -> None:
        if np is None:
            raise RuntimeError("Random stage 需要 numpy。請安裝 numpy 或移除此 stage。")
        # 參數
        drop_rows = float(self.stage_args.get("drop_rows", 0.0))
        drop_cols = float(self.stage_args.get("drop_cols", 0.0))
        drop_edges = float(self.stage_args.get("drop_edges", 0.0))
        rng_seed = int(self.stage_args.get("seed", self.args.seed))
        rng = np.random.default_rng(rng_seed)

        X_p, M_p, E_p = self.contract.p_X(), self.contract.p_mask(), self.contract.p_edges()
        if not X_p.exists() or not M_p.exists():
            raise FileNotFoundError(f"找不到 baseline X/mask：{X_p} / {M_p}。請先做 baseline 準備。")
        X = np.load(X_p)
        M = np.load(M_p).copy()
        n, d = X.shape
        if M.shape != (n, d):
            raise RuntimeError(f"mask 形狀不符：{M.shape} vs {(n, d)}")

        kept_rows = np.arange(n)
        kept_cols = np.arange(d)

        # 軟刪：把被抽中的 row/col 在遮罩上設為 0（不改 X 形狀）
        if drop_rows > 0:
            k = int(np.floor(n * drop_rows))
            if k > 0:
                idx = rng.choice(n, size=k, replace=False)
                M[idx, :] = 0
                kept_rows = np.setdiff1d(np.arange(n), idx, assume_unique=False)
        if drop_cols > 0:
            k = int(np.floor(d * drop_cols))
            if k > 0:
                jdx = rng.choice(d, size=k, replace=False)
                M[:, jdx] = 0
                kept_cols = np.setdiff1d(np.arange(d), jdx, assume_unique=False)

        # 保存至 variant_dir（不覆蓋 baseline）
        ensure_dir(self.variant_dir)
        np.save(self.variant_dir / "mask_random.npy", M.astype(np.uint8))

        # 邊級隨機刪除 → 生成 keep 向量（可選）
        if drop_edges > 0 and E_p.exists():
            try:
                E = np.load(E_p)
                rows, cols = E.get("rows"), E.get("cols")
                if rows is not None and cols is not None and rows.shape[0] == cols.shape[0]:
                    m = rows.shape[0]
                    k = int(np.floor(m * drop_edges))
                    if k > 0:
                        idx = rng.choice(m, size=k, replace=False)
                        keep = np.ones(m, dtype=bool)
                        keep[idx] = False
                        np.save(self.variant_dir / "edge_keep_random.npy", keep)
            except Exception as e:
                print(f"  [WARN] random 邊處理失敗：{e}")

        # 記錄 log
        log = {
            "stage": self.name,
            "params": self.stage_args,
            "n": int(n), "d": int(d),
            "kept_rows": int(kept_rows.size),
            "kept_cols": int(kept_cols.size),
            "time": now_str(),
        }
        ensure_dir(self.contract.p_logs())
        write_json(self.contract.p_logs() / "random_log.json", log)

class T2GStage(Stage):
    name = "t2g"

    def run(self) -> None:
        # TODO: 在這裡產生 t2g 的遮罩與/或邊 keep，存到 self.variant_dir
        # 例如：np.save(self.variant_dir / "mask_t2g.npy", M)
        ensure_dir(self.contract.p_logs())
        write_json(self.contract.p_logs() / "t2g_log.json", {
            "stage": self.name,
            "params": self.stage_args,
            "note": "TODO: 產生 mask_t2g.npy / edge_keep_t2g.npy 到 variants 目錄",
            "time": now_str(),
        })

class LUNARStage(Stage):
    name = "lunar"

    def run(self) -> None:
        # TODO: 在這裡產生 lunar 的遮罩與/或邊 keep，存到 self.variant_dir
        ensure_dir(self.contract.p_logs())
        write_json(self.contract.p_logs() / "lunar_log.json", {
            "stage": self.name,
            "params": self.stage_args,
            "note": "TODO: 產生 mask_lunar.npy / edge_keep_lunar.npy 到 variants 目錄",
            "time": now_str(),
        })

class GRAPEStage(Stage):
    name = "grape"

    def run(self) -> None:
        # 在 orchestrator 主流程中處理（建立 manifest 與呼叫），此處不使用
        raise RuntimeError("GRAPEStage.run 不應被直接呼叫；請使用 orchestrator 中的 grape 流程。")

STAGE_REGISTRY = {
    "random": RandomStage,
    "t2g": T2GStage,
    "lunar": LUNARStage,
    "grape": GRAPEStage,
}

# =========================
# 參數解析
# =========================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRAFT Orchestrator with strict manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 共通
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--artifact_dir", type=str, required=True)
    p.add_argument("--modules", type=str, required=True, help="Comma modules: t2g,lunar,random,grape")
    p.add_argument("--order", type=str, default=None, help="Execution order like t2g>lunar>grape; defaults to modules order")
    p.add_argument("--mask_op", type=str, default="AND", choices=["AND", "OR"], help="Mask composition op")
    p.add_argument("--cache", action="store_true", help="(Reserved) Enable cache for stages")

    # GRAPE 腳本位置（你已把它移到 pipelines/）與 GRAPE 專案根目錄
    p.add_argument("--grape_script", type=str, default="pipelines/run_baseline_grape.py",
                   help="Path to run_baseline_grape.py")
    p.add_argument("--grape_root", type=str, required=True, help="Path to GRAPE project root (contains train_mdi.py/train_y.py)")

    # Baseline 準備
    p.add_argument("--auto_prep", action="store_true", help="Auto run GRAPE to dump baseline intermediates if missing")

    args, unknown = p.parse_known_args(argv)

    def harvest_prefix(prefix: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        i = 0
        while i < len(unknown):
            tok = unknown[i]
            if not tok.startswith("--" + prefix + "."):
                i += 1
                continue
            key = tok[2 + len(prefix) + 1:].replace('-', '_')
            val: Any = True
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                val = unknown[i + 1]
                i += 1
            out[key] = val
            i += 1
        return out

    args._t2g = harvest_prefix("t2g")
    args._lunar = harvest_prefix("lunar")
    args._random = harvest_prefix("random")
    args._grape = harvest_prefix("grape")

    return args

# =========================
# Orchestrator 主流程
# =========================

def run_grape_prep(args: argparse.Namespace, contract: Contract) -> None:
    """若 baseline 中介物不存在，呼叫 GRAPE 進行完整訓練以導出中介物（透過 inject_artifact_flags）。
    嚴格 manifest 模式下，這裡會先建立一份『prep 專用 manifest』並用 ENV 傳入，
    內容僅包含 baseline 的 mask 路徑（即使檔案尚未存在，gnn 的 EXPORT 之後就會產生）。
    """
    grape = Path(args.grape_script).resolve()
    if not grape.exists():
        raise FileNotFoundError(f"找不到 GRAPE 腳本：{grape}")
    
    # 準備 prep manifest：baseline/<ds>/seedX/variants/_prep/overlay_manifest.json
    prep_dir = contract.p_variants_root() / "_prep"
    ensure_dir(prep_dir)
    prep_manifest = {
        "masks": [str(contract.p_mask())],   # 僅 baseline mask；gnn 會在 EXPORT 先把它寫出
        "edge_keeps": [],
        "order": "grape",
        "mask_op": os.getenv("GRAFT_MASK_OP", args.mask_op),
        "mode": "soft",
        "variant_dir": str(prep_dir),
        "baseline_dir": str(contract.baseline_dir()),
    }
    prep_manifest_p = prep_dir / "overlay_manifest.json"
    write_json(prep_manifest_p, prep_manifest)

    print("[PREP] Baseline intermediates not found → run GRAPE to dump via inject_artifact_flags…")
    cli = [sys.executable, str(grape),
           "--grape_root", args.grape_root,
           "--dataset", args.dataset,
           "--seed", str(args.seed),
           "--artifact_dir", str(args.artifact_dir),
           "--task", "both",
           "--inject_artifact_flags"]

    # 嚴格 manifest 必要的環境變數
    env = os.environ.copy()
    env["GRAFT_OVERLAY_MANIFEST"] = str(prep_manifest_p)
    env["GRAFT_MASK_OP"] = prep_manifest["mask_op"]

    print("  →", " ".join(cli))
    proc = subprocess.run(cli, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"GRAPE prep 失敗（returncode={proc.returncode}）")


def build_run_token(args: argparse.Namespace, contract: Contract) -> str:
    fingerprint = {
        "dataset": args.dataset,
        "seed": args.seed,
        "modules": args.modules,
        "order": args.order,
        "mask_op": args.mask_op,
        "t2g": args._t2g,
        "lunar": args._lunar,
        "random": args._random,
        "grape": args._grape,
        "inputs": [
            (str(contract.p_X()), mtime_safe(contract.p_X())),
            (str(contract.p_mask()), mtime_safe(contract.p_mask())),
            (str(contract.p_edges()), mtime_safe(contract.p_edges())),
            (str(contract.p_split()), mtime_safe(contract.p_split())),
        ],
    }
    return sha1_text(json.dumps(fingerprint, sort_keys=True))


def make_manifest(args: argparse.Namespace, contract: Contract, variant_dir: Path, modules_order: List[str]) -> Path:
    """收集 baseline 與各 stage 在 variant_dir 的遮罩/邊 keep，組合 manifest。"""
    masks: List[str] = []
    edge_keeps: List[str] = []

    # 基底 mask 一定放第一個
    base_mask = contract.p_mask()
    if not base_mask.exists():
        raise FileNotFoundError(f"找不到 baseline mask：{base_mask}。請先做 baseline 準備（或 --auto_prep）。")
    masks.append(str(base_mask))

    # 依順序收集 variants 下的產物
    mod2mask = {
        "t2g": "mask_t2g.npy",
        "lunar": "mask_lunar.npy",
        "random": "mask_random.npy",
    }
    mod2keep = {
        "t2g": "edge_keep_t2g.npy",
        "lunar": "edge_keep_lunar.npy",
        "random": "edge_keep_random.npy",
    }
    for m in modules_order:
        if m == "grape":
            continue
        mp = variant_dir / mod2mask.get(m, "__none__.npy")
        if mp.exists():
            masks.append(str(mp))
        kp = variant_dir / mod2keep.get(m, "__none__.npy")
        if kp.exists():
            edge_keeps.append(str(kp))

    manifest = {
        "masks": masks,
        "edge_keeps": edge_keeps,
        "order": ">".join(modules_order),
        "mask_op": args.mask_op,
        "mode": "soft",
        "variant_dir": str(variant_dir),
        "baseline_dir": str(contract.baseline_dir()),
    }
    manifest_p = variant_dir / "overlay_manifest.json"
    write_json(manifest_p, manifest)
    return manifest_p


def call_grape(args: argparse.Namespace, contract: Contract, manifest_p: Path) -> None:
    grape = Path(args.grape_script).resolve()
    if not grape.exists():
        raise FileNotFoundError(f"找不到 GRAPE 腳本：{grape}")

    # 構造子程序 CLI
    def to_cli(k: str, v: Any) -> List[str]:
        flag = f"--{k}"
        if isinstance(v, bool):
            return [flag] if v else []
        return [flag, str(v)]

    base_cli = [sys.executable, str(grape),
                "--grape_root", args.grape_root,
                "--dataset", args.dataset,
                "--seed", str(args.seed),
                "--artifact_dir", str(args.artifact_dir)]

    # 將 grape.* 內容展開
    extra_cli: List[str] = []
    for k, v in sorted(args._grape.items()):
        extra_cli += to_cli(k, v)

    cli = base_cli + extra_cli

    # 傳遞環境變數（嚴格 manifest）
    env = os.environ.copy()
    env["GRAFT_OVERLAY_MANIFEST"] = str(manifest_p)
    env["GRAFT_MASK_OP"] = os.getenv("GRAFT_MASK_OP", args.mask_op)

    print("  → 呼叫 GRAPE:", " ".join(cli))
    proc = subprocess.run(cli, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"GRAPE 執行失敗（returncode={proc.returncode}）")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    contract = Contract(artifact_dir=Path(args.artifact_dir).resolve(), dataset=args.dataset, seed=args.seed)
    ensure_dir(contract.baseline_dir())
    ensure_dir(contract.p_logs())
    ensure_dir(contract.p_variants_root())

    # Baseline 檢查/準備
    need_prep = not (contract.p_X().exists() and contract.p_mask().exists())
    if need_prep:
        if args.auto_prep:
            run_grape_prep(args, contract)
        else:
            raise RuntimeError("找不到 baseline X/mask，請先以 --auto_prep 跑一次或手動產出 baseline 中介物。")

    # 決定執行順序
    modules = [m.strip() for m in args.modules.split(',') if m.strip()]
    if 'grape' not in modules:
        raise RuntimeError("modules 需包含 'grape'（本 orchestrator 以呼叫 GRAPE 為終點）。")
    order = [m.strip() for m in args.order.split('>')] if args.order else modules

    # 建立 run_token 與 variant_dir
    run_token = build_run_token(args, contract)
    variant_dir = contract.p_variants_root() / run_token
    ensure_dir(variant_dir)

    # 記錄此次執行參數
    run_args = {
        "time": now_str(),
        "dataset": args.dataset,
        "seed": args.seed,
        "artifact_dir": str(args.artifact_dir),
        "modules": modules,
        "order": order,
        "mask_op": args.mask_op,
        "variant_dir": str(variant_dir),
        "prefix_args": {"t2g": args._t2g, "lunar": args._lunar, "random": args._random, "grape": args._grape},
    }
    write_json(contract.p_logs() / "run_args.json", run_args)

    # 逐 stage 執行（非 GRAPE）
    for m in order:
        if m == "grape":
            continue
        cls = STAGE_REGISTRY.get(m)
        if cls is None:
            raise KeyError(f"未知模組：{m}。可用：{sorted(STAGE_REGISTRY.keys())}")
        stage_args = {"t2g": args._t2g, "lunar": args._lunar, "random": args._random}.get(m, {})
        stage = cls(args=args, contract=contract, variant_dir=variant_dir, stage_args=stage_args)
        print(f"[RUN ] {m}: start @ {now_str()}")
        stage.run()
        print(f"[DONE] {m}: end   @ {now_str()}")

    # 構造 manifest（嚴格）並呼叫 GRAPE
    manifest_p = make_manifest(args, contract, variant_dir, order)
    print(f"[INFO] manifest 寫入：{manifest_p}")

    print(f"[RUN ] grape: start @ {now_str()}")
    call_grape(args, contract, manifest_p)
    print(f"[DONE] grape: end   @ {now_str()}")

    print(f"[ALL DONE] modules={modules} order={order} dataset={args.dataset} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
