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
- 預設 `--grape_script` 指向 `pipelines/run_baseline_grape.py`。
- **必填 `--grape_root`**（GRAPE 專案根目錄，內含 `train_mdi.py`/`train_y.py`）。
- 預設 `--mask_op=AND`，可用 `--mask_op OR` 或環境變數 `GRAFT_MASK_OP` 覆寫。
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

from pathlib import Path
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

try:
    from pipelines.lunar_adapter import run_lunar_stage as _run_lunar
except Exception:
    _run_lunar = None

try:
    from pipelines.t2g_adapter import run_t2g_stage as _run_t2g
except Exception:
    _run_t2g = None
import subprocess

try:
    import numpy as np
except Exception:
    np = None

# =========================
# Tool
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

# =========================================
# Contract: Baseline Location and File Name
# =========================================

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
    def p_split_json(self) -> Path: return self.baseline_dir() / "split_idx.json"
    def p_split_npz(self) -> Path:  return self.baseline_dir() / "split_idx.npz"
    def p_split_any(self) -> Path:
        return self.p_split_json() if self.p_split_json().exists() else self.p_split_npz()
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
        for p in [self.contract.p_X(), self.contract.p_mask(), self.contract.p_edges(), self.contract.p_split_any()]:
            fps.append((str(p), mtime_safe(p)))
        return fps

    def run(self) -> None:
        raise NotImplementedError

class RandomStage(Stage):
    name = "random"

    def run(self) -> None:
        if np is None:
            raise RuntimeError("Random stage needs numpy.")
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
        if _run_t2g is None:
            raise ModuleNotFoundError(
                "T2G 未就緒：偵測不到 t2g_adapter 或其依賴。請完成對應設定再使用。"
            )
        # 若上一個 stage 已匯出權重，且這裡沒給 weights_glob，就幫忙帶入
        if not self.stage_args.get("weights_glob"):
            cand = self.variant_dir / "t2g_weights"
            if cand.exists():
                self.stage_args["weights_glob"] = str(cand / "W_layer*.npy")
        _run_t2g(self.args, self.contract, self.variant_dir, self.stage_args)
        
        mask_p = self.variant_dir / "mask_t2g.npy"
        if not mask_p.exists():
            raise FileNotFoundError(
                f"[t2g] 期望的輸出不存在：{mask_p}。\n"
                f"請檢查 weights_glob 是否指向有效的 W_layer*.npy，或先跑 t2gexp。"
            )
        ensure_dir(self.contract.p_logs())
        write_json(self.contract.p_logs() / "t2g_log.json", {
            "stage": self.name,
            "params": self.stage_args,
            "time": now_str(),
        })

class LUNARStage(Stage):
    name = "lunar"

    def run(self) -> None:
        if _run_lunar is None:
            raise ModuleNotFoundError(
                "LUNAR 未就緒：偵測不到 lunar_adapter 或其依賴。\n"
                "請設定環境變數 GRAFT_LUNAR_DIR 指向含 LUNAR.py 的資料夾，"
                "或把第三方程式碼放到 <repo_root>/third_party/LUNAR。"
            )

        # 從 prefix 取用命令列參數（允許 --lunar.keep 或 --lunar.keep_ratio）
        opts = dict(self.stage_args or {})
        keep_ratio = opts.get("keep_ratio", opts.get("keep", None))
        keep_ratio = float(keep_ratio) if keep_ratio is not None else None

        # 讓輸出寫到 orchestrator 這次 run 的 variants 目錄（與其他 stage 對齊）
        variant_name = self.variant_dir.name

        # 交給 adapter 跑 LUNAR，並產出 mask_lunar.npy / edge_keep_lunar.npy / lunar_scores.npy ...
        _run_lunar(
            artifact_dir=str(self.contract.artifact_dir),
            dataset=self.contract.dataset,
            seed=int(self.contract.seed),
            variant=variant_name,
            k=(int(opts["k"]) if "k" in opts else None),
            keep_ratio=keep_ratio,
            samples=str(opts.get("samples", "MIXED")),
            val_size=float(opts.get("val_size", 0.10)),
            rescale=bool(opts.get("rescale", False)),
            train_new_model=bool(opts.get("train_new_model", True)),
        )

        # 補一個簡短 log（可選）
        ensure_dir(self.contract.p_logs())
        write_json(self.contract.p_logs() / "lunar_log.json", {
            "stage": self.name,
            "params": self.stage_args,
            "variant_dir": str(self.variant_dir),
            "time": now_str(),
        })

class GRAPEStage(Stage):
    name = "grape"

    def run(self) -> None:
        # 在 orchestrator 主流程中處理（建立 manifest 與呼叫），此處不使用
        raise RuntimeError("GRAPEStage.run 不應被直接呼叫；請使用 orchestrator 中的 grape 流程。")

class T2GExportStage(Stage):
    name = "t2gexp"
    def run(self) -> None:
        import glob
        # --- 準備輸出目錄與 log ---
        out_dir = self.variant_dir / "t2g_weights"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir = self.variant_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "t2gexp_stdout.log"

        # --- 取得並修正 repo 路徑（容錯：傳到 bin/ 或 t2g_former.py 也能自救回 repo 根）---
        repo_arg = self.stage_args.get("t2g_repo")
        if not repo_arg:
            raise ValueError("t2gexp 需要 --t2gexp.t2g_repo 指向 t2g-former 專案根目錄")
        repo = Path(str(repo_arg))
        if repo.is_file():
            # e.g. .../bin/t2g_former.py -> 專案根
            repo = repo.parent.parent
        elif repo.name == "bin":
            # e.g. .../bin/ -> 專案根
            repo = repo.parent

        # --- 組 CLI ---
        cli = [
            sys.executable, "-u",  # 不緩衝
            "pipelines/t2g_export_from_grape.py",
            "--t2g_repo", str(repo),
            "--baseline_dir", str(self.contract.baseline_dir()),
            "--variant_dir", str(self.variant_dir),
            "--output", str(out_dir),
        ]
        # 轉傳可選參數
        passthrough_keys = [
            "epochs", "lr", "batch_size", "max_batches", "train_parts", "train_on",
            "head_reduce", "batch_reduce", "module_regex", "device", "task",
            "norm_per_layer", "print_modules",
        ]
        for k in passthrough_keys:
            if k not in self.stage_args:
                continue
            v = self.stage_args[k]
            if isinstance(v, bool):
                if v:  # 只在 True 時加入旗標
                    cli += [f"--{k}"]
            elif v is not None and str(v) != "":
                cli += [f"--{k}", str(v)]

        print("  → export T2G weights:", " ".join(cli))

        # --- 執行（即時顯示＋寫 variants 專屬 log）---
        env = dict(os.environ, PYTHONUNBUFFERED="1")  # 子程式不緩衝
        with subprocess.Popen(
            cli,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,              # 行緩衝
            env=env,
        ) as p, open(log_file, "w", encoding="utf-8") as f:
            assert p.stdout is not None
            for line in p.stdout:
                print(f"[t2gexp] {line}", end="")  # 顯示到父行程（會被你外層 > 或 tee 接走）
                f.write(line)
            ret = p.wait()

        if ret != 0:
            raise RuntimeError(f"t2gexp 匯權重失敗（code={ret}），詳見 {log_file}")

        # --- 基本產物檢查：至少要有一個 W_layer*.npy ---
        w_files = sorted(glob.glob(str(out_dir / "W_layer*.npy")))
        if not w_files:
            raise RuntimeError(
                f"t2gexp 看起來沒有輸出層權重（{out_dir} 下無 W_layer*.npy）。請檢查 {log_file}"
            )
        print(f"[t2gexp] exported {len(w_files)} layer matrices → {out_dir}")
        print(f"[t2gexp] log saved → {log_file}")

STAGE_REGISTRY = {
    "t2gexp": T2GExportStage,
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
    p.add_argument("--auto_prep", action="store_true", help="Auto build baseline intermediates if missing (prep-only)")
    p.add_argument("--prep_only", action="store_true", help="Only build baseline intermediates and exit (no final GRAPE training)")
    p.add_argument("--force_prep", action="store_true", help="Rebuild baseline intermediates even if files already exist")

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

    args._t2gexp = harvest_prefix("t2gexp")
    args._t2g = harvest_prefix("t2g")
    args._lunar = harvest_prefix("lunar")
    args._random = harvest_prefix("random")
    args._grape = harvest_prefix("grape")

    return args

# =========================
# Orchestrator 主流程
# =========================

def run_grape_prep(args: argparse.Namespace, contract: Contract) -> None:
    """Prep-only: export GRAPE baseline intermediates, do NOT train.
    It writes variants/_prep/overlay_manifest.json and sets GRAPT_PREP_ONLY=1 for the child process.
    """
    grape = Path(args.grape_script).resolve()
    if not grape.exists():
        raise FileNotFoundError(f"找不到 GRAPE 腳本：{grape}")

    # 準備 prep manifest：baseline/<ds>/seedX/variants/_prep/overlay_manifest.json
    prep_dir = contract.p_variants_root() / "_prep"
    ensure_dir(prep_dir)
    prep_manifest = {
        "masks": [str(contract.p_mask())],
        "edge_keeps": [],
        "order": "grape",
        "mask_op": os.getenv("GRAFT_MASK_OP", "AND"),
        "mode": "soft",
        "variant_dir": str(prep_dir),
        "baseline_dir": str(contract.baseline_dir()),
    }
    prep_manifest_p = prep_dir / "overlay_manifest.json"
    write_json(prep_manifest_p, prep_manifest)

    print("[PREP] Baseline intermediates not found → export via GRAPE (prep-only)…")
    cli = [sys.executable, str(grape),
           "--grape_root", args.grape_root,
           "--dataset", args.dataset,
           "--seed", str(args.seed),
           "--artifact_dir", str(args.artifact_dir),
           "--task", "both",
           "--prep_only",
           "--inject_artifact_flags"]

    # 嚴格 manifest + PREP_ONLY
    env = os.environ.copy()
    env["GRAFT_OVERLAY_MANIFEST"] = str(prep_manifest_p)
    env["GRAFT_MASK_OP"] = prep_manifest["mask_op"]
    env["GRAPT_PREP_ONLY"] = "1"

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
            (str(contract.p_split_any()), mtime_safe(contract.p_split_any())),
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
    need_prep = (
        args.force_prep or
        not (
            contract.p_X().exists() and contract.p_mask().exists() and
            contract.p_edges().exists() and contract.p_split_any().exists()
        )
    )
    if need_prep:
        if args.auto_prep:
            run_grape_prep(args, contract)
        else:
            raise RuntimeError("找不到完整 baseline 中介物（X/mask/edges/split）。請使用 --auto_prep 或先手動產出。")
        
    # Optional early exit (prep-only mode)
    if args.prep_only:
        print("[EXIT] prep_only=True → 已產 baseline 中介物，未執行最終 GRAPE 訓練。")
        return 0

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
        stage_args = {
            "t2gexp": args._t2gexp, "t2g": args._t2g,
            "lunar": args._lunar, "random": args._random
        }.get(m, {})
        stage = cls(args=args, contract=contract, variant_dir=variant_dir, stage_args=stage_args)
        print(f"[RUN ] {m}: start @ {now_str()}")
        stage.run()
        print(f"[DONE] {m}: end   @ {now_str()}")

    # def preflight_masks(contract: Contract, manifest_p: Path, strict: bool = False) -> None:
    #     """檢查 manifest 指到的 masks/edge_keeps 是否存在、形狀與 dtype 是否合理。
    #     - strict=False：只警告不擋流程
    #     - strict=True ：遇到問題直接 raise（一般不建議）
    #     """
    #     def _warn(msg: str):
    #         print(f"[warn preflight] {msg}")

    #     try:
    #         obj = read_json(manifest_p)
    #         if not obj:
    #             raise RuntimeError("manifest 解析失敗")
    #         masks = obj.get("masks", [])
    #         keeps = obj.get("edge_keeps", [])
    #         mask_op = obj.get("mask_op", None)
    #         order = obj.get("order", None)

    #         # 基本欄位
    #         if not masks:
    #             raise RuntimeError("manifest.masks 為空")
    #         if mask_op not in ("AND", "OR"):
    #             raise RuntimeError("manifest.mask_op 必須是 AND/OR")
    #         if not order or "grape" not in order:
    #             raise RuntimeError("manifest.order 缺少 'grape'")

    #         # 讀 baseline X 的形狀
    #         import numpy as np
    #         Xp = contract.p_X()
    #         if not Xp.exists():
    #             raise RuntimeError(f"缺少 baseline X：{Xp}")
    #         X = np.load(Xp, mmap_mode="r")
    #         n, d = X.shape

    #         # 檢查每個 mask
    #         for mp in masks:
    #             mp = Path(mp)
    #             if not mp.exists():
    #                 msg = f"mask 不存在：{mp}"
    #                 if strict: raise RuntimeError(msg)
    #                 _warn(msg); continue
    #             M = np.load(mp, mmap_mode="r")
    #             if M.shape != (n, d):
    #                 msg = f"mask 形狀不符：{mp.name} {M.shape} ≠ {(n, d)}"
    #                 if strict: raise RuntimeError(msg)
    #                 _warn(msg)
    #             if M.dtype not in (np.bool_, np.uint8, np.int8, np.int32, np.int64):
    #                 _warn(f"mask dtype 非 bool/uint8：{mp.name} ({M.dtype})")
    #             elif M.dtype == np.uint8:
    #                 _warn(f"mask dtype uint8（預期 bool）→ 可接受 0/1，將於訓練時視為布林：{mp.name}")

    #         # edge_keep 長度（如果有）
    #         Ep = contract.p_edges()
    #         if keeps and Ep.exists():
    #             try:
    #                 E = np.load(Ep)
    #                 rows, cols = E.get("rows"), E.get("cols")
    #                 m = rows.shape[0] if (rows is not None and cols is not None) else None
    #             except Exception:
    #                 m = None
    #             if m is None:
    #                 _warn("無法讀取 bipartite_edges.npz 的 rows/cols，略過 keep 檢查")
    #             else:
    #                 for kp in keeps:
    #                     kp = Path(kp)
    #                     if not kp.exists():
    #                         msg = f"edge_keep 不存在：{kp}"
    #                         if strict: raise RuntimeError(msg)
    #                         _warn(msg); continue
    #                     keep = np.load(kp, mmap_mode="r")
    #                     if keep.shape[0] != m:
    #                         msg = f"edge_keep 長度不符：{kp.name} {keep.shape[0]} ≠ {m}"
    #                         if strict: raise RuntimeError(msg)
    #                         _warn(msg)
    #     except Exception as e:
    #         if strict:
    #             raise
    #         print(f"[warn preflight] 檢查時發生例外：{e}（已放行）")

    # run_pipeline.py（在 make_manifest 前，加一段嚴格檢查）
    expected = {"lunar": "mask_lunar.npy", "t2g": "mask_t2g.npy", "random": "mask_random.npy"}
    missing = []
    for m in order:
        if m == "grape": continue
        exp = expected.get(m)
        if exp and not (variant_dir / exp).exists():
            missing.append(f"{m}:{exp}")
    if missing:
        raise FileNotFoundError(f"Stage outputs missing → {missing}；請檢查 {variant_dir} 內的產物或 stage 執行紀錄")

    # 構造 manifest（嚴格）並呼叫 GRAPE
    manifest_p = make_manifest(args, contract, variant_dir, order)
    # preflight_masks(contract, manifest_p)
    # Optional early exit (只準備中介物就離開)
    if args.prep_only:
        print("[EXIT] prep_only=True → 已產 baseline 中介物，未執行最終 GRAPE 訓練。")
        return 0
    print(f"[INFO] manifest 寫入：{manifest_p}")

    print(f"[RUN ] grape: start @ {now_str()}")
    call_grape(args, contract, manifest_p)
    print(f"[DONE] grape: end   @ {now_str()}")

    print(f"[ALL DONE] modules={modules} order={order} dataset={args.dataset} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
