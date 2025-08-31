# GRAFT/pipelines/t2g_adapter.py
from __future__ import annotations
import os, json, glob
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

"""
T2G Adapter (soft feature-prune → mask_t2g.npy)
-----------------------------------------------
設計目標：
- 只取 T2G-Former 中「對特徵的重要度/關聯度打分」這一段，輸出一張列為 rows、欄為 cols 的遮罩，
  讓 orchestrator 以 AND/OR 疊上 GRAPE baseline 的 mask，交由 GRAPE 訓練。
- GE/FR-Graph 權重矩陣 (layers) → 以加權度數/中心性做打分

輸出：
  variants/<run_token>/
    ├─ mask_t2g.npy           # (n, d) 的 uint8/bool；被移除的欄整欄為 0
    ├─ edge_keep_t2g.npy      # (E,) 的 bool（可選；若 bipartite_edges.npz 存在則產生）
    └─ t2g_log.json           # 記錄參數、保留欄數/比例、分數統計

慣例參數（由 run_pipeline.py 的 --t2g.* 前綴傳入 stage_args）：
  keep_cols_ratio: float in (0,1]，要保留的欄位比例（與 k 擇一；兩者皆給時以 k 為主）
  k:               int，要保留的欄位數
  scores_npy:      str，指向一個 (d,) 的 .npy 分數檔（C/B）
  weights_npy:     str，指向單一 (d,d) 權重矩陣（A）
  weights_glob:    str，glob pattern 尋找多層 (d,d) 權重矩陣（A）
  layer_weights:   str，逗號分隔的每層權重，例如 "1,1,0.5"（與 weights_glob 對應）
  abs_weights:     bool，是否取絕對值（預設 True）
  symmetrize:      bool，是否做 0.5*(W + W^T) 後再打分（預設 True）
  degree_mode:     str，"both"|"in"|"out"（預設 "both"）—用來計算加權度數
  fallback_mode:   str，"var"|"std"|"l1"（預設 "var"）—若沒有任何權重或分數檔時的備援打分
  coverage_power:  float，將 baseline mask 的欄覆蓋率 (0~1) 以 p 次方乘到分數上（預設 1.0；0 表示不使用覆蓋率）

注意：
- 本檔不重建圖、不物理刪欄；僅輸出列×欄的遮罩（軟刪）。
- 若 bipartite_edges.npz 存在，會順帶輸出 edge_keep_t2g.npy（以「保留欄」決定邊是否保留）。
"""


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    _ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_optional_npy(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"指定的檔案不存在：{p}")
    return np.load(p)


def _collect_layer_weights(glob_pattern: str) -> List[np.ndarray]:
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"找不到任何權重檔（weights_glob）：{glob_pattern}")
    mats = []
    for fp in files:
        M = np.load(fp)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"權重矩陣形狀不正確：{fp} 形狀={M.shape}，預期為 (d,d)")
        mats.append(M.astype(np.float64, copy=False))
    return mats


def _degree_scores(W: np.ndarray, mode: str = "both") -> np.ndarray:
    """從單層 (d,d) 權重矩陣計算每個節點的（加權）度數。"""
    if mode == "in":
        return np.asarray(W).sum(axis=0)
    if mode == "out":
        return np.asarray(W).sum(axis=1)
    # both = in + out
    return np.asarray(W).sum(axis=0) + np.asarray(W).sum(axis=1)


def _combine_layer_scores(mats: List[np.ndarray],
                          layer_weights: Optional[List[float]] = None,
                          abs_weights: bool = True,
                          symmetrize: bool = True,
                          mode: str = "both") -> np.ndarray:
    d = mats[0].shape[0]
    if any(M.shape != (d, d) for M in mats):
        raise ValueError("多層權重矩陣的形狀不一致。")
    if layer_weights is None:
        layer_weights = [1.0] * len(mats)
    if len(layer_weights) != len(mats):
        raise ValueError("layer_weights 的長度需與權重矩陣層數一致。")

    total = np.zeros(d, dtype=np.float64)
    for w, M in zip(layer_weights, mats):
        A = M
        if abs_weights:
            A = np.abs(A)
        if symmetrize:
            A = 0.5 * (A + A.T)
        total += float(w) * _degree_scores(A, mode=mode)
    return total


def _fallback_scores(X: np.ndarray, mask: np.ndarray, mode: str = "var", coverage_power: float = 1.0) -> np.ndarray:
    # X: (n, d), mask: (n, d) — 這裡的 mask 是 baseline 的可見性（用來估 coverage）
    n, d = X.shape
    cov = mask.mean(axis=0).astype(np.float64)  # 每個欄的覆蓋率（0~1）
    if mode == "std":
        s = X.std(axis=0, ddof=1)
    elif mode == "l1":
        s = np.abs(X).mean(axis=0)
    else:  # "var"
        s = X.var(axis=0, ddof=1)
    if coverage_power and coverage_power != 0.0:
        s = s * np.power(cov, float(coverage_power))
    return s.astype(np.float64, copy=False)


def _select_topk(scores: np.ndarray, k: Optional[int] = None, keep_ratio: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    d = scores.shape[0]
    if (k is None or k <= 0) and (keep_ratio is None or keep_ratio <= 0 or keep_ratio > 1):
        keep_ratio = 1.0
    if k is None or k <= 0:
        k = max(1, int(round(d * float(keep_ratio))))
    k = max(1, min(d, int(k)))
    # 取分數最高的 k 個欄位
    order = np.argsort(-scores, kind="mergesort")  # stable
    kept = order[:k]
    removed = order[k:]
    return kept, removed


def run_t2g_stage(args, contract, variant_dir: Path, stage_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrator 的入口。讀 baseline 的 X/mask，依 stage_args 打分選欄，輸出 mask_t2g.npy。
    """
    base = contract.baseline_dir()
    X_p = base / "X_norm.npy"
    M_p = base / "mask.npy"
    E_p = base / "bipartite_edges.npz"
    if not X_p.exists() or not M_p.exists():
        raise FileNotFoundError(f"找不到 baseline X/mask：{X_p} / {M_p} — 請先以 --auto_prep 產生中介物。")
    X = np.load(X_p)            # (n, d)
    M0 = np.load(M_p).astype(np.uint8)  # (n, d)
    n, d = X.shape
    if M0.shape != (n, d):
        raise RuntimeError(f"baseline mask 形狀不符：{M0.shape} vs {(n, d)}")

    # ---- 讀取 CLI 參數 ----
    keep_cols_ratio = stage_args.get("keep_cols_ratio", stage_args.get("keep", None))
    keep_cols_ratio = None if keep_cols_ratio is None else float(keep_cols_ratio)
    k = stage_args.get("k", None)
    k = None if k is None else int(k)

    # 權重/分數來源（優先序：scores_npy > weights_glob/weights_npy > fallback）
    scores_npy = stage_args.get("scores_npy", None)
    weights_glob = stage_args.get("weights_glob", None)
    weights_npy = stage_args.get("weights_npy", None)
    layer_weights_str = stage_args.get("layer_weights", None)
    layer_weights = None
    if layer_weights_str:
        layer_weights = [float(x) for x in str(layer_weights_str).split(",") if x.strip() != ""]

    abs_weights = bool(stage_args.get("abs_weights", True))
    symmetrize = bool(stage_args.get("symmetrize", True))
    degree_mode = str(stage_args.get("degree_mode", "both")).lower()
    fallback_mode = str(stage_args.get("fallback_mode", "var")).lower()
    coverage_power = float(stage_args.get("coverage_power", 1.0))

    # ---- 計算 scores ----
    # 只允許 FR-Graph
    if scores_npy is not None:
        raise ValueError("T2G only supports FR-Graph weights. Remove --t2g.scores_npy.")
    if (weights_glob is None) and (weights_npy is None):
        raise RuntimeError("FR-Graph required: set --t2g.weights_glob or --t2g.weights_npy.")

    # 收集矩陣 → 檢查形狀 → 合成分數
    mats = []
    if weights_glob is not None:
        mats.extend(_collect_layer_weights(str(weights_glob)))  # 會檢查 (d,d) 並報錯 :contentReference[oaicite:11]{index=11}
    if weights_npy is not None:
        W = _load_optional_npy(str(weights_npy))
        if W is None or W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"weights_npy 需為 (d,d) 矩陣：{weights_npy} 實際 {None if W is None else W.shape}")  # :contentReference[oaicite:12]{index=12}
        mats.append(W.astype(np.float64, copy=False))
    if not mats:
        raise RuntimeError("未能載入任何權重矩陣。")  # :contentReference[oaicite:13]{index=13}

    scores = _combine_layer_scores(
        mats,
        layer_weights=layer_weights,   # 若未給，預設全 1.0  :contentReference[oaicite:14]{index=14}
        abs_weights=abs_weights,
        symmetrize=symmetrize,
        mode=degree_mode               # "in" | "out" | "both"   :contentReference[oaicite:15]{index=15}
    )
    src = "graph_weights"

    # ---- 依分數選欄 ----
    kept_cols, removed_cols = _select_topk(scores, k=k, keep_ratio=keep_cols_ratio)
    keep_mask = np.ones((n, d), dtype=np.uint8)   # T2G 的軟遮罩：保留欄=1，被移除欄整欄設 0
    if removed_cols.size > 0:
        keep_mask[:, removed_cols] = 0

    # ---- 若有邊檔，產生 edge_keep（保留的邊其 column 必須在 kept_cols 之中）----
    edge_keep_path = None
    if E_p.exists():
        try:
            E = np.load(E_p, allow_pickle=True)
            rows = E["rows"]
            cols = E["cols"]
            # rows/cols 是二部圖端點索引；cols ∈ [0, d)
            kept_flag = np.zeros(d, dtype=bool)
            kept_flag[kept_cols] = True
            edge_keep = kept_flag[cols.astype(int)]
            edge_keep_path = variant_dir / "edge_keep_t2g.npy"
            np.save(edge_keep_path, edge_keep.astype(np.bool_))
        except Exception as e:
            # 不阻斷流程；只在 log 中記錄
            edge_keep_path = None

    # ---- 寫檔 ----
    mask_path = variant_dir / "mask_t2g.npy"
    np.save(mask_path, keep_mask)

    log = {
        "source": src,
        "keep_cols_ratio_req": keep_cols_ratio,
        "k_req": k,
        "k_eff": int(kept_cols.size),
        "n_cols": int(d),
        "kept_cols": kept_cols.tolist(),
        "removed_cols": removed_cols.tolist() if removed_cols.size <= 2048 else f"{removed_cols.size} indices (omitted)",
        "scores_stat": {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "q10": float(np.quantile(scores, 0.10)),
            "q90": float(np.quantile(scores, 0.90)),
        },
        "abs_weights": abs_weights,
        "symmetrize": symmetrize,
        "degree_mode": degree_mode,
        "fallback_mode": fallback_mode,
        "coverage_power": coverage_power,
        "files": {
            "mask_t2g": str(mask_path),
            "edge_keep_t2g": (str(edge_keep_path) if edge_keep_path else None),
        },
    }
    _write_json(variant_dir / "t2g_log.json", log)

    # 提供固定入口（方便你在 Notebook/Colab 快速取用最新一次的 t2g 輸出）
    latest = base / "variants" / "_latest_t2g"
    _ensure_dir(latest)
    def _link_or_copy(src: Path, dst: Path) -> None:
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
        except Exception:
            import shutil
            shutil.copy2(src, dst)

    for fn in ["mask_t2g.npy", "t2g_log.json", "edge_keep_t2g.npy"]:
        p = variant_dir / fn
        if p.exists():
            _link_or_copy(p, latest / fn)

    return {
        "variant_dir": str(variant_dir),
        "mask_t2g": str(mask_path),
        "edge_keep_t2g": (str(edge_keep_path) if edge_keep_path else None),
        "k_eff": int(kept_cols.size),
    }
