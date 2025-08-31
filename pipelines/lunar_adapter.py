# GRAFT/pipelines/lunar_adapter.py
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import os


# ===== 你提供的預設（可依需要調整）=====
DEFAULTS = {
    "yacht":   {"k": 20,  "keep": 0.95},
    "housing": {"k": 25,  "keep": 0.95},
    "energy":  {"k": 30,  "keep": 0.90},
    "concrete":{"k": 35,  "keep": 0.90},
    "wine":    {"k": 40,  "keep": 0.85},
    "kin8nm":  {"k": 100, "keep": 0.85},
    "power":   {"k": 100, "keep": 0.85},
    "naval":   {"k": 100, "keep": 0.85},
    "protein": {"k": 100, "keep": 0.80},
}

# ===== 把 third_party/LUNAR 加進 import path =====
def _ensure_lunar_on_path() -> bool:
    lunar_dir = os.getenv("GRAFT_LUNAR_DIR") or str(Path(__file__).resolve().parents[1] / "third_party" / "LUNAR")
    if lunar_dir and os.path.isdir(lunar_dir) and lunar_dir not in sys.path:
        sys.path.insert(0, lunar_dir)
    try:
        import LUNAR  # noqa: F401
        return True
    except Exception:
        return False

_HAS_LUNAR = _ensure_lunar_on_path()

# ===== 小工具 =====
def _round_to_5(x: int) -> int: return int(round(x / 5) * 5)

def _auto_k(n_train: int, val_size: float) -> int:
    n_eff = max(1, int(n_train * (1.0 - val_size)))
    k0 = int(np.clip(_round_to_5(int(np.sqrt(max(1, n_train)))), 15, 100))
    return min(k0, max(1, n_eff - 1))

def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

# ===== 主要 API：從 GRAPE 中介檔跑 LUNAR，輸出遮罩與 manifest =====
def run_lunar_stage(
    *,
    artifact_dir: str,
    dataset: str,
    seed: int,
    variant: str | None = None,
    k: int | None = None,                 # 若未指定，用 DEFAULTS 或自動
    keep_ratio: float | None = None,      # 若未指定，用 DEFAULTS
    val_size: float = 0.10,               # LUNAR 內部 train/val 切分
    samples: str = "MIXED",               # 'UNIFORM'/'SUBSPACE'/'MIXED'
    rescale: bool = False,                # X_norm 多半已正規化，預設 False
    train_new_model: bool = True          # 每次都重訓
) -> dict:
    if not _HAS_LUNAR:
        raise ModuleNotFoundError(
            "LUNAR 未就緒：請設定 GRAFT_LUNAR_DIR 或放到 third_party/LUNAR；只有 --modules 包含 lunar 時才需要。"
        )
    import LUNAR as lunar_impl
    base = Path(artifact_dir) / "baseline" / dataset / f"seed{seed}"
    X_p, split_p = base / "X_norm.npy", base / "split_idx.json"
    assert X_p.exists() and split_p.exists(), f"缺檔案：{X_p} 或 {split_p}"

    # 讀資料與 split
    X = np.load(X_p).astype("float32")  # (n, d)
    n, d = X.shape
    split = json.loads(split_p.read_text())
    train_idx = np.array(split.get("train", []), dtype=np.int64)
    if train_idx.size == 0:  # 沒有 row 級 split（例如純 impute）
        train_idx = np.arange(n, dtype=np.int64)

    # 取預設 k / keep（可被 CLI 覆蓋）
    ds_def = DEFAULTS.get(dataset, {})
    if k is None or k <= 0:
        k = int(ds_def.get("k", _auto_k(len(train_idx), val_size)))
    if keep_ratio is None:
        kr = ds_def.get("keep", None)
        if kr is None:
            raise ValueError(
                f"未提供 keep_ratio，且 DEFAULTS 沒有此資料集（{dataset}）的 keep 值"
            )
        keep_ratio = float(kr)
    else:
        keep_ratio = float(keep_ratio)

    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio 不在 (0,1]：{keep_ratio}")

    # 切 train/val（只在訓練 rows 上切）
    rng = np.random.default_rng(int(seed))
    n_val = max(1, int(round(val_size * train_idx.size)))
    sel_val = rng.choice(train_idx.size, size=n_val, replace=False)
    sel_tr  = np.setdiff1d(np.arange(train_idx.size), sel_val)
    tr_idx, va_idx = train_idx[sel_tr], train_idx[sel_val]

    # 選擇是否再做額外縮放（通常不需要）
    X_used = X
    if rescale:
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler().fit(X_used[tr_idx])
        X_used = mm.transform(X_used).astype("float32")

    # 組 LUNAR 的 train/val/test（test 放「全 rows」好回傳完整分數）
    train_x = X_used[tr_idx]; train_y = np.zeros(len(train_x), dtype="float32")
    val_x   = X_used[va_idx]; val_y   = np.zeros(len(val_x), dtype="float32")
    test_x  = X_used;         test_y  = np.zeros(len(test_x), dtype="float32")

    # 執行 LUNAR，回傳 test 段異常分數（越大越異常）
    scores = lunar_impl.run(
        train_x, train_y, val_x, val_y, test_x, test_y,
        dataset="GRAPE", seed=int(seed), k=int(k),
        samples=samples, train_new_model=train_new_model
    ).numpy().astype("float32")

    # 以 keep_ratio 形成 row 保留（分位數；分數愈小愈正常）
    thr = float(np.quantile(scores, keep_ratio))
    keep_rows = scores <= thr
    row_keep_idx = np.where(keep_rows)[0].astype(np.int64)

    # variant 目錄
    if variant is None:
        variant = f"lunar_r{int(round(keep_ratio * 100))}"
    var_dir = base / "variants" / variant
    var_dir.mkdir(parents=True, exist_ok=True)

    # 產物：scores / row_keep / mask
    np.save(var_dir / "lunar_scores.npy", scores)
    np.save(var_dir / "row_keep_idx.npy", row_keep_idx)

    mask = np.ones_like(X, dtype=bool)
    mask[~keep_rows, :] = False
    np.save(var_dir / "mask_lunar.npy", mask)

    # 可選：若存在二部圖邊檔，順帶產 edge_keep（簡單實用）
    edges_p = base / "bipartite_edges.npz"
    edge_keep_path = None

    # overlay manifest：GRAPE 端會 AND 疊這張 mask（與邊）
    mani = {
        "op": "AND",
        "masks": [str(var_dir / "mask_lunar.npy")],
        "edge_keeps": []   # << 先空陣列
    }
    mani_p = var_dir / "overlay_manifest.json"
    _write_json(mani_p, mani)

    # 簡易 log
    _write_json(var_dir / "lunar_log.json", {
        "n": int(n), "d": int(d),
        "k_used": int(k),
        "keep": int(keep_rows.sum()),
        "keep_ratio": float(keep_rows.mean()),
        "threshold(score@quantile)": thr,
        "manifest": str(mani_p)
    })

    # ========== Development Space ==========
    # TODO: 支援 keep_ratio 網格掃描，讀回 GRAPE valid MAE 選最佳
    # TODO: 支援 score τ 直接下限（非分位數）
    # TODO: 支援 column 級（X.T 再跑 LUNAR），產 col_keep 與特徵遮罩

    # === 建立固定入口捷徑：variants/_latest_lunar/*
    import os, shutil
    latest = base / "variants" / "_latest_lunar"
    latest.mkdir(parents=True, exist_ok=True)

    def _link_or_copy(src, dst):
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)  # Colab/Linux OK；雲端硬碟有時不支援
        except Exception:
            shutil.copy2(src, dst)

    # 需要暴露給「固定入口」的檔案
    for fn in ["mask_lunar.npy", "lunar_scores.npy", "row_keep_idx.npy", "overlay_manifest.json"]:
        _link_or_copy(var_dir / fn, latest / fn)

    return {
        "variant_dir": str(var_dir),
        "manifest": str(mani_p),
        "k_used": int(k),
        "keep_ratio": float(keep_rows.mean()),
    }
