# ===== 新增檔：pipelines/pack_support.py =====
from __future__ import annotations
from pathlib import Path
import json
import numpy as np


def build_grape_cmd(task: str, args) -> str:
    """
    回傳呼叫 GRAPE 的命令列（train_y.py 或 train_mdi.py）。
    依 args.grape_domain 切換 uci/pack 兩種 domain。
    task: 'y' (label) 或 'mdi' (impute)
    """
    assert task in {"y", "mdi"}
    prog = f"train_{'y' if task=='y' else 'mdi'}.py"
    domain = "pack" if getattr(args, "grape_domain", "uci") == "pack" else "uci"
    if domain == "uci":
        head = f"python {prog} uci --data {args.dataset}"
    else:
        if not getattr(args, "pack_root", None):
            raise RuntimeError("pack domain 需要 --pack_root 指向 baseline 目錄（含 X_norm.npy 等）。")
        head = f"python {prog} pack --root {args.pack_root} --data {args.dataset}"

    # 其餘通用參數可依你原本的 run_pipeline 接續串上（hidden、layers、epochs、seed…）
    return head


def infer_shape_from_pack(args) -> tuple[int, int]:
    """從 pack_root 讀 X_norm.npy 以取得 (N, d) 形狀。"""
    pack_root = Path(args.pack_root)
    X = np.load(pack_root / "X_norm.npy")
    return int(X.shape[0]), int(X.shape[1])


def make_random_mask(save_root: Path, args, *, mode: str = "cell", keep_ratio: float = 0.7, seed: int = 0) -> Path:
    """
    產生隨機遮罩：
      mode='cell' → MCAR（逐 cell 以 keep_ratio 保留）
      mode='col'  → 隨機保留某比例欄位（整欄保留/移除）
    會在 save_root/overlays 下輸出 mask_random.npy，並回傳路徑。
    需要能取得 (N, d)：
      - pack domain：從 args.pack_root 的 X_norm.npy 讀形狀
      - uci domain：你可改成讀你 baseline_dir 內的 X_norm.npy
    """
    overlays = Path(save_root) / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)

    if getattr(args, "grape_domain", "uci") == "pack":
        N, d = infer_shape_from_pack(args)
    else:
        raise RuntimeError("目前 random 模組預設搭配 pack domain 使用；若要支援 uci，請改為讀取 baseline_dir/X_norm.npy 的形狀。")

    rng = np.random.default_rng(int(seed))
    if mode == "cell":
        K = rng.random((N, d)) < float(keep_ratio)
    elif mode == "col":
        # 隨機保留 ⌊d*keep_ratio⌋ 個欄位，整欄全 1，其餘全 0
        k = max(1, int(np.floor(d * float(keep_ratio))))
        cols = rng.choice(np.arange(d), size=k, replace=False)
        K = np.zeros((N, d), dtype=bool)
        K[:, cols] = True
    else:
        raise ValueError("random.mode 僅支援 'cell' 或 'col'")

    out = overlays / "mask_random.npy"
    np.save(out, K.astype(np.uint8))
    return out
