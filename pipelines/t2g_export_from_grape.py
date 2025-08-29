#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
以 GRAPE 的中介物 (baseline_dir) 為資料來源，
在 *不洩漏測試集* 的前提下，跑 T2G-Former（可選擇是否實際訓練），
並以 forward-hook 匯出每層的 FR-Graph 權重矩陣 W_layer*.npy。

常見用法（做為 run_pipeline 的一個 stage：t2gexp）：
  python pipelines/t2g_export_from_grape.py \
    --t2g_repo /content/t2g-former \
    --baseline_dir /content/grapt_artifacts/baseline/housing/seed0 \
    --variant_dir  /content/grapt_artifacts/baseline/housing/seed0/variants/RUN123 \
    --output       /content/grapt_artifacts/baseline/housing/seed0/variants/RUN123/t2g_weights \
    --train_parts train --epochs 0 --max_batches 0

預設不做訓練（epochs=0 → 只 forward），可用 --epochs > 0 啟動簡單的監督訓練（MSE / Adam）。
若存在 LUNAR 的遮罩 (mask_lunar.npy) 或其它遮罩，且 --train_on overlay，
則 train split 只使用 overlay 後仍保留的 rows。
"""

import os, re, sys, json, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# 參數
# -------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--t2g_repo', type=str, required=True, help='t2g-former 專案路徑 (加入 PYTHONPATH)')
    p.add_argument('--baseline_dir', type=str, required=True, help='GRAPE baseline 目錄（含 X_norm.npy, mask.npy, split_idx.npz 等）')
    p.add_argument('--variant_dir', type=str, required=True, help='本次 variants 目錄；可從這裡讀取 LUNAR 等遮罩')
    p.add_argument('--output', type=str, required=True, help='輸出的 W_layer*.npy 目錄')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs', type=int, default=0, help='>0 時進行簡單監督訓練；=0 只 forward 匯權重')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--max_batches', type=int, default=0, help='>0 時每個 split 僅跑前 N 個 batch（for 快速匯出）')
    p.add_argument('--train_parts', type=str, default='train', help='逗號分隔：train,val；匯權重時會將這些 split 都跑過一遍')
    p.add_argument('--head_reduce', type=str, choices=['mean','sum','max'], default='mean')
    p.add_argument('--batch_reduce', type=str, choices=['sum','mean'], default='sum')
    p.add_argument('--abs', dest='do_abs', action='store_true', default=True)
    p.add_argument('--no-abs', dest='do_abs', action='store_false')
    p.add_argument('--sym', dest='do_sym', action='store_true', default=True)
    p.add_argument('--no-sym', dest='do_sym', action='store_false')
    p.add_argument('--norm_per_layer', action='store_true', help='每層以 Frobenius 範數正規化 W')
    p.add_argument('--module_regex', type=str, default=r'(GE|Graph|Attention)')
    p.add_argument('--print_modules', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--task', type=str, default='reg', choices=['reg','clf'], help='任務型態（回歸/分類），影響 loss 與輸出')
    p.add_argument('--train_on', type=str, default='baseline', choices=['baseline','overlay'], help='train rows 來源：baseline split 或 套用 variants 遮罩後')
    p.add_argument('--accept_plus1', action='store_true', default=True, help='允許抓到 (d+1)x(d+1) 的方陣，並去除 CLS 一列一行')
    p.add_argument('--cls_pos', type=str, choices=['first','last'], default='last', help='當矩陣是 (d+1)x(d+1) 時，CLS 的位置（預設 last）')

    return p.parse_args()

# -------------------------
# Util
# -------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_repo_to_path(repo: Path):
    sys.path.insert(0, str(repo.resolve()))


def load_baseline_artifacts(base: Path) -> Dict[str, Any]:
    X = np.load(base / 'X_norm.npy')           # (n, d)
    M = np.load(base / 'mask.npy')             # (n, d)

    # y（可選）
    y = None
    for cand in ['y.npy', 'label.npy']:
        p = base / cand
        if p.exists():
            y = np.load(p)
            break

    # splits：支援 .npz 或 .json
    splits: Dict[str, np.ndarray] = {}
    npz_p = base / 'split_idx.npz'
    json_p = base / 'split_idx.json'
    if npz_p.exists():
        z = np.load(npz_p)
        for k in ['train_idx','val_idx','test_idx']:
            if k in z.files:
                splits[k] = z[k].astype(np.int64)
    elif json_p.exists():
        import json
        with open(json_p, 'r', encoding='utf-8') as f:
            j = json.load(f)
        def _get(keys):
            # 允許平鋪或包在 indices 裡
            for k in keys:
                if k in j and isinstance(j[k], list):
                    return np.asarray(j[k], dtype=np.int64)
            if 'indices' in j and isinstance(j['indices'], dict):
                for k in keys:
                    v = j['indices'].get(k)
                    if isinstance(v, list):
                        return np.asarray(v, dtype=np.int64)
            return None
        splits['train_idx'] = _get(['train_idx','train','train_indices'])
        splits['val_idx']   = _get(['val_idx','val','valid','valid_idx'])
        splits['test_idx']  = _get(['test_idx','test'])
        # 轉成空陣列以免後面 None
        for k in ['train_idx','val_idx','test_idx']:
            if splits.get(k) is None:
                splits[k] = np.arange(0, dtype=np.int64)
    else:
        # 最後備援：單檔 .npy
        for k in ['train_idx','val_idx','test_idx']:
            p = base / f'{k}.npy'
            if p.exists():
                splits[k] = np.load(p).astype(np.int64)

    if 'train_idx' not in splits or splits['train_idx'].size == 0:
        raise FileNotFoundError('找不到 train_idx，請確認 baseline_dir 有 split_idx.[npz|json] 或 train_idx.npy')

    return {'X': X, 'M': M, 'y': y, 'splits': splits}

def derive_overlay_row_keep(base_M: np.ndarray, variant_dir: Path) -> Optional[np.ndarray]:
    """以 AND 疊合 baseline mask 與 variants 中已存在的遮罩（不含 t2g 自己），
    以 .any(axis=1) 推導 row 是否保留。若找不到任何遮罩，回傳 None。
    """
    masks = [base_M.astype(bool)]
    for p in variant_dir.glob('mask_*.npy'):
        if p.name == 'mask_t2g.npy':
            continue
        try:
            masks.append(np.load(p).astype(bool))
        except Exception:
            pass
    if len(masks) <= 1:
        return None
    final = masks[0]
    for mk in masks[1:]:
        final = final & mk
    return final.any(axis=1)


def _iter_tensors(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, (list, tuple)):
        for xi in x:
            yield from _iter_tensors(xi)
    elif isinstance(x, dict):
        for xi in x.values():
            yield from _iter_tensors(xi)


def _is_square(t: torch.Tensor) -> bool:
    return torch.is_tensor(t) and (t.ndim >= 2) and (t.shape[-1] == t.shape[-2])

def _pick_square_cand(tensors, d: int, accept_plus1: bool, cls_pos: str):
    """
    從一堆 tensor 挑一個方陣，優先 d×d，其次 (d+1)×(d+1)（會切掉 CLS）。
    回傳：np.ndarray[d,d] 或 None
    """
    squares = [t for t in tensors if _is_square(t)]
    if not squares:
        return None, None  # (array, raw_shape)

    # 依尺寸排序：先 d，再 d+1，再其他（保底）
    squares.sort(key=lambda t: (
        0 if t.shape[-1] == d else (1 if (accept_plus1 and t.shape[-1] == d+1) else 2),
        t.shape[-1]
    ))
    A = squares[0]
    raw_shape = tuple(A.shape)

    # 多頭合併（跟你原本一樣）
    if A.ndim == 4:  # (B, H, n, n)
        if args.head_reduce == 'sum':
            A = A.sum(dim=1)
        elif args.head_reduce == 'max':
            A = A.max(dim=1).values
        else:
            A = A.mean(dim=1)

    # 批次合併
    if A.ndim == 3:  # (B, n, n)
        A = (A.sum(dim=0) if args.batch_reduce == 'sum' else A.mean(dim=0))

    # 取絕對值
    if args.do_abs:
        A = A.abs()

    # 若是 (d+1)x(d+1) → 去 CLS
    n = A.shape[-1]
    if n == d + 1 and accept_plus1:
        if cls_pos == 'first':
            A = A[1:, 1:]
        else:  # 'last'
            A = A[:-1, :-1]

    # 對稱化
    if args.do_sym:
        A = 0.5 * (A + A.T)

    return A.detach().to('cpu').numpy(), raw_shape

def reduce_heads(A: torch.Tensor, mode='mean') -> torch.Tensor:
    if A.ndim == 4:  # (B, H, d, d)
        if mode == 'sum':
            A = A.sum(dim=1)
        elif mode == 'max':
            A = A.max(dim=1).values
        else:
            A = A.mean(dim=1)
    return A


def run(args):
    set_seed(args.seed)
    repo = Path(args.t2g_repo)
    add_repo_to_path(repo)

    # 匯入官方元件
    from bin import T2GFormer

    base = Path(args.baseline_dir)
    var  = Path(args.variant_dir)
    out  = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 載入 GRAPE 中介物
    B = load_baseline_artifacts(base)
    X, M, y, sp = B['X'], B['M'], B['y'], B['splits']
    n, d = X.shape
    device = torch.device(args.device)

    # 決定 train/val indices（避免資料洩漏）
    train_idx = sp['train_idx'] if 'train_idx' in sp else None
    val_idx   = sp['val_idx']   if 'val_idx' in sp else None
    if train_idx is None:
        raise FileNotFoundError('找不到 train_idx，請確認 baseline_dir 有 split_idx.npz 或 train_idx.npy')

    # 若選擇 overlay，則將 train rows 交集 overlay_keep
    overlay_keep = None
    if args.train_on == 'overlay':
        overlay_keep = derive_overlay_row_keep(M, var)
        if overlay_keep is None:
            print('[t2gexp] 未找到可用的 overlay 遮罩，退回 baseline train split')
        else:
            train_idx = train_idx[overlay_keep[train_idx]]
            if val_idx is not None:
                val_idx = val_idx[overlay_keep[val_idx]]

    # 建立 Tensor / DataLoader
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy((y.astype(np.float32) if y is not None else np.zeros((n,1), np.float32)))
    if y_t.ndim == 1:
        y_t = y_t[:, None]

    def make_dl(idx, shuffle):
        td = TensorDataset(X_t[idx]) if y is None else TensorDataset(X_t[idx], y_t[idx])
        return DataLoader(td, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_dl(train_idx, shuffle=(args.epochs > 0))
    val_loader   = make_dl(val_idx, shuffle=False) if val_idx is not None else None

    # ===== 讀官方預設 config：configs/default/T2GFormer/cfg.json 並建模 =====
    import json

    cfg_path = Path(args.t2g_repo) / "configs" / "default" / "T2GFormer" / "cfg.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到官方 config：{cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # 官方 cfg 的模型參數通常在 "model" 欄位；沒有就直接用整個物件
    # 先取 model 區塊，然後把頂層（若有）同名鍵覆蓋進來
    model_kwargs = dict(cfg.get("model", {}))
    for k in ("token_bias", "kv_compression", "kv_compression_sharing"):
        if k in cfg:  # 有些repo把這些放在cfg頂層
            model_kwargs[k] = cfg[k]

    # 若仍缺，補安全預設（僅針對這三個鍵）
    model_kwargs.setdefault("token_bias", True)
    model_kwargs.setdefault("kv_compression", None)         # T2G-Former 論文沒有把它當貢獻點描述
    model_kwargs.setdefault("kv_compression_sharing", None) # T2G-Former 論文沒有把它當貢獻點描述
    # 缺，補安全預設（僅針對這三個鍵）

    # 決定 d_out（其餘參數完全照官方 config）
    if args.task == "reg" or y is None:
        d_out = 1
    else:
        try:
            d_out = int(np.unique(y).size)
        except Exception:
            d_out = 1

    model = T2GFormer(
        d_numerical=d,
        categories=None,
        d_out=d_out,
        **model_kwargs,
    ).to(device)
    os.environ["T2G_EXPORT_DIR"] = str(out) # test
    model.eval()
    print(f"[t2gexp] use T2GFormer config from: {cfg_path}")

    # Forward hooks 蒐集 (B,H,d,d) / (B,d,d)
    buckets: Dict[str, torch.Tensor] = {}
    layer_shapes: Dict[str, tuple] = {}   # ← 新增：記錄每層原始 shape
    seen: List[str] = []
    pat = re.compile(args.module_regex)

    def hook(name):
        def _iter_all(x):
            if torch.is_tensor(x):
                yield x
            elif isinstance(x, (list, tuple)):
                for xi in x: 
                    yield from _iter_all(xi)
            elif isinstance(x, dict):
                for xi in x.values(): 
                    yield from _iter_all(xi)

        def fn(module, inp, out):
            # 1) 來自 in/out 的所有張量
            tensors = list(_iter_all(inp)) + list(_iter_all(out))

            # 2) 再把 module 的 buffers / 直屬參數 / 直屬屬性裡的張量也納入
            for _, b in module.named_buffers(recurse=False):
                tensors.append(b)
            for _, p in module.named_parameters(recurse=False):
                tensors.append(p)
            for v in module.__dict__.values():
                if torch.is_tensor(v):
                    tensors.append(v)

            A_np, raw_shape = _pick_square_cand(
                tensors, d=d, accept_plus1=args.accept_plus1, cls_pos=args.cls_pos
            )
            if A_np is None:
                return
            if name not in buckets:
                buckets[name] = torch.zeros((d, d), device='cpu', dtype=torch.float32)
            buckets[name] += torch.from_numpy(A_np).to(buckets[name].dtype)
            layer_shapes[name] = raw_shape
        return fn

    handles = []
    for name, module in model.named_modules():
        if pat.search(name) or pat.search(module.__class__.__name__):
            seen.append(f"{name} <{module.__class__.__name__}>")
            handles.append(module.register_forward_hook(hook(name)))

    if args.print_modules:
        print("Matched modules:\n" + "\n".join(seen))
        for h in handles: h.remove()
        return

    # 簡單訓練 loop（可選；epochs=0 則只 forward）
    criterion = nn.MSELoss() if args.task=='reg' else nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_loader(dl, train=False, max_batches=0):
        if dl is None: return
        model.train(train)
        for bi, batch in enumerate(dl):
            xb = batch[0].to(device)
            yb = (batch[1].to(device) if len(batch) > 1 else None)
            out = model(xb, None)
            if train and yb is not None:
                loss = criterion(out, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            if max_batches and (bi+1) >= max_batches:
                break

    # 訓練/匯出
    if args.epochs > 0:
        for ep in range(args.epochs):
            run_loader(train_loader, train=True, max_batches=args.max_batches)
            run_loader(val_loader,   train=False, max_batches=args.max_batches)
    else:
        # 只 forward（train/val 都跑以覆蓋更多分佈，但不觸 test）
        run_loader(train_loader, train=False, max_batches=args.max_batches)
        run_loader(val_loader,   train=False, max_batches=args.max_batches)

    from bin.t2g_former import flush_t2g_exports    #test
    flush_t2g_exports()                             #test

    # 存檔
    names = sorted(buckets.keys())
    saved = []
    for li, name in enumerate(names):
        W = buckets[name].detach().cpu().numpy()
        if args.do_sym:
            W = 0.5 * (W + W.T)
        if args.norm_per_layer:
            denom = np.linalg.norm(W)
            if denom > 0: W = W / denom
        fp = Path(out) / f'W_layer{li:02d}.npy'
        np.save(fp, W.astype(np.float32))
        saved.append({'name': name, 'shape': list(W.shape), 'file': str(fp)})

    # 紀錄 log 以利重現
    log = {
        'baseline_dir': str(base), 'variant_dir': str(var), 'output': str(out),
        'n': int(n), 'd': int(d), 'epochs': args.epochs, 'lr': args.lr,
        'train_on': args.train_on, 'max_batches': args.max_batches,
        'matched_modules': seen,
    }
    with open(Path(out)/'t2g_export_log.json','w',encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    for h in handles: h.remove()
    print(f'[t2gexp] Exported {len(saved)} layer matrices → {out}')


if __name__ == '__main__':
    args = get_args()
    run(args)
