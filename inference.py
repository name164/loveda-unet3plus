# -*- coding: utf-8 -*-
"""Web 端复用的推理接口（不依赖 PyQt）。"""

import os
import time

import cv2
import numpy as np
import torch

import config as C
from eval import sliding_predict
from metrics import colorize
from model import build_model

_CACHE = {}


def load_model(ckpt_path=None):
    """加载并缓存模型；权重文件更新后自动重载。"""
    path = os.path.abspath(ckpt_path or os.path.join(C.CKPT_DIR, 'best.pt'))
    if not os.path.exists(path):
        raise FileNotFoundError(f'找不到权重: {path}')
    key = (path, os.path.getmtime(path))
    if key not in _CACHE:
        model = build_model()
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        if C.CHANNELS_LAST:
            model = model.to(memory_format=torch.channels_last)
        model.eval()
        _CACHE.clear()
        _CACHE[key] = (model, ckpt)
    return _CACHE[key]


def _to_rgb(image):
    if hasattr(image, 'convert'):
        return np.asarray(image.convert('RGB'))
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1).astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, :3].astype(np.uint8))


def infer(image, ckpt_path=None, window=512, stride=256, max_side=2048,
          progress_cb=None):
    """PIL 图或 ndarray -> 原图 / 染色图 / 叠加图 / 类别统计。"""
    t0 = time.time()
    rgb = _to_rgb(image)

    h, w = rgb.shape[:2]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)),
                         interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]

    original = rgb
    # 小于窗口的图先补齐，推理后裁回
    ph, pw = max(0, window - h), max(0, window - w)
    if ph or pw:
        rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REPLICATE)

    x = rgb.astype(np.float32) / 255.0
    x = (x - np.asarray(C.NORM_MEAN, np.float32)) / np.asarray(C.NORM_STD, np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()

    model, _ = load_model(ckpt_path)
    pred = sliding_predict(model, tensor, window, stride, progress_cb).numpy()[:h, :w]

    colored = colorize(pred)
    overlay = (original.astype(np.float32) * 0.55
               + colored.astype(np.float32) * 0.45).astype(np.uint8)

    counts = np.bincount(pred.ravel(), minlength=C.NUM_CLASSES)
    total = max(int(counts.sum()), 1)
    stats = [[name, int(n), f'{n / total * 100:.2f}%']
             for name, n in zip(C.CLASS_NAMES, counts)]

    return dict(original=original, colored=colored, overlay=overlay,
                stats=stats, elapsed=time.time() - t0)
