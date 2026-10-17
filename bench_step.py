# -*- coding: utf-8 -*-
"""
单步训练耗时基准 —— 用来定位「到底是数据加载慢，还是计算慢」

背景：全量训练实测约 4.4 s/step（batch 2, 512²），而探针估的是 3.14 s/step。
差在哪？这个脚本把「纯计算」和「端到端」分开测，逐个变量对照。

    python bench_step.py

几组对照：
  [纯计算]         数据预先读进内存，只测 forward+backward+optimizer
                    -> 这是速度下限，数据加载再快也超不过它
  [端到端]         真实 DataLoader 迭代，叠加数据加载与线程争抢
  cv2 线程数       OpenCV 默认每个进程开满核，4 个 worker 就是 4×16 线程
                    和主进程的 torch 线程抢同一批核 —— 经典 CPU 训练超订
"""

import time

import cv2
import numpy as np
import torch

import config as C
from dataset import build_loader
from loss import CombinedLoss
from model import build_model

WARMUP = 1
MEASURE = 4


class CvThreadInit:
    """
    必须在模块级且用**类实例**：闭包/局部函数无法 pickle，Windows 的 spawn
    起 worker 时会直接崩。

    为什么不能用模块级函数 + 全局变量：spawn 模式下 worker 会重新 import 本模块，
    父进程改过的全局变量在子进程里读不到，拿到的还是默认值。实例自带属性就没这问题。

    OpenCV 默认在每个进程里开满核做 resize/色彩转换。4 个 worker 就是 4×16 线程，
    和主进程的 torch 线程抢同一批物理核 —— 这是 CPU 训练最常见的超订来源。
    """

    def __init__(self, n_threads):
        self.n_threads = n_threads

    def __call__(self, _worker_id):
        cv2.setNumThreads(self.n_threads)


def _prep(model):
    """每个配置都从一份干净的优化器状态开始，避免状态累积影响计时。"""
    model = model.to(memory_format=torch.channels_last) if C.CHANNELS_LAST else model
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    return model, opt


def _one_step(model, opt, crit, img, mask, clip=True):
    img = img.contiguous(memory_format=torch.channels_last) if C.CHANNELS_LAST else img
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        outs = model(img)
        loss, _ = crit(outs, mask)
    loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
    opt.step()
    return float(loss)


def _run(model, opt, crit, batches, clip=True, label=''):
    for i in range(WARMUP):
        _one_step(model, opt, crit, *batches[i % len(batches)], clip=clip)
    ts = []
    for i in range(MEASURE):
        t0 = time.perf_counter()
        _one_step(model, opt, crit, *batches[i % len(batches)], clip=clip)
        ts.append(time.perf_counter() - t0)
    mean = float(np.mean(ts))
    print(f"  {label:<34} {mean:6.3f} s/step   (最快 {min(ts):.3f}  最慢 {max(ts):.3f})")
    return mean


def main():
    torch.backends.cudnn.enabled = False
    print("=" * 74)
    print("单步耗时基准   batch=%d  尺寸=%d  bf16=%s  channels_last=%s"
          % (C.BATCH_SIZE, C.CROP_SIZE, C.AMP_BF16, C.CHANNELS_LAST))
    print("=" * 74)

    print(f"\n物理核/逻辑核请自行对照；cv2 默认线程数 = {cv2.getNumThreads()}")

    # ---------------- 预备数据：读进内存，后面纯计算不再碰磁盘 ----------------
    print("\n预读数据中...")
    loader = build_loader('train', 'train', batch_size=C.BATCH_SIZE)
    batches = []
    for i, (img, mask, _) in enumerate(loader):
        batches.append((img, mask))
        if i >= MEASURE + WARMUP - 1:
            break
    print(f"已预读 {len(batches)} 个 batch（后续纯计算测试复用）")

    crit = CombinedLoss()

    # ================= 一、纯计算：找出速度下限 =================
    print("\n" + "-" * 74)
    print("[一] 纯计算（数据已在内存，无 DataLoader 干扰）")
    print("-" * 74)
    for th in [16, 8, 12]:
        torch.set_num_threads(th)
        model, opt = _prep(build_model())
        _run(model, opt, crit, batches, label=f'torch 线程 = {th}')
        del model, opt

    # 梯度裁剪本身在 CPU 上要遍历所有参数算范数，单独看看它值多少
    torch.set_num_threads(8)
    model, opt = _prep(build_model())
    _run(model, opt, crit, batches, clip=False, label='torch 线程 = 8（不做梯度裁剪）')
    del model, opt

    # ================= 二、端到端：加上数据加载 =================
    print("\n" + "-" * 74)
    print("[二] 端到端（真实 DataLoader）")
    print("-" * 74)

    for nw, nthread_cv, th in [(4, 16, 16), (4, 1, 16), (2, 1, 16), (0, 16, 16)]:
        torch.set_num_threads(th)

        base = build_loader('train', 'train', batch_size=C.BATCH_SIZE)
        from torch.utils.data import DataLoader
        dl = DataLoader(base.dataset, batch_size=C.BATCH_SIZE, shuffle=True,
                        num_workers=nw, drop_last=True, pin_memory=False,
                        persistent_workers=(nw > 0),
                        prefetch_factor=2 if nw > 0 else None,
                        worker_init_fn=CvThreadInit(nthread_cv) if nw > 0 else None)

        model, opt = _prep(build_model())
        it = iter(dl)
        for i in range(WARMUP):
            emb = next(it)
            _one_step(model, opt, crit, emb[0], emb[1])
        ts = []
        for i in range(MEASURE):
            emb = next(it)
            t0 = time.perf_counter()
            _one_step(model, opt, crit, emb[0], emb[1])
            ts.append(time.perf_counter() - t0)
        mean = float(np.mean(ts))
        print(f"  workers={nw}  cv2线程={nthread_cv}  torch={th:<4} "
              f"{mean:6.3f} s/step   (最快 {min(ts):.3f}  最慢 {max(ts):.3f})")
        del model, opt, dl, it

    print("\n" + "=" * 74)
    print("解读要点")
    print("=" * 74)
    print("  - [一] 里最低的那一行就是速度下限。端到端若明显高于它，瓶颈在数据加载/线程争抢。")
    print("  - 若 cv2线程=1 明显快于 16，说明是 OpenCV 多线程和 torch 抢核。")
    print("  - 若「不做梯度裁剪」明显更快，就把它关掉或改成每 N 步一次。")


if __name__ == '__main__':
    main()
