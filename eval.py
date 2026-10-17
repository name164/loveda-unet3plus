# -*- coding: utf-8 -*-
"""
最终评估：完整 1024x1024 滑窗推理 + 分域指标

    python eval.py                      # 用 best.pt 在 test 集上跑
    python eval.py --split val          # 换成验证集
    python eval.py --limit 30           # 只跑 30 张，快速看一眼
    python eval.py --stride 512         # 取消重叠，速度翻倍（边缘质量略降）

为什么不用缩放或中心裁剪
------------------------
训练时用的是 1024 原图里随机裁的 512 块。测试时如果把整张 1024 缩到 512，
地物尺度就和训练不一致，小目标会糊掉；只取中心裁剪则丢掉 3/4 的画面。
滑窗是在原分辨率上逐块推理再把结果拼回去，既不丢分辨率也不丢画面，
是这批数据上最可信的评估方式。

重叠区域按 softmax 概率求平均（不是对 argmax 投票），这样接缝处更平滑。
"""

import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

import config as C
from dataset import build_loader, key_to_paths
from metrics import DomainMetrics, save_prediction_grid
from model import build_model


def to_cl(x):
    if C.CHANNELS_LAST:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def autocast_ctx():
    if C.AMP_BF16:
        return torch.autocast('cpu', dtype=torch.bfloat16)
    return torch.autocast('cpu', enabled=False)


def _window_starts(total, window, stride):
    """算出滑窗在每个轴上的起点，保证最后一个窗口刚好贴到边界。"""
    if total <= window:
        return [0]
    starts = list(range(0, total - window + 1, stride))
    if starts[-1] != total - window:
        starts.append(total - window)
    return starts


@torch.no_grad()
def sliding_predict(model, img, window, stride, progress_cb=None):
    """
    img : (3, H, W) float32，已归一化
    返回 (H, W) 的类别索引图

    progress_cb : 可选，签名 (已完成窗口数, 总窗口数) -> bool。
                  返回 False 表示要求中断，本函数随即返回 None。
                  gui.py 用它来驱动进度条和「取消」按钮；命令行调用不传，
                  行为与从前完全一致。
    """
    _, H, W = img.shape

    prob_sum = torch.zeros(C.NUM_CLASSES, H, W, dtype=torch.float32)
    count = torch.zeros(1, H, W, dtype=torch.float32)

    ys = _window_starts(H, window, stride)
    xs = _window_starts(W, window, stride)
    n_total = len(ys) * len(xs)

    for i, (y, x) in enumerate((y, x) for y in ys for x in xs):
        patch = img[:, y:y + window, x:x + window].unsqueeze(0)
        with autocast_ctx():
            outs = model(to_cl(patch))
        # 只用主输出 d1，转回 fp32 再 softmax
        prob = outs[0].float().softmax(dim=1)[0]
        prob_sum[:, y:y + window, x:x + window] += prob
        count[:, y:y + window, x:x + window] += 1

        if progress_cb is not None and progress_cb(i + 1, n_total) is False:
            return None

    return (prob_sum / count).argmax(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--ckpt', default=None, help='默认 checkpoints/best.pt')
    ap.add_argument('--window', type=int, default=C.SLIDING_WINDOW)
    ap.add_argument('--stride', type=int, default=C.SLIDING_STRIDE)
    ap.add_argument('--limit', type=int, default=0, help='只评估前 N 张，0 表示全部')
    ap.add_argument('--vis', type=int, default=8, help='输出多少张可视化对比图')
    ap.add_argument('--out', default=None, help='可视化图片输出目录')
    args = ap.parse_args()

    torch.set_num_threads(C.TORCH_THREADS)
    C.ensure_dirs()

    ckpt_path = args.ckpt or os.path.join(C.CKPT_DIR, 'best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重 {ckpt_path}，先跑 train.py")

    print("=" * 78)
    print("UNet3+ / LoveDA 滑窗评估")
    print("=" * 78)

    # ---------------- 模型 ----------------
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    if C.CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    n_img = args.limit or len(build_loader(args.split, 'full', batch_size=1).dataset)
    per_img = len(_window_starts(1024, args.window, args.stride)) ** 2
    print(f"  权重     {ckpt_path}")
    print(f"  训练轮数 {ckpt.get('epoch', '?')}   验证集 best mIoU {ckpt.get('best_miou', float('nan')):.4f}")
    print(f"  评估集   {args.split}   共 {n_img} 张 1024x1024")
    print(f"  滑窗     {args.window} 尺寸 / {args.stride} 步长 -> 每张 {per_img} 个窗口"
          f"（重叠区按 softmax 概率平均）")
    print(f"  预计每张约 {per_img * 1.5:.1f} 秒，总计约 {n_img * per_img * 1.5 / 60:.0f} 分钟")

    # ---------------- 推理 ----------------
    # 用 batch_size=1、mode='full' 的 loader，拿到未裁剪的原图
    loader = build_loader(args.split, 'full', batch_size=1)
    dm = DomainMetrics()
    vis_samples = []
    out_dir = args.out or os.path.join(C.OUT_DIR, f'sliding_{args.split}')
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    pbar = tqdm(total=min(n_img, len(loader)), desc='滑窗推理', ncols=110)
    for i, (img, mask, keys) in enumerate(loader):
        if args.limit and i >= args.limit:
            break

        key = keys[0]
        pred = sliding_predict(model, img[0], args.window, args.stride)

        domain = key.split('/', 1)[0]
        dm.update(pred, mask[0], domain)

        if len(vis_samples) < args.vis:
            vis_samples.append(dict(image=img[0].numpy(), gt=mask[0].numpy(),
                                    pred=pred.numpy(), title=key))
        pbar.update(1)
    pbar.close()

    elapsed = (time.time() - t0) / 60
    print(f"\n推理完成，用时 {elapsed:.1f} 分钟")

    # ---------------- 结果 ----------------
    res = dm.report(f"评估结果：{args.split} 集（完整 1024 原分辨率滑窗）")

    # ---------------- 可视化 ----------------
    if vis_samples:
        path = os.path.join(out_dir, 'predictions.png')
        save_prediction_grid(vis_samples, path, max_rows=len(vis_samples))
        print(f"\n可视化已写出 -> {path}")

    # ---------------- 存一份文本报告 ----------------
    txt = os.path.join(out_dir, 'report.txt')
    with open(txt, 'w', encoding='utf-8') as f:
        f.write(f"权重: {ckpt_path}\n")
        f.write(f"划分: {args.split}   滑窗 {args.window}/{args.stride}\n")
        f.write(f"用时: {elapsed:.1f} 分钟\n\n")
        f.write(f"{'域':<8}{'mIoU':>10}{'mDice':>10}{'aAcc':>10}{'oAcc':>10}\n")
        for k in ['ALL'] + C.DOMAINS:
            r = res[k]
            f.write(f"{k:<8}{r['miou']:>10.4f}{r['mdice']:>10.4f}"
                    f"{r['aacc']:>10.4f}{r['oacc']:>10.4f}\n")
        f.write(f"\n{'类别':<8}{'IoU':>10}{'Dice':>10}{'占比':>10}\n")
        for c, name in enumerate(C.CLASS_NAMES):
            v = res['ALL']['iou'][c]
            d = res['ALL']['dice'][c]
            share = res['ALL']['gt_pixels'][c] / max(res['ALL']['total_pixels'], 1)
            iou_s = '--' if np.isnan(v) else f'{v:.4f}'
            dice_s = '--' if np.isnan(d) else f'{d:.4f}'
            f.write(f"{name:<8}{iou_s:>10}{dice_s:>10}{share * 100:>9.2f}%\n")
    print(f"文本报告已写出 -> {txt}")


if __name__ == '__main__':
    main()
