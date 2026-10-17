# -*- coding: utf-8 -*-
"""
训练主循环

    python train.py                 # 正常训练（会自动从 last.pt 续训）
    python train.py --smoke         # 冒烟测试：只用少量样本跑 2 轮，验证链路
    python train.py --epochs 5      # 临时改轮数
    python train.py --fresh         # 忽略已有断点，从头开始

学习率策略：线性 warmup 3 轮 + 余弦退火到 1e-6
    warmup 是为了压制 Dice 在训练初期的震荡（纯 Dice 多分类的已知问题），
    之后按余弦曲线平滑退火，这是 config 里指定的策略。
"""

import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config as C
from dataset import build_loader
from loss import CombinedLoss
from metrics import DomainMetrics, save_prediction_grid
from model import build_model, count_parameters


# ============================================================================
# 工具
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_runtime():
    """线程数与内存布局。这两项对 CPU 训练速度影响很大，见 config 里的实测数据。"""
    torch.set_num_threads(C.TORCH_THREADS)
    C.ensure_dirs()
    # 让显卡那套报错别来烦人，我们是纯 CPU
    torch.backends.cudnn.enabled = False


def move_to_cl(model):
    """把模型转成 channels_last。纯布局调整，不改变任何数学。"""
    if C.CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    return model


def to_cl(x):
    if C.CHANNELS_LAST:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def make_scheduler(optimizer, epochs, warmup, eta_min_ratio):
    """
    线性 warmup + 余弦退火，合成一条曲线，用 LambdaLR 实现。

        epoch <  warmup : lr = base * (epoch+1) / warmup          线性爬升
        epoch >= warmup : lr = base * (eta_min + (1-eta_min) * 0.5*(1+cos(pi*t)))
                          其中 t = (epoch-warmup)/(epochs-warmup)
    """
    def lr_lambda(epoch):
        if epoch < warmup:
            return float(epoch + 1) / max(1, warmup)
        t = (epoch - warmup) / max(1, epochs - warmup)
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def autocast_ctx():
    """按配置决定要不要开 bf16。"""
    if C.AMP_BF16:
        return torch.autocast('cpu', dtype=torch.bfloat16)
    return torch.autocast('cpu', enabled=False)


# ============================================================================
# 训练 / 验证
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs):
    model.train()
    running = 0.0
    n_seen = 0
    parts_sum = {}

    pbar = tqdm(loader, desc=f"训练 {epoch}/{total_epochs}", ncols=110, leave=False)
    for img, mask, _ in pbar:
        img = to_cl(img)
        # mask 是 int64，channels_last 对它没有意义，保持原样

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            outs = model(img)
            loss, parts = criterion(outs, mask)

        loss.backward()
        # Dice 容易出尖峰，裁剪一下
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        optimizer.step()

        bs = img.size(0)
        running += loss.item() * bs
        n_seen += bs
        for k, v in parts.items():
            parts_sum[k] = parts_sum.get(k, 0.0) + v * bs

        pbar.set_postfix(loss=f"{running / max(n_seen, 1):.4f}",
                         gn=f"{float(gn):.2f}")

    avg = running / max(n_seen, 1)
    return avg, {k: v / max(n_seen, 1) for k, v in parts_sum.items()}


@torch.no_grad()
def validate(model, loader, epoch, total_epochs, collect_vis=0):
    """
    每轮验证。用中心裁剪 512，跑得快，只用来跟踪训练趋势。
    注意：它只覆盖每张图的中心区域，指标不能直接和最终滑窗测试比。
    """
    model.eval()
    dm = DomainMetrics()
    samples = []

    pbar = tqdm(loader, desc=f"验证 {epoch}/{total_epochs}", ncols=110, leave=False)
    for img, mask, keys in pbar:
        img_cl = to_cl(img)
        with autocast_ctx():
            outs = model(img_cl)
        # 主输出 d1，转回 fp32 再取 argmax
        pred = outs[0].float().argmax(dim=1)

        for b in range(pred.size(0)):
            domain = keys[b].split('/', 1)[0]
            dm.update(pred[b], mask[b], domain)

            if len(samples) < collect_vis and collect_vis > 0:
                samples.append(dict(
                    image=img[b].numpy(),
                    gt=mask[b].numpy(),
                    pred=pred[b].numpy(),
                    title=f"{keys[b]}",
                ))

    return dm, samples


# ============================================================================
# 断点
# ============================================================================

def save_ckpt(path, model, optimizer, scheduler, epoch, best_miou, history,
              sched_sig=None):
    torch.save(dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        epoch=epoch,
        best_miou=best_miou,
        history=history,
        config=dict(
            enc_filters=C.ENC_FILTERS,
            num_classes=C.NUM_CLASSES,
            crop_size=C.CROP_SIZE,
        ),
        sched_sig=sched_sig,
    ), path)


def load_ckpt(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt


# ============================================================================
# 主流程
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='冒烟测试：少量样本 + 2 轮')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--fresh', action='store_true', help='忽略已有断点，从头训')
    args = ap.parse_args()

    epochs = args.epochs or (2 if args.smoke else C.EPOCHS)
    batch = args.batch or C.BATCH_SIZE
    smoke_n = 200 if args.smoke else C.SMOKE_SUBSET

    setup_runtime()
    set_seed(C.SEED)

    print("=" * 78)
    print("UNet3+ / LoveDA 训练" + ("   [冒烟模式]" if args.smoke else ""))
    print("=" * 78)
    print(f"  CPU 线程 {C.TORCH_THREADS}   channels_last {C.CHANNELS_LAST}   "
          f"bf16 {C.AMP_BF16}")
    print(f"  分辨率 {C.CROP_SIZE}   batch {batch}   epochs {epochs}")
    print(f"  优化器 AdamW  lr {C.LR}  wd {C.WEIGHT_DECAY}")
    print(f"  学习率 warmup {C.WARMUP_EPOCHS} 轮 + 余弦退火到 {C.ETA_MIN}")
    print(f"  损失 纯 Dice" if C.CE_WEIGHT == 0 else f"  损失 Dice + {C.CE_WEIGHT}*CE")
    print(f"  深监督 {C.DS_LOSS}  权重 {C.DS_WEIGHTS}")

    # ---------------- 数据 ----------------
    train_loader = build_loader('train', 'train', batch_size=batch, smoke=smoke_n)
    val_loader = build_loader('val', 'val', batch_size=batch, smoke=smoke_n)
    print(f"\n  训练样本 {len(train_loader.dataset)}   验证样本 {len(val_loader.dataset)}")

    # ---------------- 模型 ----------------
    model = move_to_cl(build_model())
    total, _ = count_parameters(model)
    print(f"  模型参数量 {total:,}")

    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR,
                                  weight_decay=C.WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, epochs, C.WARMUP_EPOCHS,
                               C.ETA_MIN / C.LR)
    # lr 曲线的全部参数，随断点一起存。
    # LambdaLR.state_dict() 只存 last_epoch、不存曲线本身，所以续训时若这些值
    # 变了，曲线会按新参数重算 —— 例如 30 轮改 40 轮，第 31 轮的 lr 会从 1e-6
    # 突跳回高位。存下来是为了在续训启动时就拦下这种错误，而不是跑几小时后才发现。
    sched_sig = dict(epochs=epochs, warmup=C.WARMUP_EPOCHS,
                     eta_min_ratio=C.ETA_MIN / C.LR)

    # ---------------- 断点 ----------------
    # 冒烟测试写独立的文件名。否则 --smoke 产生的 2 轮模型会覆盖正式训练的断点，
    # 之后正常启动会「从第 3 轮续训」，几十小时的成果等于没了。
    ckpt_tag = 'smoke_' if args.smoke else ''
    last_path = os.path.join(C.CKPT_DIR, f'{ckpt_tag}last.pt')
    best_path = os.path.join(C.CKPT_DIR, f'{ckpt_tag}best.pt')
    start_epoch, best_miou, history = 1, -1.0, []

    if os.path.exists(last_path) and not args.fresh:
        ckpt = load_ckpt(last_path, model, optimizer, scheduler)

        # 校验 lr 曲线参数没被改过。老断点没这个字段，跳过。
        old_sig = ckpt.get('sched_sig')
        if old_sig is not None and old_sig != sched_sig:
            print("\n" + "!" * 78)
            print("拒绝续训：断点里的学习率曲线参数和本次启动的不一致")
            print("!" * 78)
            for k in sorted(set(old_sig) | set(sched_sig)):
                a, b = old_sig.get(k), sched_sig.get(k)
                if a != b:
                    print(f"    {k:<16} 断点里是 {a}   本次启动是 {b}")
            print("\n  LambdaLR 只存 last_epoch、不存曲线本身，沿用会让 lr 从退火终点")
            print("  突跳回高位（例如 1e-6 -> 4e-4），足以毁掉整次训练。")
            print("\n  怎么办：")
            print("    * 想接着训  -> 把参数改回断点里的值，直接 python train.py")
            print("    * 想换轮数  -> 只能加 --fresh 从头重训，lr 曲线无法中途延长")
            print("!" * 78 + "\n")
            raise SystemExit(1)

        start_epoch = ckpt['epoch'] + 1
        best_miou = ckpt.get('best_miou', -1.0)
        history = ckpt.get('history', [])
        print(f"\n  已从断点续训：上次跑到第 {ckpt['epoch']} 轮，"
              f"best mIoU {best_miou:.4f}，本次从第 {start_epoch} 轮继续")
        if start_epoch > epochs:
            print(f"  已经跑满 {epochs} 轮，直接退出。想接着跑就把 --epochs 调大。")
            return

    # ---------------- 日志 ----------------
    stamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(C.LOG_DIR, f'{ckpt_tag}train_{stamp}.csv')
    log_file = open(log_path, 'w', newline='', encoding='utf-8')
    # 把裸地单列出来。它的 IoU 从 0 恢复到 0.3 只让总 mIoU 涨 0.04，
    # 会被其它类的正常波动淹没 —— 不盯着这一列，就发现不了修复到底有没有生效。
    #
    # 列名原来叫 val_iou_forest，是类名错位时期的误标：当时以为下标 4 是林地，
    # 实际按 LoveDA 官方顺序下标 4 是裸地（详见 config.CLASS_NAMES 上方注释）。
    # 旧日志里那一列仍叫 forest，gui.py 读的时候两个列名都认。
    tracked_idx = C.CLASS_NAMES.index('裸地')
    writer = csv.writer(log_file)
    writer.writerow(['epoch', 'lr', 'train_loss', 'dice_d1', 'dice_d2', 'dice_d3',
                     'dice_d4', 'dice_d5', 'val_miou', 'val_mdice', 'val_aacc',
                     'val_oacc', 'val_miou_rural', 'val_miou_urban',
                     'val_iou_barren', 'minutes'])
    print(f"  日志 {log_path}\n")

    # ---------------- 训练循环 ----------------
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]['lr']

        train_loss, parts = train_one_epoch(model, train_loader, criterion,
                                            optimizer, epoch, epochs)

        # 验证时每轮收几张图做可视化
        collect = 4 if (epoch % C.SAVE_VIS_EVERY == 0 or epoch == epochs) else 0
        dm, vis_samples = validate(model, val_loader, epoch, epochs, collect_vis=collect)
        res = dm.report(f"第 {epoch}/{epochs} 轮验证结果", quiet=True)

        scheduler.step()
        mins = (time.time() - t0) / 60

        # ---------------- 记录 ----------------
        row = dict(
            epoch=epoch, lr=lr_now, train_loss=train_loss,
            val_miou=res['ALL']['miou'], val_mdice=res['ALL']['mdice'],
            val_aacc=res['ALL']['aacc'], val_oacc=res['ALL']['oacc'],
            val_miou_rural=res['Rural']['miou'], val_miou_urban=res['Urban']['miou'],
            val_iou_barren=res['ALL']['iou'][tracked_idx],
            minutes=mins,
        )
        # 该类在验证集里完全没出现时 IoU 记 NaN，写成 '--'
        tracked_s = ('--' if np.isnan(row['val_iou_barren'])
                     else f"{row['val_iou_barren']:.5f}")
        writer.writerow([
            epoch, f"{lr_now:.6f}", f"{train_loss:.5f}",
            *[f"{parts.get(f'dice_d{i}', 0.0):.5f}" for i in range(1, 6)],
            f"{row['val_miou']:.5f}", f"{row['val_mdice']:.5f}",
            f"{row['val_aacc']:.5f}", f"{row['val_oacc']:.5f}",
            f"{row['val_miou_rural']:.5f}", f"{row['val_miou_urban']:.5f}",
            tracked_s,
            f"{mins:.2f}",
        ])
        log_file.flush()

        history.append(row)

        # ---------------- 保存 ----------------
        # 先判定改进、更新 best_miou，再存 last.pt。
        # 反过来的话 last.pt 里的 best_miou 会落后一轮，续训时起点偏低，
        # 下一轮只要略好于那个旧值就会被判成「改进」，用更差的权重覆盖 best.pt。
        improved = row['val_miou'] > best_miou
        if improved:
            best_miou = row['val_miou']

        save_ckpt(last_path, model, optimizer, scheduler, epoch, best_miou, history,
                  sched_sig)

        if improved:
            save_ckpt(best_path, model, optimizer, scheduler, epoch, best_miou, history,
                      sched_sig)

        # ---------------- 可视化 ----------------
        if vis_samples:
            save_prediction_grid(vis_samples,
                                 os.path.join(C.OUT_DIR, f'epoch{epoch:03d}_pred.png'))

        print(f"[第 {epoch:3d}/{epochs} 轮] loss {train_loss:8.4f} | "
              f"mIoU {row['val_miou']:.4f}  aAcc {row['val_aacc']:.4f} | "
              f"Rural {row['val_miou_rural']:.4f}  Urban {row['val_miou_urban']:.4f} | "
              f"裸地IoU {tracked_s} | "
              f"{mins:.1f} 分钟" + ("   ★ best" if improved else ""))

    log_file.close()

    # ---------------- 收尾 ----------------
    print("\n" + "=" * 78)
    print(f"训练结束。best mIoU = {best_miou:.4f}")
    print(f"权重: {best_path}")
    print(f"日志: {log_path}")
    print("=" * 78)

    dm, _ = validate(model, val_loader, epochs, epochs)
    dm.report("最终验证集结果（中心裁剪 512，仅用于趋势参考）")

    if not args.smoke:
        print("\n下一步：跑 eval.py 做完整 1024 滑窗测试，那才是最终可信的指标。")


if __name__ == '__main__':
    main()
