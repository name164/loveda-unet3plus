# -*- coding: utf-8 -*-
"""
评价指标与可视化

指标基于混淆矩阵累加，一次算全：mIoU / aAcc / 每类 IoU / 每类 Dice(F1)。

为什么一定要**分域**报告
------------------------
本数据集的 Rural 和 Urban 差异极大（道路占比 2.2% vs 19.7%）。
把两个域混在一起算总 mIoU，结果会被占比大的域主导，
一个在乡村上很差、在城市上很好的模型，总指标看起来可能还不错。
所以 DomainMetrics 同时维护 Rural / Urban / 全体 三套矩阵，分开出报告。
"""

import os

import numpy as np
import torch

import config as C

# 7 个类别的显示颜色（RGB），顺序对应 config.CLASS_NAMES。
#
# 这里是 **LoveDA 官方调色板，逐位照抄**，取自官方参考实现（mmsegmentation 的
# mmseg/datasets/loveda.py 与 TorchGeo 的 torchgeo/datasets/loveda.py，两者一致）：
#
#     classes = ('background','building','road','water','barren','forest','agriculture')
#     palette = [[255,255,255], [255,0,0], [255,255,0], [0,0,255],
#                [159,129,183], [0,255,0], [255,195,128]]
#
# 之所以能直接照抄、不用重排：官方掩码原值 0=未标注 / 1=background /
# 2=building / 3=road / 4=water / 5=barren / 6=forest / 7=agriculture，
# 而 dataset.py 是**保序**重映射 1..7 -> 0..6，所以本项目的下标 0..6 与官方
# classes 元组顺序完全一致 —— **背景同样排在第 0 位**。
#
# （曾经这里和 config.CLASS_NAMES 一起左移了一位，把水体染成灰色、把裸地染成紫色。
#   判定依据见 config.CLASS_NAMES 上方注释：值4 最暗 = 水、值5 最亮偏土黄 = 裸地。
#   注意「背景排第 0 位」是**对的**，别再把这一位挪到末尾。）
#
# 之所以不能改成「好看的相近色」：这几个颜色是用来和官方论文/榜单图、
# 以及别人复现的结果直接肉眼对拍的，偏一点就没法比了。
PALETTE = np.array([
    [255, 255, 255],   # 0 背景   Background
    [255,   0,   0],   # 1 建筑   Building
    [255, 255,   0],   # 2 道路   Road
    [  0,   0, 255],   # 3 水体   Water
    [159, 129, 183],   # 4 裸地   Barren
    [  0, 255,   0],   # 5 林地   Forest
    [255, 195, 128],   # 6 耕地   Agriculture
], dtype=np.uint8)

IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)     # 忽略像素画成黑色
ERROR_COLOR = np.array([255, 0, 255], dtype=np.uint8)  # 错误像素画成品红


# ============================================================================
# 混淆矩阵与指标
# ============================================================================

class SegMetrics:
    """单一一组数据的混淆矩阵与派生指标。"""

    def __init__(self, num_classes=C.NUM_CLASSES, ignore_index=C.IGNORE_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        # cm[真实类, 预测类]，忽略像素不进入统计
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, pred, target):
        """
        pred   : (B, H, W) 或 (H, W)，类别索引
        target : 同形状，取值 0..C-1 或 ignore_index
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        pred = pred.astype(np.int64).ravel()
        target = target.astype(np.int64).ravel()

        mask = target != self.ignore_index
        target = target[mask]
        pred = pred[mask]

        # 理论上 pred 一定在 [0, C)，加个保险丝，防止越界把矩阵搞坏
        ok = (pred >= 0) & (pred < self.num_classes)
        target = target[ok]
        pred = pred[ok]

        np.add.at(self.cm, (target, pred), 1)

    # ------------------------------------------------------------------
    def compute(self):
        """返回一个字典，含所有指标。"""
        cm = self.cm
        tp = np.diag(cm).astype(np.float64)             # 每类真正例
        gt_sum = cm.sum(axis=1).astype(np.float64)      # 每类真实像素数（行和）
        pred_sum = cm.sum(axis=0).astype(np.float64)    # 每类预测像素数（列和）
        total = cm.sum()

        union = gt_sum + pred_sum - tp
        # 该类在真实和预测里都没出现 -> IoU 无定义，记 NaN，聚合时跳过
        iou = np.where(union > 0, tp / np.maximum(union, 1e-9), np.nan)
        dice = np.where((gt_sum + pred_sum) > 0,
                        2 * tp / np.maximum(gt_sum + pred_sum, 1e-9), np.nan)

        return {
            'miou': float(np.nanmean(iou)) if total > 0 else 0.0,
            'aacc': float(np.nanmean(np.where(gt_sum > 0,
                                              tp / np.maximum(gt_sum, 1e-9),
                                              np.nan))) if total > 0 else 0.0,
            'oacc': float(tp.sum() / total) if total > 0 else 0.0,
            'mdice': float(np.nanmean(dice)) if total > 0 else 0.0,
            'iou': iou,
            'dice': dice,
            'gt_pixels': gt_sum,
            'total_pixels': int(total),
        }


class DomainMetrics:
    """
    同时维护 Rural / Urban / 全体 三组混淆矩阵。

    update(pred, target, domain) 里 domain 传 'Rural' / 'Urban'。
    """

    def __init__(self):
        self.groups = {d: SegMetrics() for d in C.DOMAINS}
        self.groups['ALL'] = SegMetrics()

    def reset(self):
        for m in self.groups.values():
            m.reset()

    def update(self, pred, target, domain):
        self.groups[domain].update(pred, target)
        self.groups['ALL'].update(pred, target)

    # ------------------------------------------------------------------
    def report(self, title='', quiet=False):
        """打印分域指标表。返回 {'ALL': {...}, 'Rural': {...}, ...}。"""
        results = {}

        if title:
            print(f"\n{'=' * 74}")
            print(title)
            print('=' * 74)

        # --- 总览 ---
        print(f"\n{'域':<8}{'mIoU':>9}{'mDice':>9}{'aAcc':>9}{'oAcc':>9}{'有效像素':>14}")
        print('-' * 58)
        for key in ['ALL'] + C.DOMAINS:
            r = self.groups[key].compute()
            results[key] = r
            if r['total_pixels'] == 0:
                print(f"{key:<8}{'(无数据)':>9}")
                continue
            print(f"{key:<8}{r['miou']:>9.4f}{r['mdice']:>9.4f}"
                  f"{r['aacc']:>9.4f}{r['oacc']:>9.4f}{r['total_pixels']:>14,}")

        # --- 每类明细 ---
        print(f"\n{'类别':<8}", end='')
        for key in ['ALL'] + C.DOMAINS:
            print(f"{'IoU(' + key + ')':>13}", end='')
        print(f"{'占比(ALL)':>11}")
        print('-' * (8 + 13 * (1 + len(C.DOMAINS)) + 11))

        r_all = results['ALL']
        for c, name in enumerate(C.CLASS_NAMES):
            print(f"{name:<8}", end='')
            for key in ['ALL'] + C.DOMAINS:
                v = results[key]['iou'][c]
                print(f"{'  --':>13}" if np.isnan(v) else f"{v:>13.4f}", end='')
            share = r_all['gt_pixels'][c] / max(r_all['total_pixels'], 1)
            print(f"{share * 100:>10.2f}%")

        if not quiet:
            print(f"\n注：'--' 表示该类在对应域里完全没出现，IoU 无定义，未计入 mIoU。")
            print(f"    最后一列是该类在全部有效像素里的占比，据此判断类别是否稀有。")

        return results


# ============================================================================
# 可视化
# ============================================================================

def colorize(label_map, ignore_index=C.IGNORE_INDEX):
    """把 (H,W) 的类别索引图染成 RGB 图。忽略像素涂黑。"""
    label_map = np.asarray(label_map)
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    ignore = label_map == ignore_index
    valid = ~ignore & (label_map >= 0) & (label_map < len(PALETTE))
    out[valid] = PALETTE[label_map[valid]]
    out[ignore] = IGNORE_COLOR
    return out


def denormalize(img_chw, mean=C.NORM_MEAN, std=C.NORM_STD):
    """把归一化过的 (3,H,W) 还原成可显示的 (H,W,3) uint8。"""
    img = np.asarray(img_chw, dtype=np.float32).transpose(1, 2, 0)
    img = img * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def error_map(pred, target, ignore_index=C.IGNORE_INDEX):
    """
    错误图：预测正确的地方显示原类别颜色（调暗），错误的地方显示品红。
    忽略像素不参与评判，涂成黑色。
    """
    pred = np.asarray(pred)
    target = np.asarray(target)

    out = colorize(target, ignore_index)
    out = (out.astype(np.float32) * 0.45).astype(np.uint8)      # 正确的地方压暗

    ignore = target == ignore_index
    wrong = (~ignore) & (pred != target)
    out[wrong] = ERROR_COLOR
    out[ignore] = IGNORE_COLOR
    return out


_CJK_FONT_READY = False


def _setup_cjk_font():
    """
    让 matplotlib 显示中文。

    matplotlib 默认的 DejaVu Sans 没有中文字形，'原图'、'真值' 这类标题会渲染成
    一串方框 —— 而且它只在 stderr 抛一条 Glyph missing 警告，图照样正常存出来，
    不特意看警告根本发现不了。这里按优先级挑第一个系统里真实存在的字体。

    只做一次，之后全局生效。
    """
    global _CJK_FONT_READY
    if _CJK_FONT_READY:
        return

    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'SimSun'):
        if name in available:
            matplotlib.rcParams['font.sans-serif'] = [name]
            break
    else:
        print('[metrics] 警告：系统里没找到中文字体，可视化图的标题会显示成方框')

    # 中文字体的 Unicode 负号也是方框，换成 ASCII 减号
    matplotlib.rcParams['axes.unicode_minus'] = False
    _CJK_FONT_READY = True


def save_prediction_grid(samples, path, max_rows=4):
    """
    存一张四联图：原图 | 真值 | 预测 | 错误图。

    samples : list of dict，每项含 'image'(3,H,W 归一化过的)、'gt'、'pred'、'title'
    """
    import matplotlib
    matplotlib.use('Agg')            # 不弹窗，直接写文件
    import matplotlib.pyplot as plt

    _setup_cjk_font()

    rows = min(len(samples), max_rows)
    if rows == 0:
        return

    fig, axes = plt.subplots(rows, 4, figsize=(4 * 4, 4 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['原图', '真值 (GT)', '预测', '错误图 (品红=错)']

    for r in range(rows):
        s = samples[r]
        panels = [
            denormalize(s['image']),
            colorize(s['gt']),
            colorize(s['pred']),
            error_map(s['pred'], s['gt']),
        ]
        for c in range(4):
            ax = axes[r][c]
            ax.imshow(panels[c])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)
            if c == 0 and s.get('title'):
                ax.set_ylabel(s['title'], fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=90, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 自测
# ============================================================================

def _selftest():
    print("=" * 74)
    print("metrics 自测")
    print("=" * 74)

    m = SegMetrics()

    # 构造一个已知答案的小例子
    # 真值 2x2： 0 0 / 1 1 ；预测： 0 1 / 1 1
    # 混淆矩阵应为 cm[0,0]=1, cm[0,1]=1, cm[1,1]=2
    gt = np.array([[0, 0], [1, 1]])
    pd = np.array([[0, 1], [1, 1]])
    m.update(pd, gt)

    assert m.cm[0, 0] == 1 and m.cm[0, 1] == 1 and m.cm[1, 1] == 2, f"混淆矩阵不对\n{m.cm}"
    print(f"\n[1] 混淆矩阵累加正确")

    r = m.compute()
    # 类 0: tp=1, union = 2(真实) + 1(预测) - 1 = 2 -> IoU=0.5
    # 类 1: tp=2, union = 2 + 3 - 2 = 3 -> IoU=2/3
    assert abs(r['iou'][0] - 0.5) < 1e-6, r['iou']
    assert abs(r['iou'][1] - 2 / 3) < 1e-6, r['iou']
    print(f"[2] IoU 计算正确  类0={r['iou'][0]:.4f}  类1={r['iou'][1]:.4f}")

    # 没出现的类别应是 NaN，而不是 0（0 会拉低 mIoU，NaN 会被跳过）
    assert np.isnan(r['iou'][2]), "未出现的类别 IoU 应为 NaN"
    assert abs(r['miou'] - np.nanmean([0.5, 2 / 3])) < 1e-6
    print(f"[3] 未出现的类别记为 NaN 并跳过，mIoU={r['miou']:.4f}")

    # 忽略像素完全不参与
    m2 = SegMetrics()
    gt2 = np.array([[0, C.IGNORE_INDEX], [1, 1]])
    pd2 = np.array([[0, 1], [1, 1]])          # 忽略位置上预测错了，但不应被统计
    m2.update(pd2, gt2)
    assert m2.cm.sum() == 3, f"忽略像素应被排除，实际统计了 {m2.cm.sum()} 个"
    print(f"[4] 忽略像素被正确排除（统计了 {m2.cm.sum()} 个像素，而非 4 个）")

    # 分域容器
    dm = DomainMetrics()
    dm.update(pd, gt, 'Rural')
    dm.update(pd, gt, 'Urban')
    res = dm.report('自测报告', quiet=True)
    # 每次 update 是 2x2 = 4 个像素，Rural 一次 + Urban 一次
    assert res['Rural']['total_pixels'] == 4
    assert res['Urban']['total_pixels'] == 4
    assert res['ALL']['total_pixels'] == 8, res['ALL']['total_pixels']
    print(f"[5] 分域累加正确  有效像素：Rural=4 Urban=4 ALL=8")

    # 可视化不崩
    import tempfile
    img = np.random.rand(3, 32, 32).astype(np.float32)
    p = os.path.join(tempfile.gettempdir(), '_metrics_selftest.png')
    save_prediction_grid([dict(image=img, gt=gt, pred=pd, title='t')], p)
    assert os.path.exists(p) and os.path.getsize(p) > 0
    print(f"[6] 四联图能正常写出 -> {p}")

    # 配色和类别顺序必须逐位等于 LoveDA 官方。这两条断言是故意写死的：
    # 名字和颜色要和官方论文/榜单图肉眼对拍，任何人（包括以后的自己）
    # 想「顺手调好看一点」或者「把背景挪到最后更顺眼」都会在这里被拦下，
    # 而不是等到对不上图时才发现。
    #
    # 顺序错位一位这个坑本项目真踩过：整场「林地从恒 0 恢复到 0.3」的调查，
    # 追的其实是**裸地**（下标 4），因为当时把下标 4 叫成了林地。
    official_order = ['背景', '建筑', '道路', '水体', '裸地', '林地', '耕地']
    official_rgb = [(255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255),
                    (159, 129, 183), (0, 255, 0), (255, 195, 128)]

    assert list(C.CLASS_NAMES) == official_order, (
        f"类别顺序必须与 LoveDA 官方一致（背景在最前，不在最后）：现在 {C.CLASS_NAMES}")
    for i, name in enumerate(official_order):
        got = tuple(int(v) for v in PALETTE[i])
        assert got == official_rgb[i], (
            f"下标 {i} {name} 的配色不是 LoveDA 官方值："
            f"现在 {got}，应为 {official_rgb[i]}")
    print(f"[7] 类别顺序 + 7 个类别的配色逐位等于 LoveDA 官方定义")

    print("\n" + "=" * 74)
    print("全部通过")
    print("=" * 74)


if __name__ == '__main__':
    _selftest()
