# -*- coding: utf-8 -*-
"""
数据集：划分生成 + 读取 + 增强 + DataLoader

数据布局
--------
    LDA/Train/
      Rural/images_png/0.png      Rural/masks_png/0.png      (1366 对)
      Urban/images_png/1366.png   Urban/masks_png/1366.png   (1156 对)

    图像 1024x1024 uint8 RGB；掩码 1024x1024 uint8 单通道，取值 0..7。
    0 = 未标注（忽略），1..7 = 背景/建筑/道路/水体/裸地/林地/耕地（LoveDA 官方顺序）。

标签重映射
----------
    原始 0      -> 255  (config.IGNORE_INDEX，损失里忽略)
    原始 1..7   -> 0..6 (网络输出的 7 个类别)
用 config.LABEL_LUT 查表完成，比链式 np.where 快很多。

划分策略
--------
按域**各自独立**做 7:1:2 随机划分，固定 seed。这样 train/val/test 里
都同时含乡村和城市，不会出现「验证集全是城市」的情况。
划分结果落盘到 splits.json，之后训练只读取，**绝不重新划分**。

    Rural 1366 -> train 956 / val 136 / test 274
    Urban 1156 -> train 809 / val 115 / test 232
    合计 2522 -> train 1765 / val 251 / test 506
    （用 int() 向下取整，所以 val/test 的实际张数会比 7:1:2 的比例略偏）
"""

import json
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config as C


# ============================================================================
# 一、划分
# ============================================================================

def _list_pairs(domain):
    """列出某个域下所有 (相对路径, 图像绝对路径, 掩码绝对路径)，并校验图文配对。"""
    idir = os.path.join(C.DATA_ROOT, domain, 'images_png')
    mdir = os.path.join(C.DATA_ROOT, domain, 'masks_png')

    if not os.path.isdir(idir):
        raise FileNotFoundError(f"找不到图像目录: {idir}")

    names = sorted(os.listdir(idir))
    if not names:
        raise FileNotFoundError(f"{idir} 是空的")

    pairs = []
    missing = []
    for n in names:
        ip = os.path.join(idir, n)
        mp = os.path.join(mdir, n)
        if not os.path.exists(mp):
            missing.append(n)
            continue
        pairs.append((f'{domain}/{n}', ip, mp))

    if missing:
        raise RuntimeError(f"{domain} 下有 {len(missing)} 张图没有对应掩码，"
                           f"例如 {missing[:5]}")
    return pairs


def build_splits(force=False, verbose=True):
    """
    生成（或读取）划分清单。

    已存在 splits.json 且 force=False 时直接读，保证整个训练周期里划分不变。
    """
    if os.path.exists(C.SPLIT_FILE) and not force:
        with open(C.SPLIT_FILE, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        if verbose:
            print(f"读取已有划分 {C.SPLIT_FILE}")
            _print_split_stats(splits)
        return splits

    rng = random.Random(C.SEED)
    ratios = C.SPLIT_RATIO
    assert abs(sum(ratios) - 1.0) < 1e-6, "SPLIT_RATIO 必须和为 1"

    splits = {'train': [], 'val': [], 'test': []}
    per_domain = {}

    for domain in C.DOMAINS:
        pairs = _list_pairs(domain)
        keys = [p[0] for p in pairs]          # 只对键做划分，路径可重建
        rng.shuffle(keys)

        n = len(keys)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        parts = {
            'train': keys[:n_train],
            'val': keys[n_train:n_train + n_val],
            'test': keys[n_train + n_val:],
        }
        per_domain[domain] = {k: len(v) for k, v in parts.items()}
        for k, v in parts.items():
            splits[k].extend(v)

    # ---------------- 校验 ----------------
    _validate(splits, per_domain, verbose)

    with open(C.SPLIT_FILE, 'w', encoding='utf-8') as f:
        json.dump(splits, f, ensure_ascii=False, indent=1)

    if verbose:
        print(f"划分已写入 {C.SPLIT_FILE}（之后训练只读取，不会重划）")
        _print_split_stats(splits)
    return splits


def _validate(splits, per_domain, verbose):
    """三集合无交集、并集等于全集 —— 划分错了后面全白干，必须检查。"""
    sets = {k: set(v) for k, v in splits.items()}

    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        dup = sets[a] & sets[b]
        if dup:
            raise RuntimeError(f"划分有泄漏：{a} 与 {b} 交集 {len(dup)} 个，例如 {list(dup)[:3]}")

    all_keys = set()
    for domain in C.DOMAINS:
        all_keys |= {p[0] for p in _list_pairs(domain)}

    union = sets['train'] | sets['val'] | sets['test']
    if union != all_keys:
        missing = all_keys - union
        extra = union - all_keys
        raise RuntimeError(f"划分与全集不一致：漏了 {len(missing)} 个，多了 {len(extra)} 个")

    if verbose:
        print("划分校验通过：无交集、无遗漏、图文配对齐全")


def _print_split_stats(splits):
    print(f"\n{'域':<8}{'train':>8}{'val':>8}{'test':>8}{'合计':>8}")
    print('-' * 40)
    for domain in C.DOMAINS:
        c = [sum(1 for k in splits[s] if k.startswith(domain + '/'))
             for s in ['train', 'val', 'test']]
        print(f"{domain:<8}{c[0]:>8}{c[1]:>8}{c[2]:>8}{sum(c):>8}")
    c = [len(splits[s]) for s in ['train', 'val', 'test']]
    print('-' * 40)
    print(f"{'合计':<8}{c[0]:>8}{c[1]:>8}{c[2]:>8}{sum(c):>8}\n")


def key_to_paths(key):
    """'Rural/0.png' -> (图像路径, 掩码路径)"""
    domain, name = key.split('/', 1)
    return (os.path.join(C.DATA_ROOT, domain, 'images_png', name),
            os.path.join(C.DATA_ROOT, domain, 'masks_png', name))


# ============================================================================
# 二、增强
# ============================================================================

def build_transform(mode):
    """
    mode:
      'train' -> 随机裁剪 512 + 几何/色彩增强
      'val'   -> 中心裁剪 512（快，用于每轮跟踪趋势）
      'full'  -> 不裁剪，返回完整 1024x1024（最终滑窗测试用）

    几何增强只用翻转和 90 度整数倍旋转：俯视遥感图没有方向先验，
    而且这两种变换**不引入任何插值**，掩码边缘不会糊。
    不做 ShiftScaleRotate / Affine —— 它们会破坏掩码边界。
    """
    mean, std = C.NORM_MEAN, C.NORM_STD

    if mode == 'train':
        ops = [
            A.RandomCrop(C.CROP_SIZE, C.CROP_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
            A.Normalize(mean=mean, std=std),
        ]
    elif mode == 'val':
        ops = [
            A.CenterCrop(C.CROP_SIZE, C.CROP_SIZE),
            A.Normalize(mean=mean, std=std),
        ]
    elif mode == 'full':
        ops = [
            A.Normalize(mean=mean, std=std),
        ]
    else:
        raise ValueError(f"未知的 mode: {mode}")

    # mask_interpolation 在 albumentations 2.x 里是 Compose 级别的参数，
    # 显式写明用最近邻，不依赖默认值
    return A.Compose(ops, mask_interpolation=cv2.INTER_NEAREST)


# ============================================================================
# 三、Dataset
# ============================================================================

class LoveDADataset(Dataset):
    """LoveDA 数据集。返回 (image[3,H,W] float32, mask[H,W] int64, key)。"""

    def __init__(self, keys, mode='train', transform=None):
        self.keys = list(keys)
        self.mode = mode
        self.transform = transform or build_transform(mode)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        ipath, mpath = key_to_paths(key)

        # cv2 读进来是 BGR，转成 RGB 再送进网络
        img = cv2.imread(ipath, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"读不到图像: {ipath}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"读不到掩码: {mpath}")

        # 标签重映射：0 -> 255(忽略)，1..7 -> 0..6
        mask = C.LABEL_LUT[mask]

        out = self.transform(image=img, mask=mask)
        img = out['image']        # (H, W, 3) float32，已归一化
        mask = out['mask']        # (H, W) 值域 {0..6, 255}

        # 手动转成 CHW，不用 ToTensorV2，少一层依赖，也看得更清楚
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()

        return img, mask, key

    def domain_of(self, idx):
        return self.keys[idx].split('/', 1)[0]


# ============================================================================
# 四、DataLoader
# ============================================================================

def get_keys(split, smoke=None):
    """取某个 split 的 key 列表。smoke 给定时只取前 N 个（冒烟测试用）。"""
    splits = build_splits(verbose=False)
    keys = list(splits[split])
    if smoke:
        # 冒烟时也要保证两个域都有
        rng = random.Random(C.SEED)
        by_domain = {d: [k for k in keys if k.startswith(d + '/')] for d in C.DOMAINS}
        picked = []
        for d in C.DOMAINS:
            rng.shuffle(by_domain[d])
            picked.extend(by_domain[d][:max(1, smoke // len(C.DOMAINS))])
        keys = picked
    return keys


def build_loader(split, mode, batch_size=None, shuffle=None, smoke=None):
    keys = get_keys(split, smoke=smoke)
    ds = LoveDADataset(keys, mode=mode)

    if shuffle is None:
        shuffle = (mode == 'train')

    # 训练时才开多进程和持久 worker；full 模式（滑窗测试）单进程即可，省内存
    num_workers = C.NUM_WORKERS if mode == 'train' else 0

    kw = {}
    if num_workers > 0:
        kw['persistent_workers'] = C.PERSISTENT_WORKERS
        kw['prefetch_factor'] = 2

    return DataLoader(
        ds,
        batch_size=batch_size or C.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=(mode == 'train'),   # 丢掉最后一个不满的 batch，BN 更稳
        pin_memory=False,              # CPU 训练，不需要锁页内存
        **kw,
    )


# ============================================================================
# 五、入口：生成划分 + 增强效果预览
# ============================================================================

def _preview_augmentation(out_path, n=3):
    """存一张增强效果对比图，确认掩码和图像变换严格同步。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from metrics import colorize, denormalize

    keys = get_keys('train')[:n]
    ds = LoveDADataset(keys, mode='train')

    fig, axes = plt.subplots(n, 3, figsize=(11, 3.6 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    # 同一个样本连取两次，展示增强的随机性
    val_ds = LoveDADataset(keys, mode='val')

    for r in range(n):
        img, mask, key = ds[r]
        vimg, vmask, _ = val_ds[r]

        panels = [
            (denormalize(img.numpy()), f'训练增强: {key}'),
            (colorize(mask.numpy()), '对应掩码'),
            (denormalize(vimg.numpy()), '验证(中心裁剪)'),
        ]
        for c, (panel, title) in enumerate(panels):
            axes[r][c].imshow(panel)
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(title, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=85, bbox_inches='tight')
    plt.close(fig)
    print(f"增强预览已写出 -> {out_path}")


def _selftest():
    print("=" * 74)
    print("dataset 自测")
    print("=" * 74)

    splits = build_splits()

    print("\n[1] 取一个 batch 检查形状与取值范围")
    loader = build_loader('train', 'train', batch_size=2)
    img, mask, keys = next(iter(loader))
    print(f"    image {tuple(img.shape)} {img.dtype}  "
          f"值域 [{img.min():.3f}, {img.max():.3f}]")
    print(f"    mask  {tuple(mask.shape)} {mask.dtype}  "
          f"出现过的取值 {sorted(set(mask.flatten().tolist()))}")
    assert img.shape == (2, 3, C.CROP_SIZE, C.CROP_SIZE), img.shape
    assert mask.shape == (2, C.CROP_SIZE, C.CROP_SIZE), mask.shape

    uniq = set(mask.flatten().tolist())
    assert uniq <= set(range(C.NUM_CLASSES)) | {C.IGNORE_INDEX}, f"标签取值异常: {uniq}"
    print(f"    OK：标签只含 0..{C.NUM_CLASSES - 1} 和 {C.IGNORE_INDEX}(忽略)")

    print("\n[2] 三种 mode 的输出尺寸")
    for mode, expect in [('train', C.CROP_SIZE), ('val', C.CROP_SIZE), ('full', 1024)]:
        d = LoveDADataset(get_keys('val')[:2], mode=mode)
        i, m, k = d[0]
        print(f"    {mode:<6} image {tuple(i.shape)}  mask {tuple(m.shape)}")
        assert i.shape[1] == expect and i.shape[2] == expect, f"{mode} 尺寸不对"

    print("\n[3] 忽略像素确实存在（否则 DiceLoss 的 ignore 逻辑没被用上）")
    found = 0
    d = LoveDADataset(get_keys('train')[:20], mode='full')
    for i in range(len(d)):
        _, m, _ = d[i]
        if (m == C.IGNORE_INDEX).any():
            found += 1
    print(f"    前 20 张里有 {found} 张含忽略像素")
    assert found > 0, "居然一张含忽略像素的都没有，检查 LABEL_LUT"

    print("\n[4] 划分统计")
    _print_split_stats(splits)

    print("[5] 增强预览")
    _preview_augmentation(os.path.join(C.OUT_DIR, 'aug_preview.png'))

    print("\n" + "=" * 74)
    print("全部通过")
    print("=" * 74)


if __name__ == '__main__':
    _selftest()
