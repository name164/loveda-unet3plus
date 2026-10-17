# -*- coding: utf-8 -*-
"""
UNet3+
论文：UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation (ICASSP 2020)

本文件对照论文作者官方实现 ZJUGiveLab/UNet-Version 的 models/UNet_3Plus.py 与
models/layers.py 编写，结构逐项对齐。与官方实现的差异只有下面四处，都是有意为之：

  1. 编码器通道由 [64,128,256,512,1024] 减半为 [32,64,128,256,512]（CPU 算力所限）。
     CatChannels / UpChannels 会自动跟随（官方定义为 filters[0] 与 CatChannels*5）。
  2. 输出不做 sigmoid，直接返回 logits —— 官方是二分类用的 sigmoid，
     我们是 7 类多分类，softmax 在损失函数内部做。
  3. 删除了 CGM 模块（理由见文件末尾「关于 CGM」）。
  4. Conv 用 bias=False（官方用默认的 bias=True）。BN 紧随其后，bias 会被 BN 减掉，
     留着纯属多余参数，去掉不改变任何行为。

-------------------------------------------------------------------------------
结构速览（以 512x512 输入、通道减半后的配置为例）
-------------------------------------------------------------------------------

【编码器】5 个 stage，每个 stage 是「两层 3x3 Conv + BN + ReLU」，stage 之间 2x2 maxpool

    x    (3,   512, 512)
    h1   (32,  512, 512)   1/1      <-- conv1 = ConvBnReLU(3, 32)
    h2   (64,  256, 256)   1/2      <-- conv2 = ConvBnReLU(32, 64)
    h3   (128, 128, 128)   1/4      <-- conv3 = ConvBnReLU(64, 128)
    h4   (256, 64,  64)    1/8      <-- conv4 = ConvBnReLU(128, 256)
    hd5  (512, 32,  32)    1/16     <-- conv5 = ConvBnReLU(256, 512)  ← 瓶颈

【解码器】全尺度跳跃连接，这是 UNet3+ 的核心

    解码节点只有 4 个（hd4/hd3/hd2/hd1），加上瓶颈 hd5 一共 5 个尺度。
    每个解码节点恒定接收 5 路输入，每路先对齐到该节点的分辨率，再各自用
    3x3 卷积压到 CatChannels(32) 通道，concat 成 UpChannels(160)，最后过一层
    3x3 卷积融合。对齐方式：

        hd4 (1/8)  <- h1↓8, h2↓4, h3↓2, h4 同尺度, hd5↑2
        hd3 (1/4)  <- h1↓4, h2↓2, h3 同尺度, hd4↑2, hd5↑4
        hd2 (1/2)  <- h1↓2, h2 同尺度, hd3↑2, hd4↑4, hd5↑8
        hd1 (1/1)  <- h1 同尺度, hd2↑2, hd3↑4, hd4↑8, hd5↑16

    ↓ 是 MaxPool2d(k, ceil_mode=True)，↑ 是双线性插值。

【深监督】5 个侧输出头，全部上采样回全分辨率

    d1 = outconv1(hd1)                    (7, 512, 512)   来自解码节点
    d2 = up2  (outconv2(hd2))             (7, 512, 512)   来自解码节点
    d3 = up4  (outconv3(hd3))             (7, 512, 512)   来自解码节点
    d4 = up8  (outconv4(hd4))             (7, 512, 512)   来自解码节点
    d5 = up16 (outconv5(hd5))             (7, 512, 512)   来自编码器瓶颈 ← 易错点

    注意 outconv5 接的是编码器瓶颈 hd5，输入通道是 ENC_FILTERS[4]，不是 UpChannels。
    这 5 路是并列的，官方不求和、不融合，训练时各算一次损失再相加。

用 inspect_model.py 可以把上面每一条都打印出来验证。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# ============================================================================
# 基础组件
# ============================================================================

def _num_groups(ch):
    """给 GroupNorm 选一个能整除 ch 的组数，优先 8。"""
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def _make_norm(ch, kind):
    """按配置造归一化层。官方统一用 BatchNorm2d。"""
    if kind == 'bn':
        return nn.BatchNorm2d(ch)
    if kind == 'gn':
        return nn.GroupNorm(_num_groups(ch), ch)
    raise ValueError(f"未知的 NORM_LAYER: {kind}")


class ConvBnReLU(nn.Module):
    """
    对应官方的 unetConv2：n 层 (3x3 Conv -> BN -> ReLU)。

    参数
    ----
    n : 层数。官方默认 n=2，编码器每个 stage 就是两层。
        解码器里那个「压通道」的侧支路只有一层，传 n=1。
    """

    def __init__(self, in_ch, out_ch, n=2, norm_kind='bn'):
        super().__init__()
        layers = []
        for i in range(n):
            layers += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch,
                          kernel_size=3, stride=1, padding=1, bias=False),
                _make_norm(out_ch, norm_kind),
                nn.ReLU(inplace=True),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderNode(nn.Module):
    """
    一个解码节点（官方 forward 里那一大坨 h1_PT_hd4 / h4_Cat_hd4 / hd5_UT_hd4 ...）。

    官方写法是把 4 个解码节点 x 5 路输入 = 20 条侧支路全部手写展开，
    可读性很差。这里改成：对齐分工在 UNet3Plus.forward 里显式做（看代码就知道
    哪一路从哪来），本类只负责「各自压通道 -> concat -> 融合」这一段。

    参数
    ----
    in_channels : 长度 5 的列表，5 路输入**对齐之后**各自的通道数
    cat_channels: 每路压到多少通道（官方 CatChannels = filters[0]）
    up_channels : concat 后融合卷积的通道数（官方 UpChannels = cat_channels * 5）
    """

    def __init__(self, in_channels, cat_channels, up_channels, norm_kind='bn'):
        super().__init__()
        assert len(in_channels) == 5, \
            f"解码节点必须恰好 5 路输入，收到 {len(in_channels)} 路"

        # 每路一个 3x3 卷积压通道（官方用的是 3x3，不是 1x1）
        self.laterals = nn.ModuleList([
            ConvBnReLU(ch, cat_channels, n=1, norm_kind=norm_kind)
            for ch in in_channels
        ])

        # 5 路 concat 之后的融合卷积
        self.fuse = ConvBnReLU(up_channels, up_channels, n=1, norm_kind=norm_kind)

    def forward(self, aligned_inputs):
        """
        aligned_inputs : 长度 5 的列表，5 路张量已经对齐到同一分辨率
        """
        assert len(aligned_inputs) == 5

        # 对齐没做对的话，这里会给出比 cat 报错清楚得多的提示
        hw = aligned_inputs[0].shape[-2:]
        for i, t in enumerate(aligned_inputs):
            if t.shape[-2:] != hw:
                raise RuntimeError(
                    f"解码节点第 {i} 路输入分辨率 {tuple(t.shape[-2:])} "
                    f"与第 0 路 {tuple(hw)} 不一致，检查上/下采样倍数"
                )

        feats = [lat(t) for lat, t in zip(self.laterals, aligned_inputs)]
        # 5 路各 cat_channels 通道 -> up_channels
        return self.fuse(torch.cat(feats, dim=1))


# ============================================================================
# 主模型
# ============================================================================

class UNet3Plus(nn.Module):
    """
    UNet3+（多分类版，无 CGM，带深监督）。

    forward 返回长度 5 的列表 [d1, d2, d3, d4, d5]，全部是**全分辨率 logits**：

        d1 : 最细，来自解码节点 hd1，也是推理时的主输出
        d2 : 来自 hd2，上采样 2 倍
        d3 : 来自 hd3，上采样 4 倍
        d4 : 来自 hd4，上采样 8 倍
        d5 : 来自编码器瓶颈 hd5，上采样 16 倍

    这与官方 UNet_3Plus_DeepSup.forward 的返回形式一致（官方外面多套了个 sigmoid）。
    """

    def __init__(self,
                 in_channels=3,
                 num_classes=C.NUM_CLASSES,
                 filters=None,
                 norm_kind=None):
        super().__init__()

        filters = filters or C.ENC_FILTERS
        norm_kind = norm_kind or C.NORM_LAYER
        assert len(filters) == 5, "编码器必须是 5 个 stage"

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.filters = list(filters)

        # 官方定义，不要手改
        self.cat_channels = filters[0]                 # 每路压到多少通道
        self.up_channels = self.cat_channels * 5       # 融合后通道数（恒等于 5 路之和）

        # ---------------------------------------------------------------
        # 编码器：5 个 stage，stage 内两层 3x3，stage 间 2x2 maxpool
        # ---------------------------------------------------------------
        self.conv1 = ConvBnReLU(in_channels, filters[0], n=2, norm_kind=norm_kind)
        self.conv2 = ConvBnReLU(filters[0], filters[1], n=2, norm_kind=norm_kind)
        self.conv3 = ConvBnReLU(filters[1], filters[2], n=2, norm_kind=norm_kind)
        self.conv4 = ConvBnReLU(filters[2], filters[3], n=2, norm_kind=norm_kind)
        self.conv5 = ConvBnReLU(filters[3], filters[4], n=2, norm_kind=norm_kind)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # ---------------------------------------------------------------
        # 解码器的对齐算子（无参数，纯几何操作）
        # 官方用 MaxPool2d(k, k, ceil_mode=True) 和 Upsample(scale_factor, bilinear)
        # ---------------------------------------------------------------
        self.down2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.down4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.down8 = nn.MaxPool2d(8, 8, ceil_mode=True)

        # align_corners 显式写 False：与官方默认行为一致，同时避免新版 torch 的告警
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

        # ---------------------------------------------------------------
        # 4 个解码节点。每个的 in_channels 就是它 5 路输入对齐后的通道数。
        # 对照上面的「结构速览」，这 4 行是把那张表翻译成代码。
        # ---------------------------------------------------------------
        #                h1↓8          h2↓4          h3↓2          h4            hd5↑2
        self.dec4 = DecoderNode(
            [filters[0], filters[1], filters[2], filters[3], filters[4]],
            self.cat_channels, self.up_channels, norm_kind)

        #                h1↓4          h2↓2          h3            hd4↑2         hd5↑4
        self.dec3 = DecoderNode(
            [filters[0], filters[1], filters[2], self.up_channels, filters[4]],
            self.cat_channels, self.up_channels, norm_kind)

        #                h1↓2          h2            hd3↑2         hd4↑4         hd5↑8
        self.dec2 = DecoderNode(
            [filters[0], filters[1], self.up_channels, self.up_channels, filters[4]],
            self.cat_channels, self.up_channels, norm_kind)

        #                h1            hd2↑2         hd3↑4         hd4↑8         hd5↑16
        self.dec1 = DecoderNode(
            [filters[0], self.up_channels, self.up_channels, self.up_channels, filters[4]],
            self.cat_channels, self.up_channels, norm_kind)

        # ---------------------------------------------------------------
        # 深监督：5 个侧输出头
        # d1~d4 接解码节点（UpChannels 通道），d5 接编码器瓶颈（filters[4] 通道）
        # ---------------------------------------------------------------
        self.outconv1 = nn.Conv2d(self.up_channels, num_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.up_channels, num_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.up_channels, num_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.up_channels, num_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], num_classes, 3, padding=1)

        # d1 本来就在全分辨率，不需要上采样
        self.upscore2 = self.up2
        self.upscore3 = self.up4
        self.upscore4 = self.up8
        self.upscore5 = self.up16

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """
        Kaiming 初始化卷积，BN 置为恒等映射。

        但 5 个 softmax 头（outconv1..5）例外，用小初始化。
        它们输出直接进 softmax，若照着 fan_out 算：fan_out = 7*9 = 63，
        std 高达 0.178（softmax 头的惯例是 0.01，差 18 倍），实测初始 maxprob
        就有 0.78 —— 从第 0 步起 softmax 就接近 one-hot，Dice 梯度被 p(1-p)
        削弱，后期 logit 会一路涨到几百彻底饱和。而饱和正是"某个类死掉"的
        入口：饱和后 Dice 对该类的梯度下溢为 0，再也拉不回来。
        """
        heads = {id(m) for m in (self.outconv1, self.outconv2, self.outconv3,
                                 self.outconv4, self.outconv5)}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if id(m) in heads:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    def forward(self, x):
        # ================= 编码器 =================
        h1 = self.conv1(x)                              # (B,  32, 512, 512)  1/1
        h2 = self.conv2(self.maxpool1(h1))              # (B,  64, 256, 256)  1/2
        h3 = self.conv3(self.maxpool2(h2))              # (B, 128, 128, 128)  1/4
        h4 = self.conv4(self.maxpool3(h3))              # (B, 256,  64,  64)  1/8

        h5 = self.maxpool4(h4)                          # 池化结果
        hd5 = self.conv5(h5)                            # (B, 512,  32,  32)  1/16 ← 瓶颈

        # ============ 解码器：全尺度跳跃连接 ============
        # 每个节点的 5 路输入在这里显式写出来，对照上面的「结构速览」表读

        hd4 = self.dec4([                              # 目标分辨率 1/8
            self.down8(h1),                            #   h1  -> ↓8
            self.down4(h2),                            #   h2  -> ↓4
            self.down2(h3),                            #   h3  -> ↓2
            h4,                                        #   同尺度
            self.up2(hd5),                             #   瓶颈 -> ↑2
        ])

        hd3 = self.dec3([                              # 目标分辨率 1/4
            self.down4(h1),                            #   h1  -> ↓4
            self.down2(h2),                            #   h2  -> ↓2
            h3,                                        #   同尺度
            self.up2(hd4),                             #   hd4 -> ↑2
            self.up4(hd5),                             #   瓶颈 -> ↑4
        ])

        hd2 = self.dec2([                              # 目标分辨率 1/2
            self.down2(h1),                            #   h1  -> ↓2
            h2,                                        #   同尺度
            self.up2(hd3),                             #   hd3 -> ↑2
            self.up4(hd4),                             #   hd4 -> ↑4
            self.up8(hd5),                             #   瓶颈 -> ↑8
        ])

        hd1 = self.dec1([                              # 目标分辨率 1/1
            h1,                                        #   同尺度
            self.up2(hd2),                             #   hd2 -> ↑2
            self.up4(hd3),                             #   hd3 -> ↑4
            self.up8(hd4),                             #   hd4 -> ↑8
            self.up16(hd5),                            #   瓶颈 -> ↑16
        ])

        # ============ 深监督的 5 个侧输出（全部回到全分辨率）============
        d1 = self.outconv1(hd1)                        # (B, 7, 512, 512) 已是全分辨率
        d2 = self.upscore2(self.outconv2(hd2))         # (B, 7, 512, 512)
        d3 = self.upscore3(self.outconv3(hd3))         # (B, 7, 512, 512)
        d4 = self.upscore4(self.outconv4(hd4))         # (B, 7, 512, 512)
        d5 = self.upscore5(self.outconv5(hd5))         # (B, 7, 512, 512) ← 接瓶颈，不是解码节点

        # 注意：没有 sigmoid。多分类用 softmax，在损失里做。
        return [d1, d2, d3, d4, d5]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x):
        """
        推理：把 5 路侧输出合成一张类别预测图。

        返回 (B, H, W) 的 int64 类别索引。
        用哪一路由 config.INFER_FUSE 决定：'d1' 或 'mean'。
        """
        outs = self.forward(x)

        if C.INFER_FUSE == 'd1':
            logits = outs[0]
        elif C.INFER_FUSE == 'mean':
            # 5 路 logits 取平均。注意这是 logits 层面的平均，
            # 等价于概率的几何平均（差一个 softmax 归一化常数），是可用的集成方式
            logits = torch.stack(outs, dim=0).mean(dim=0)
        else:
            raise ValueError(f"未知的 INFER_FUSE: {C.INFER_FUSE}")

        return logits.argmax(dim=1)


# ============================================================================
# 工具
# ============================================================================

def count_parameters(model, verbose=False):
    """统计参数量。verbose=True 时顺便列出参数量最大的几个子模块。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        rows = []
        for name, mod in model.named_children():
            n = sum(p.numel() for p in mod.parameters())
            rows.append((n, name))
        rows.sort(reverse=True)
        print("  子模块参数量 Top:")
        for n, name in rows:
            print(f"    {name:12s} {n:>10,}  ({n / total * 100:5.1f}%)")

    return total, trainable


def build_model(**kwargs):
    """按 config 造一个 UNet3+。train.py / eval.py / inspect_model.py 统一走这里。"""
    model = UNet3Plus(
        in_channels=3,
        num_classes=C.NUM_CLASSES,
        filters=C.ENC_FILTERS,
        norm_kind=C.NORM_LAYER,
        **kwargs,
    )
    return model


# ============================================================================
# 关于 CGM（这一段是注释，不是代码）
# ============================================================================
#
# CGM = Classification Guided Module，官方只在 UNet_3Plus_DeepSup_CGM 里提供。
# 它的实现是：
#
#     self.cls = nn.Sequential(
#         nn.Dropout(p=0.5),
#         nn.Conv2d(filters[4], 2, 1),      # ← 输出通道硬编码为 2
#         nn.AdaptiveMaxPool2d(1),
#         nn.Sigmoid())
#     ...
#     cls_branch = self.cls(hd5).squeeze(3).squeeze(2)   # (B, 2)
#     cls_branch_max = cls_branch.argmax(dim=1).float()  # (B,)
#     d1 = self.dotProduct(d1, cls_branch_max)           # d2..d5 同样
#
# 去掉它是安全的，而且对于 7 分类任务是必要的：
#
#   1. 安全 —— CGM 只作用在输出端（d1..d5 各乘一次分类门控）。编码器、4 个解码
#      节点、5 个深监督头的主路径与梯度路径逐行未变。删掉它，训练路径完全一致。
#
#   2. 必要 —— self.cls 的输出通道写死为 2，是二分类「前景/背景」的门控。用
#      argmax 得到一个标量去乘整张预测图，对多分类会直接把整类预测清零。
#      在 7 类 LoveDA 上保留它只会破坏结果。
#      官方 README 也标注了 CGM 仅在 OUTPUT.CLASSES = 1 时可训练。
#
# 所以本文件直接实现不带 CGM 的版本，把 outconv1..5 的输出原样返回。
