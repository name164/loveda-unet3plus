# -*- coding: utf-8 -*-
"""
损失函数：多分类 Dice + 深监督加权

设计要点
--------
1. **软 Dice**：对 softmax 后的概率和 one-hot 标签逐类计算
       dice_c = (2 * Σ p_c * g_c + eps) / (Σ p_c + Σ g_c + eps)
   然后对类别取平均，loss = 1 - mean(dice_c)。

2. **忽略像素**（mask 里值为 255 的位置，对应原始的 0=未标注）：
   在分子分母里**同时剔除**，不参与任何类别的统计。
   做法是把概率和 one-hot 都乘上 valid 掩码。

3. **深监督**：UNet3+ 有 5 路侧输出（d1..d5），每路都算一次 Dice，
   再按 config.DS_WEIGHTS 加权求和。这与官方「5 路各算损失再相加」的约定一致。

4. **空类别的处理**：某个 batch 里如果某类完全没出现，
   分子分母都是 0，dice = eps/eps = 1，该项损失为 0。
   这是标准行为，不会产生错误梯度，但也意味着
   **看 loss 数值本身判断不出稀有类的好坏**，必须看 metrics.py 的 per-class Dice。

5. **纯 Dice 的已知问题**：多分类下训练初期容易震荡。所以留了 CE_WEIGHT 开关，
   默认 0（纯 Dice），若 loss 不降或剧烈抖动，把它开到 0.1 即可，不用改代码。
   注意本文件里 CE 与 Dice **同口径（5 路求和）**，所以 0.1 不算小 —— 改口径
   之前 CE 取的是 mean，那时候写 0.5 的实际强度就等于现在写 0.1。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class MultiClassDiceLoss(nn.Module):
    """多分类软 Dice 损失（含忽略像素处理）。"""

    def __init__(self,
                 num_classes=C.NUM_CLASSES,
                 ignore_index=C.IGNORE_INDEX,
                 eps=C.DICE_EPS,
                 class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

        if class_weights is None:
            self.register_buffer('class_weights', torch.ones(num_classes))
        else:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            assert w.numel() == num_classes, "class_weights 长度必须等于类别数"
            # 归一化到均值 1，避免整体损失尺度随权重变化
            self.register_buffer('class_weights', w / w.mean())

    # ------------------------------------------------------------------
    def dice_per_class(self, logits, target):
        """
        logits : (B, C, H, W)  未过 softmax 的原始输出
        target : (B, H, W)     int64，取值 0..C-1，忽略像素为 ignore_index
        返回   : (C,) 每类的 dice 值
        """
        prob = logits.softmax(dim=1)                          # (B, C, H, W)

        valid = (target != self.ignore_index)                 # (B, H, W)
        v = valid.unsqueeze(1).to(prob.dtype)                 # (B, 1, H, W)

        # 忽略像素的概率清零，这样它们在分子分母里都不贡献
        prob = prob * v

        # 忽略像素的标签先随便填 0（one_hot 要求取值在 [0, C)），
        # 紧接着会被 v 清零，所以填什么无所谓
        tgt = target.clone()
        tgt[~valid] = 0
        onehot = F.one_hot(tgt, self.num_classes)             # (B, H, W, C)
        onehot = onehot.permute(0, 3, 1, 2).to(prob.dtype)    # (B, C, H, W)
        onehot = onehot * v

        dims = (0, 2, 3)
        inter = (prob * onehot).sum(dim=dims)                 # (C,)
        denom = prob.sum(dim=dims) + onehot.sum(dim=dims)     # (C,)

        return (2.0 * inter + self.eps) / (denom + self.eps)

    # ------------------------------------------------------------------
    def forward_single(self, logits, target):
        dice = self.dice_per_class(logits, target)
        return (1.0 - dice * self.class_weights).mean()

    def forward(self, outputs, target):
        """
        outputs : list[Tensor]，UNet3+ 的 5 路侧输出 [d1..d5]；
                  也可以直接传单个 Tensor
        target  : (B, H, W) int64 标签

        返回 (total_loss, dict)，dict 里是各路损失，方便记录到日志。
        """
        if isinstance(outputs, (list, tuple)):
            outs = list(outputs)
        else:
            outs = [outputs]

        parts = {}
        total = 0.0

        for i, logit in enumerate(outs):
            d_loss = self.forward_single(logit, target)

            if C.DS_LOSS:
                w = C.DS_WEIGHTS[i] if i < len(C.DS_WEIGHTS) else 1.0
            else:
                # 关掉深监督损失：只在主输出 d1 上算
                w = 1.0 if i == 0 else 0.0

            if w == 0.0:
                continue

            total = total + w * d_loss
            parts[f'dice_d{i + 1}'] = float(d_loss.detach())

        # 可选的交叉熵辅助项（默认 weight=0，不产生任何开销）
        if C.CE_WEIGHT > 0:
            # 这里必须用 sum，不能用 mean —— 上面的 Dice 是 5 路**求和**，
            # 若 CE 取平均则两者基准差 5 倍，CE_WEIGHT=0.5 的真实影响力只有 0.1，
            # 而启动信息和 config 注释都会显示成 0.5。口径必须一致。
            if (target != self.ignore_index).any():
                ce_parts = []
                for i, logit in enumerate(outs):
                    if not C.DS_LOSS and i > 0:
                        continue
                    ce_parts.append(F.cross_entropy(
                        logit, target, ignore_index=self.ignore_index))
                ce = torch.stack(ce_parts).sum()
            else:
                # 整批都是忽略像素时 CE 无定义：cross_entropy 的 mean 是 0/0 = NaN，
                # 会把整个 loss 污染成 NaN，反传后权重全毁。这种 batch 本来也
                # 没有任何可学的信息（Dice 那边自然给出 0），直接跳过 CE。
                ce = outs[0].new_zeros(())
            total = total + C.CE_WEIGHT * ce
            parts['ce'] = float(ce.detach())

        parts['total'] = float(total.detach())
        return total, parts


class CombinedLoss(nn.Module):
    """
    把「对模型的 5 路输出算损失」这件事包一层。

    train.py 里只需要：
        criterion = CombinedLoss()
        outs = model(x)                  # [d1..d5]
        loss, parts = criterion(outs, y)
    """

    def __init__(self):
        super().__init__()
        self.dice = MultiClassDiceLoss(
            num_classes=C.NUM_CLASSES,
            ignore_index=C.IGNORE_INDEX,
            eps=C.DICE_EPS,
            class_weights=C.CLASS_WEIGHTS,
        )

    def forward(self, outputs, target):
        return self.dice(outputs, target)


# ============================================================================
# 自测：直接 python loss.py 跑
# ============================================================================

def _selftest():
    import config as C
    torch.manual_seed(0)
    B, S, Cc = 2, 32, C.NUM_CLASSES
    crit = CombinedLoss()

    target = torch.randint(0, Cc, (B, S, S))

    print("=" * 70)
    print("DiceLoss 自测")
    print("=" * 70)

    # --- 1. 完美预测：把正确类别打到很高的 logit ---
    perfect = torch.full((B, Cc, S, S), -20.0)
    perfect.scatter_(1, target.unsqueeze(1), 20.0)
    loss_perfect, _ = crit([perfect] * 5, target)
    print(f"\n[1] 完美预测            loss = {loss_perfect.item():.6f}   （期望接近 0）")
    assert loss_perfect.item() < 1e-3, "完美预测的 loss 应该接近 0"

    # --- 2. 完全错误：把 logit 全打到某一个错类上 ---
    # 注意：单路输出的 loss 在 [0, 1] 区间，但 5 路深监督各权重 1.0 相加，
    # 所以这里的总 loss 期望是 5 左右，不是 1。
    n_out = len(C.DS_WEIGHTS)
    wrong = torch.full((B, Cc, S, S), 20.0)
    wrong.scatter_(1, target.unsqueeze(1), -20.0)
    loss_wrong, parts_wrong = crit([wrong] * 5, target)
    per_out = parts_wrong['dice_d1']
    print(f"[2] 完全错误预测        loss = {loss_wrong.item():.6f}   "
          f"（单路 Dice {per_out:.4f} x{n_out} 路"
          + (f"，另加 {C.CE_WEIGHT} x CE {parts_wrong['ce']:.4f}" if C.CE_WEIGHT > 0 else "")
          + "）")
    assert 0.9 < per_out < 1.0, "单路完全错误的 loss 应该接近 1"

    # 总 loss 必须严格等于「5 路 Dice 之和 + CE_WEIGHT x 5 路 CE 之和」
    dice_sum = per_out * n_out
    if C.CE_WEIGHT > 0:
        ce_val = parts_wrong['ce']
        # 独立算一遍，确认 CE 走的是「5 路求和」而不是「5 路平均」——
        # 用 mean 的话 CE 的真实权重只有标称值的 1/5，显示值和实际影响力对不上
        ce_manual = sum(F.cross_entropy(wrong, target, ignore_index=C.IGNORE_INDEX)
                        for _ in range(n_out))
        assert abs(ce_val - ce_manual.item()) < 1e-3, \
            f"CE 应该是 5 路之和({ce_manual.item():.4f})，实际 {ce_val:.4f}"
        expect = dice_sum + C.CE_WEIGHT * ce_val
        print(f"    总 loss 校验: Dice {dice_sum:.4f} + {C.CE_WEIGHT} x CE {ce_val:.4f} "
              f"= {expect:.4f}  （实际 {loss_wrong.item():.4f}）")
        assert abs(loss_wrong.item() - expect) < 1e-3, "总 loss 的 Dice/CE 配比不对"
    else:
        assert abs(loss_wrong.item() - dice_sum) < 1e-3, "总 loss 应该等于各路之和"

    # --- 3. 忽略像素不参与：把一半像素标成 ignore ---
    t_masked = target.clone()
    t_masked[:, :, :S // 2] = C.IGNORE_INDEX
    l_a, _ = crit([perfect] * 5, t_masked)
    print(f"[3] 完美预测+一半忽略   loss = {l_a.item():.6f}   （期望依然接近 0）")
    assert l_a.item() < 1e-3, "忽略像素不应影响完美预测的 loss"

    # 全图都忽略 -> 没有任何有效像素，每类 dice = eps/eps = 1 -> loss = 0
    l_all_ign, _ = crit([perfect] * 5, torch.full((B, S, S), C.IGNORE_INDEX))
    print(f"    全图忽略            loss = {l_all_ign.item():.6f}   （期望 0，无有效像素）")

    # --- 4. 深监督权重确实生效 ---
    outs = [perfect, wrong, wrong, wrong, wrong]
    l_ds, parts = crit(outs, target)
    print(f"\n[4] 只有 d1 正确，d2~d5 全错")
    print(f"    深监督全开          loss = {l_ds.item():.6f}   "
          f"（d1≈0，其余 4 路各 ≈1，合计 ≈4）")
    old = C.DS_LOSS
    C.DS_LOSS = False
    l_nods, _ = crit(outs, target)
    C.DS_LOSS = old
    print(f"    关闭深监督损失      loss = {l_nods.item():.6f}   （应明显更小，只看 d1）")
    assert l_nods.item() < l_ds.item(), "只算 d1 的损失应该小于 5 路全算"

    # --- 5. 梯度能正常回传 ---
    logits = torch.randn(B, Cc, S, S, requires_grad=True)
    loss, _ = crit([logits] * 5, target)
    loss.backward()
    g = logits.grad
    print(f"\n[5] 反传检查  梯度范数 = {g.norm().item():.4f}  "
          f"NaN: {torch.isnan(g).any().item()}  Inf: {torch.isinf(g).any().item()}")
    assert torch.isfinite(g).all(), "梯度里出现了 NaN / Inf"

    print("\n" + "=" * 70)
    print("全部通过")
    print("=" * 70)


if __name__ == '__main__':
    _selftest()
