# -*- coding: utf-8 -*-
"""
UNet3+ 结构核对 + 内存/耗时探针

用法
----
    python inspect_model.py
        打印网络里每个编码 stage、每个解码节点的 5 路输入来源与形状、
        5 个深监督侧输出的形状、参数量统计。
        拿这个输出对照 model.py 顶部注释里的「结构速览」表，逐行核验拓扑。

    python inspect_model.py --probe
        内存 / 耗时探针。用合成数据走完整 forward+backward，
        分别测 fp32 与 bf16 的峰值内存和单步耗时，据此定 BATCH_SIZE。
        正式训练前必须跑一次。

    python inspect_model.py --probe --batch 4 --size 512      # 试别的配置

为什么要单独做探针：UNet3+ 里全分辨率的 hd1 节点要把 5 路输入 concat 起来
（32 + 160*4 = 672 通道 x H x W），是内存峰值所在。CPU 上这是唯一的真实风险，
先量出来再决定 batch size，比训练到一半 OOM 强。
"""

import argparse
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn

import config as C
import model as M


# ============================================================================
# 内存工具（Windows 下不依赖 psutil）
# ============================================================================

def _mem_mb():
    """
    返回 (当前工作集 MB, 峰值工作集 MB)。

    优先用 psutil；没有的话走 Windows 的 GetProcessMemoryInfo API。
    """
    try:
        import psutil
        p = psutil.Process(os.getpid())
        mi = p.memory_info()
        peak = getattr(mi, 'peak_wset', mi.rss)
        return mi.rss / 1024 ** 2, peak / 1024 ** 2
    except Exception:
        pass

    import ctypes
    import ctypes.wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PageFaultCount", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    c = PROCESS_MEMORY_COUNTERS()
    c.cb = ctypes.sizeof(c)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1024 ** 2, c.PeakWorkingSetSize / 1024 ** 2


def _total_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        pass
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(st)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullTotalPhys / 1024 ** 3


# ============================================================================
# 模式一：打印拓扑
# ============================================================================

def inspect_topology(size=512):
    """跑一次前向，把每个节点的形状打印出来。"""
    torch.manual_seed(0)
    model = M.build_model()
    model.eval()

    print("=" * 78)
    print(f"UNet3+ 结构核对   ——   输入 1x3x{size}x{size}")
    print("=" * 78)

    captured = {}

    def save(name):
        def hook(mod, inp, out):
            captured[name] = out.detach()
        return hook

    # 编码器各 stage 的输出
    for name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
        getattr(model, name).register_forward_hook(save(name))

    # 解码节点：用 pre_hook 抓「对齐之后的 5 路输入」，用 hook 抓节点输出
    dec_nodes = {'dec4': 'hd4', 'dec3': 'hd3', 'dec2': 'hd2', 'dec1': 'hd1'}
    for mname, tag in dec_nodes.items():
        node = getattr(model, mname)

        def pre_hook(mod, args, _tag=tag):
            captured[f'{_tag}_inputs'] = [t.detach() for t in args[0]]

        node.register_forward_pre_hook(pre_hook)
        node.register_forward_hook(save(tag))

    # 5 个深监督头
    for i in range(1, 6):
        getattr(model, f'outconv{i}').register_forward_hook(save(f'outconv{i}'))

    with torch.no_grad():
        outs = model(torch.zeros(1, 3, size, size))

    def shp(t):
        return f"({t.shape[1]}, {t.shape[2]}, {t.shape[3]})"

    # ---------------- 编码器 ----------------
    print("\n【编码器】每 stage = 两层 3x3 Conv + BN + ReLU，stage 间 2x2 maxpool\n")
    print(f"  输入        (3, {size}, {size})")
    stages = [('h1', 'conv1', '1/1'), ('h2', 'conv2', '1/2'),
              ('h3', 'conv3', '1/4'), ('h4', 'conv4', '1/8')]
    for tag, key, scale in stages:
        print(f"  {tag:4s}  {shp(captured[key]):>22s}   {scale}")
    print(f"  hd5   {shp(captured['conv5']):>22s}   1/16   ← 瓶颈（注意 conv5 的输入是 maxpool4(h4)）")

    # ---------------- 解码器 ----------------
    print("\n【解码器】全尺度跳跃连接：每个节点恒定 5 路输入")
    print("           每路先对齐分辨率，各自 3x3 卷积压到 "
          f"CatChannels={model.cat_channels}，concat 成 "
          f"UpChannels={model.up_channels}，再过一层 3x3 融合\n")

    # 每路输入的来源标注（顺序与 model.forward 里写的完全一致）
    src_names = {
        'hd4': ['h1↓8', 'h2↓4', 'h3↓2', 'h4同尺度', 'hd5↑2'],
        'hd3': ['h1↓4', 'h2↓2', 'h3同尺度', 'hd4↑2', 'hd5↑4'],
        'hd2': ['h1↓2', 'h2同尺度', 'hd3↑2', 'hd4↑4', 'hd5↑8'],
        'hd1': ['h1同尺度', 'hd2↑2', 'hd3↑4', 'hd4↑8', 'hd5↑16'],
    }

    for tag in ['hd4', 'hd3', 'hd2', 'hd1']:
        ins = captured[f'{tag}_inputs']
        out = captured[tag]
        raw_ch = sum(t.shape[1] for t in ins)          # 压通道之前的原始通道合计
        cat_ch = model.cat_channels * 5                # 真正 concat 的通道数
        h, w = ins[0].shape[2], ins[0].shape[3]
        print(f"  {tag} ←   (对齐后原始通道合计 {raw_ch})")
        for src, t in zip(src_names[tag], ins):
            print(f"        {src:10s} {shp(t):>22s}")
        print(f"        {'↓各压到':10s} {model.cat_channels} ch")
        print(f"        {'concat':10s} {f'({cat_ch}, {h}, {w})':>22s}"
              f"  -> 3x3 融合 -> {shp(out)}")
        print()

    # ---------------- 深监督 ----------------
    print("【深监督】5 个侧输出头，全部回到全分辨率\n")
    heads = [
        ('d1', 'outconv1', 'hd1', '—（已全分辨率，不上采样）'),
        ('d2', 'outconv2', 'hd2', '× 2'),
        ('d3', 'outconv3', 'hd3', '× 4'),
        ('d4', 'outconv4', 'hd4', '× 8'),
        ('d5', 'outconv5', 'conv5', '× 16  ← 接编码器瓶颈 hd5，输入通道是 filters[4]'),
    ]
    for tag, key, src, note in heads:
        src_ch = captured[src].shape[1]
        print(f"  {tag} = outconv({src:4s} {src_ch:>3d}ch) "
              f"-> 上采样 {note:45s} -> {shp(outs[int(tag[1]) - 1])}")

    # ---------------- 一致性自查 ----------------
    print("\n【自查】")
    ok = True

    # 1. 每个解码节点恰好 5 路
    for tag in ['hd4', 'hd3', 'hd2', 'hd1']:
        n = len(captured[f'{tag}_inputs'])
        flag = 'OK' if n == 5 else '错'
        ok &= (n == 5)
        print(f"  {tag} 输入路数 = {n}   [{flag}]")

    # 2. 解码节点输出通道 = UpChannels
    for tag in ['hd4', 'hd3', 'hd2', 'hd1']:
        ch = captured[tag].shape[1]
        flag = 'OK' if ch == model.up_channels else '错'
        ok &= (ch == model.up_channels)
        print(f"  {tag} 输出通道 = {ch}（应为 {model.up_channels}）   [{flag}]")

    # 3. 5 路侧输出都在全分辨率且类别数正确
    for i, o in enumerate(outs):
        good = (o.shape[1] == C.NUM_CLASSES
                and o.shape[2] == size and o.shape[3] == size)
        flag = 'OK' if good else '错'
        ok &= good
        print(f"  d{i + 1} 形状 {shp(o)}（应为 ({C.NUM_CLASSES}, {size}, {size})）   [{flag}]")

    # 4. 每个节点的 concat 通道数 = CatChannels * 5 = UpChannels
    for tag in ['hd4', 'hd3', 'hd2', 'hd1']:
        concat_ch = model.cat_channels * len(captured[f'{tag}_inputs'])
        flag = 'OK' if concat_ch == model.up_channels else '错'
        ok &= (concat_ch == model.up_channels)
        print(f"  {tag} concat 通道 = {concat_ch}（应为 UpChannels = {model.up_channels}）   [{flag}]")

    # ---------------- 参数量 ----------------
    total, trainable = M.count_parameters(model, verbose=True)
    print(f"\n【参数量】 总计 {total:,}   可训练 {trainable:,}")
    print(f"          官方全通道版约 26.97M，通道减半后应在 1/4 量级")

    print("\n" + ("=" * 78))
    print("全部自查通过" if ok else "★ 有自查未通过，检查 model.py 的通道配置")
    print("=" * 78)

    return ok


# ============================================================================
# 模式二：内存 / 耗时探针
# ============================================================================

def run_probe_once(batch, size, precision):
    """在本进程内跑一次 forward+backward，打印内存与耗时。由子进程调用。"""
    torch.manual_seed(0)
    torch.set_num_threads(C.TORCH_THREADS)

    model = M.build_model()
    model.train()

    # 探针按 config 里实际会用的配置来测，否则量出来的数没意义
    if C.CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    use_bf16 = (precision == 'bf16') and C.AMP_BF16

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(batch, 3, size, size)
    if C.CHANNELS_LAST:
        x = x.contiguous(memory_format=torch.channels_last)

    base_rss, _ = _mem_mb()

    # 预热一次，让内存分配器把缓存池建起来，避免把首次分配算进峰值
    warm = torch.zeros(1, 3, size, size)
    if C.CHANNELS_LAST:
        warm = warm.contiguous(memory_format=torch.channels_last)
    model(warm)[0].sum().backward()
    model.zero_grad(set_to_none=True)

    def step():
        opt.zero_grad(set_to_none=True)
        if use_bf16:
            with torch.autocast('cpu', dtype=torch.bfloat16):
                outs = model(x)
                loss = sum(o.float().mean() for o in outs)
        else:
            outs = model(x)
            loss = sum(o.mean() for o in outs)
        loss.backward()
        opt.step()

    t0 = time.time()
    step()
    t1 = time.time()
    cur, peak = _mem_mb()

    print(f"RESULT|{precision}|{batch}|{size}|{t1 - t0:.2f}|{cur - base_rss:.0f}|{peak:.0f}|{base_rss:.0f}")


def run_probe(batch, size):
    """起两个子进程分别测 fp32 / bf16，这样峰值内存互不污染。"""
    total_ram = _total_ram_gb()
    print("=" * 78)
    print(f"内存 / 耗时探针    batch={batch}  size={size}  机器总内存 {total_ram:.1f} GB")
    print("=" * 78)
    print(f"线程数 {C.TORCH_THREADS}，CPU: {torch.backends.cpu.get_cpu_capability()}，"
          f"channels_last: {'开' if C.CHANNELS_LAST else '关'}，"
          f"AMP_BF16: {'开' if C.AMP_BF16 else '关'}（config.py 里的当前设置）\n")
    print("正在测量（每个精度一个独立子进程，各跑一次完整的 forward+backward）...\n")

    results = []
    for prec in ['fp32', 'bf16']:
        cmd = [sys.executable, os.path.abspath(__file__),
               '--probe-inner', '--batch', str(batch),
               '--size', str(size), '--precision', prec]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        line = next((l for l in p.stdout.splitlines() if l.startswith('RESULT|')), None)
        if line is None:
            print(f"  {prec}: 测量失败\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
            continue
        _, prec, b, s, sec, delta, peak, base = line.split('|')
        results.append(dict(prec=prec, batch=int(b), size=int(s), sec=float(sec),
                            delta=float(delta), peak=float(peak), base=float(base)))

    if not results:
        return

    print(f"{'精度':<6} {'单步耗时':>10} {'本步增量':>12} {'进程峰值':>12}")
    print("-" * 46)
    for r in results:
        print(f"{r['prec']:<6} {r['sec']:>8.2f} s {r['delta']:>10.0f} MB {r['peak']:>10.0f} MB")

    if len(results) == 2:
        sp = results[0]['sec'] / results[1]['sec']
        print(f"\nbf16 相对 fp32 提速：{sp:.2f}x"
              f"   {'（有用，建议开启 AMP_BF16）' if sp > 1.1 else '（不明显，可关掉 AMP_BF16）'}")

    peak = max(r['peak'] for r in results)
    print(f"\n进程峰值 {peak:.0f} MB / 机器 {total_ram * 1024:.0f} MB"
          f"  —— 占用约 {peak / (total_ram * 1024) * 100:.1f}%")
    if peak > total_ram * 1024 * 0.6:
        print("★ 峰值已超总内存 60%，正式训练前建议降级：batch 减半，或分辨率降到 384。")

    # 估算一个 epoch
    train_n = 1765
    iters = train_n / batch
    print(f"\n单 epoch 估算（训练集 {train_n} 张，batch={batch}，共 {iters:.0f} 步）")
    for r in results:
        mins = iters * r['sec'] / 60
        print(f"  {r['prec']}: 约 {mins:.0f} 分钟")
    print("\n注意：这是纯计算的估算，实际还要加数据加载与验证集的时间，通常再多 20~40%。")


# ============================================================================
# 入口
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="UNet3+ 结构核对与内存探针")
    ap.add_argument('--probe', action='store_true', help='跑内存/耗时探针')
    ap.add_argument('--probe-inner', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--precision', default='fp32', choices=['fp32', 'bf16'], help=argparse.SUPPRESS)
    ap.add_argument('--batch', type=int, default=C.BATCH_SIZE, help=f'探针用的 batch size（默认 {C.BATCH_SIZE}）')
    ap.add_argument('--size', type=int, default=C.CROP_SIZE, help=f'探针用的分辨率（默认 {C.CROP_SIZE}）')
    args = ap.parse_args()

    if args.probe_inner:
        run_probe_once(args.batch, args.size, args.precision)
    elif args.probe:
        run_probe(args.batch, args.size)
    else:
        inspect_topology(args.size)


if __name__ == '__main__':
    main()
