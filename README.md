# UNet3+ on LoveDA

从零实现的 UNet3+ 语义分割，在 LoveDA 数据集上**纯 CPU** 训练。
7 分类（背景/建筑/道路/水体/裸地/林地/耕地），1024×1024 遥感影像。

**当前成绩**：完整 1024 滑窗、506 张 test —— **ALL mIoU 0.5079**（Rural 0.5246 / Urban 0.4091）

架构细节见 [`docs_unet3plus.md`](docs_unet3plus.md)；那份文档解释了网络拓扑、
与官方实现的差异、以及各类踩坑的实测证据。**本文件讲怎么用。**

---

## 快速开始

```bash
cd D:/agent/cc/loveda

python gui.py                      # 可视化界面（推荐入口）
python inspect_model.py            # 先确认模型结构正常（17 项自查）
python train.py                    # 训练（会自动从 last.pt 续训）
python eval.py --split test        # 完整滑窗评测，约 56 分钟
```

---

## 环境

| | |
|---|---|
| Python | 3.12.0（`D:\Python`） |
| torch | 2.4.1+**cpu** |
| 其他 | PyQt5 5.15.11 / matplotlib 3.10.3 / albumentations / opencv 4.8.1 / numpy 1.26.4 |
| 硬件 | 16 逻辑核 CPU，27.7 GB 内存，**无 GPU** |

直接用 `python`，不要去探测 conda 环境。

> ⚠️ 这台机器上 **PyQt5 和 PyQt6 同时装着**。`gui.py` 里的 import 顺序
> （`QT_API` → PyQt5 → matplotlib）**不能重排**，否则 matplotlib 会挑中 PyQt6，
> 画布控件塞进 PyQt5 布局直接报 `TypeError`。详见 `gui.py` 顶部注释。

---

## 数据

```
LDA/Train/
  Rural/images_png/*.png    Rural/masks_png/*.png    1366 对
  Urban/images_png/*.png    Urban/masks_png/*.png    1156 对
```

图像 1024×1024 uint8 RGB；掩码 1024×1024 uint8 单通道，值 0~7
（0 = 未标注 → 重映射成 255 忽略，1..7 → 类别 0..6）。

⚠️ **本地只有 `Train/`，没有官方的 Val/Test 目录。** 所以 train/val/test 是按域
**各自分层** 7:1:2 从 `Train/` 划出来的，结果落盘在 `splits.json`，之后只读不重划：

| | train | val | test |
|---|---|---|---|
| Rural | 956 | 136 | 274 |
| Urban | 809 | 115 | 232 |
| **合计** | **1765** | **251** | **506** |

**这意味两件事**：① 本项目的数字与 LoveDA 官方榜单**口径不同，不能直接比**；
② 官方榜单里**没有 UNet3+**（官方基线是 UNet 47.84 / UNet++ 48.20，且用了
ResNet50 + ImageNet 预训练），本实现是从零训练、无预训练、通道减半。

---

## 各脚本干什么

| 文件 | 用途 |
|---|---|
| `gui.py` | **可视化界面**，三个页签，见下节 |
| `config.py` | 全部超参数集中在这里，做实验只改这个文件 |
| `model.py` | UNet3+ 网络本体 + 权重初始化 |
| `dataset.py` | 划分生成 / 读取 / 增强 / DataLoader |
| `loss.py` | 多分类软 Dice + 深监督 + 可选 CE 项 |
| `metrics.py` | 混淆矩阵指标（mIoU/mDice/aAcc）与配色、可视化 |
| `train.py` | 训练主循环（warmup + 余弦退火） |
| `eval.py` | 完整 1024 滑窗评测 |
| `bench_step.py` | 单步耗时拆解：纯计算 vs 端到端，定位瓶颈 |
| `inspect_model.py` | 结构核对 + 内存/耗时探针 |

几个脚本可以直接单独跑，自带自测：`python loss.py` / `metrics.py` / `dataset.py`。

---

## 可视化界面

```bash
python gui.py
```

| 页签 | 做什么 |
|---|---|
| **单图推理演示** | 选一张测试集图片（或打开本地图片）→ 完整 1024 滑窗 → 出「原图 \| 真值 \| 预测 \| 错误图」四联图 + 逐类 IoU/Dice/占比表 |
| **测试集评测面板** | 选划分/权重/张数 → 批量滑窗 → 分域汇总 + 可按域切换的逐类明细，可存文本报告和可视化图 |
| **训练曲线查看** | 自动扫描并合并 `logs/train_*.csv`，画曲线并标出 best 轮与 warmup 结束位置 |

输出统一写到 `outputs/gui_eval/`。**不含训练控制**——训练仍然走命令行。

几个使用要点：

- 单张 1024 图滑窗约 **6.6 秒**（9 个窗口），506 张全跑约 **56 分钟**。
  评测页超过 60 张会先弹框报预计耗时。
- 评测的**「张数」限制**和 `eval.py --limit N` 是同一套口径，两边结果可以逐位对拍。
- 任务跑着的时候**关窗口会先拦一下**。确认退出会取消任务并等当前那张跑完。
- **权重缓存**连路径一起存，换权重会自动重载。
- 本地图片没有真值，所以只显示「原图 \| 预测」两联，不计算指标。

### 配色用的是 LoveDA 官方调色板

`metrics.py` 的 `PALETTE` 逐位等于官方参考实现（mmsegmentation `mmseg/datasets/loveda.py`
与 TorchGeo 一致），**不是**自选的相近色：

| 背景 | 建筑 | 道路 | 水体 | 裸地 | 林地 | 耕地 |
|---|---|---|---|---|---|---|
| `255,255,255` | `255,0,0` | `255,255,0` | `0,0,255` | `159,129,183` | `0,255,0` | `255,195,128` |

表头顺序就是 `config.CLASS_NAMES` 的下标顺序 —— **背景排在第 0 位**，
和官方掩码原值 `1=background … 7=agriculture` 一致（`dataset.py` 是保序重映射
`1..7 -> 0..6`，所以两边顺序天然相同，可以逐位照抄）。

这样渲染出来的图才能和官方论文、榜单图、别人的复现结果**直接肉眼对拍**。
`python metrics.py` 的自测第 7 项会逐位核对这 7 个值，想「顺手调好看一点」会被当场拦下。

> ⚠️ **这里踩过大坑：`CLASS_NAMES` 曾经整体左移一位**（写成「建筑…背景，背景在最后」），
> 于是所有图、表、文档里的类名全部错位 —— 水体被叫成裸地、裸地被叫成林地。
> 指标数值一直是对的（按类别下标算的），错的只是名字。
> 后果是整场「林地 IoU 从恒 0 恢复到 0.3」的调查，追的其实是**裸地**。
> 唯一的判据是数据本身：值 4 最暗且成片连续 = 水体，值 5 最亮偏土黄 = 裸地。

> ⚠️ 在改回官方配色**之前**生成的图已经处理过了：`epoch030_pred.png` 用当时的权重
> （就是现在的 `last.pt`）按官方配色**原样重建**；重建不了的 6 张
> （`epoch002/005/…/025`）连权重一起没了，已挪进 `outputs/archive_oldpalette/`。
> 全量 test 的 `predictions.png` 也还在旧配色，重跑 `python eval.py --split test`
> 就会变成官方配色（约 56 分钟）。

---

## 训练

```bash
python train.py                    # 正常训练，自动从 checkpoints/last.pt 续训
python train.py --smoke            # 冒烟测试：少量样本跑 2 轮，验证链路
python train.py --fresh            # 忽略已有断点，从第 1 轮重来
```

配置在 `config.py`：AdamW / lr 1e-3 / wd 1e-4，线性 warmup 3 轮 + 余弦退火到 1e-6，
512 裁剪，batch 2，30 轮。开了 `CHANNELS_LAST` + `AMP_BF16`（实测 3.14 s/step，
30 轮约 23 小时；两个都关掉是 59.5 小时）。

### 续训的两条铁律

**绝对不要加 `--fresh`**（会从第 1 轮重来，几十小时作废），
**也不要改 `--epochs`**。启动时会打印「已从断点续训：上次跑到第 N 轮」。

为什么 `--epochs` 不能改：`LambdaLR.state_dict()` **只存 `last_epoch`、不存曲线**。
原先按 8 轮跑到第 8 轮（lr 已降到 1e-6）再接 30 轮，lr 会从 1e-6 **突跳回 9.2e-4**，
足以毁掉整次训练。现已加 `sched_sig` 守卫：断点里存下
`{epochs, warmup, eta_min_ratio}`，续训时不一致就**拒绝启动并打印差异**。

> 想控制总轮数就**靠中途停车**，别改曲线参数。

**续训会新建一个 CSV 日志**，历史因此分散在多个文件里 —— 数据没丢，
`last.pt` 内存着完整 `history`，GUI 的曲线页会按 epoch 自动合并。

### 损失函数：CE_WEIGHT 不能改成 0

```python
CE_WEIGHT = 0.1        # Dice + 0.1 * CE，5 路深监督等权求和
```

第一轮用**纯 Dice**（`CE_WEIGHT=0`）跑满 30 轮，**裸地 IoU 恒为 0** ——
40 张验证图上真实裸地 629,839 像素，模型预测出的裸地 **0 像素**。
是模型真的不输出裸地，不是指标 bug。

> 当时这一节写的是「林地」，因为 `CLASS_NAMES` 错位，把下标 4（真正的裸地）叫成了林地。
> 见上面配色一节。下面统一用「裸地」。

根因是 softmax 竞争 + Dice 对缺席类的惩罚：饱和之后 Dice 对该类的梯度
**下溢为 0**（与 CE 差 22 个数量级），从此再也拉不回来。

第二轮**同时改了两处**才能救回来，两处是一起改的，所以**没法说哪一处功劳更大**：

- `CE_WEIGHT` 0 → **0.1**（给饱和类留一条还能走的梯度）
- 5 个 softmax 头初始化 → `normal_(std=0.01)`（见 `docs_unet3plus.md` §6.1，
  原 kaiming 给出的 std 是 0.178，初始 maxprob 就到 0.78，从第 0 步起就在往饱和跑）

效果：裸地 IoU 从**恒 0** 恢复到最终滑窗测试的 **0.3028**。

完整的实测证据（含一条**已证伪的修法** —— 把缺席类从 Dice 平均里剔除，
实测反而更糟）见 `docs_unet3plus.md` §7.1。

---

## 评测

```bash
python eval.py                         # best.pt 在 test 集上跑，约 56 分钟
python eval.py --split val             # 换验证集
python eval.py --limit 20              # 只跑 20 张，快速看一眼
python eval.py --stride 512            # 取消重叠，速度翻倍（边缘质量略降）
python eval.py --vis 0                 # 不输出可视化图
```

滑窗推理在**原分辨率**上做：窗口 512 / 步长 256（50% 重叠），
重叠区按 **softmax 概率平均**（不是对 argmax 投票），接缝更平滑。
不缩放是因为地物尺度会和训练时不一致，不中心裁剪是因为会丢掉 3/4 的画面。

输出在 `outputs/sliding_<split>/`：`report.txt` + `predictions.png`。

> ⚠️ **训练日志里的 `val_miou` 不是最终成绩。** 那是每轮用中心裁剪 512 算的，
> 只覆盖每张图的中心区域，比真实滑窗值**高估约 0.029**（0.5372 vs 0.5079）。
> 它只能看趋势。

---

## 当前结果

`checkpoints/best.pt` = **第 27 轮**（`last.pt` = 第 30 轮）。

完整 1024 滑窗 / 506 张 test / 用时 55.7 分钟：

| | mIoU | mDice | aAcc | oAcc |
|---|---|---|---|---|
| **ALL** | **0.5079** | 0.6676 | 0.6575 | 0.6833 |
| Rural | 0.5246 | 0.6796 | 0.6844 | 0.7084 |
| Urban | 0.4091 | 0.5633 | 0.5584 | 0.6529 |

逐类 IoU(ALL)：背景 0.5063 / 建筑 0.4573 / 道路 0.5179 / 水体 0.6269 /
**裸地 0.3028** / 林地 0.5693 / 耕地 0.5748

### checkpoints 目录

| 文件 | 说明 |
|---|---|
| `best.pt` | 第 27 轮，mIoU 0.5372（**中心裁剪口径**） |
| `last.pt` | 第 30 轮 |
| `best_epoch029_miou0.4738_forestdead.pt` | 旧版（纯 Dice）**备份，裸地 IoU 恒 0**，仅作对照。文件名里的 `forestdead` 是类名错位时期的误标，不再改（改了就对不上磁盘） |
| `last_epoch030_forestdead.pt` | 同上 |
| `smoke_*.pt` | 冒烟测试产物，与正式断点隔离 |

---

## 已知问题

1. **「稀有类在某个域里被压死」的结构性机制还在。** CE 只治好了裸地这一个病例。
   Urban 域里 `耕地 IoU = 0.1428`、`林地 IoU = 0.3070`（Rural 分别是 0.6097 / 0.6257）——
   这两类在城市域里稀有，被按类平均的 mIoU 一视同仁地重罚，光耕地一类就压掉 Urban 约 0.06。
   用 GUI 评测页的「域」下拉框切到 Urban 能直接看到。
2. **Urban 整体比 Rural 低 0.12**（0.4091 vs 0.5246）。城市域类别更碎、小目标更多。
3. **裸地仍是全类最低**（0.3028），虽已从恒 0 救回来。
4. **纯 CPU**，单张 1024 推理 6.6 秒。要提速优先考虑租 GPU 而不是优化代码。

---

## 目录

```
loveda/
├── gui.py  config.py  model.py  dataset.py  loss.py  metrics.py  train.py  eval.py
├── bench_step.py  inspect_model.py
├── docs_unet3plus.md          # 架构笔记（拓扑、踩坑、实测证据）
├── README.md                  # 本文件
├── splits.json                # 划分清单，只读不重划
├── LDA/Train/{Rural,Urban}/   # 数据
├── checkpoints/               # 权重
├── logs/                      # train_*.csv（每轮一行）+ 控制台日志
└── outputs/                   # 可视化图、评测报告、gui_eval/
```
