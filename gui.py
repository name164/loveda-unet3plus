# -*- coding: utf-8 -*-
"""
LoveDA UNet3+ 可视化界面

    python gui.py

三个页签
--------
[0] 单图推理演示   选一张图 -> 完整 1024 滑窗推理 -> 原图/真值/预测/错误图 四联图 + 逐类统计
[1] 测试集评测面板 选划分和权重 -> 批量滑窗 -> 分域指标 + 逐类明细 + 存文本报告
[2] 训练曲线查看   合并 logs/*.csv -> 画曲线，标注 best 轮与 warmup 结束位置

**不含训练控制**，train.py 不被这个界面触碰。

两个刻意的设计
--------------
1. 全程不 import matplotlib.pyplot。
   画曲线用 Figure + FigureCanvasQTAgg，四联图用纯 numpy 拼接。
   pyplot 会引入一套全局状态，和 metrics.save_prediction_grid() 里的
   matplotlib.use('Agg') 撞在一起时行为不好预测，索性不碰它。

2. 推理、评测一律在 QThread 里跑。
   单张图 1024 滑窗约 6.6 秒，506 张约 56 分钟 —— 放主线程界面会直接假死。
"""

import csv
import glob
import os
import sys
import time
import traceback

# ---- 导入顺序在这里是有意义的，别随手重排 --------------------------------
#
# matplotlib 的 qt_compat 会自己挑一个 Qt 绑定：它先看 QT_API 环境变量，
# 没有就看 sys.modules 里已经装了哪个。这台机器上 PyQt5 和 PyQt6 同时存在，
# 如果先 import matplotlib 后端、后 import PyQt5，它会挑中 PyQt6 ——
# 于是 FigureCanvasQTAgg 变成一个 PyQt6 控件，塞进 PyQt5 的布局里直接报
# TypeError: argument 1 has unexpected type 'FigureCanvasQTAgg'。
#
# 所以：先钉死 QT_API，再 import PyQt5，最后才碰 matplotlib 的 Qt 后端。
#
# 还有第二条，比上面那条更凶：**torch 必须在 QApplication 之前 import**。
# 实测（QT_QPA_PLATFORM=offscreen，本机重复 3/3 一致）：
#     先 QApplication([]) 再 import torch  -> 段错误，退出码 139，没有 traceback
#     先 import torch 再 QApplication([])  -> 正常
# 原因是 torch 加载自己的 OpenMP/DLL 运行时，和 Qt 已经初始化好的那套撞车。
# 段错误不是异常，try/except 拦不住，只会看到进程凭空消失。
#
# 本文件这个顺序天然是对的：torch 在模块顶层，QApplication 在 main() 里才建。
# main() 开头加了一句 assert 兜着，谁把 import 挪进去就会当场炸出人话。
os.environ.setdefault('QT_API', 'pyqt5')

import numpy as np
import torch

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QStatusBar, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg   # noqa: E402
from matplotlib.figure import Figure                              # noqa: E402

import config as C
from dataset import LoveDADataset, build_splits, get_keys
from eval import _window_starts, sliding_predict
from metrics import (PALETTE, DomainMetrics, SegMetrics, _setup_cjk_font,
                     colorize, denormalize, error_map, save_prediction_grid)
from model import build_model

# 单张 1024 图的滑窗耗时基准（实测：506 张 test 跑 55.7 分钟 = 每张 6.6 秒，
# 512/256 步长下每张 9 个窗口，折合 0.73 秒/窗口）。用来给用户报预计时间。
SEC_PER_WINDOW = 0.75

OUT_GUI = os.path.join(C.OUT_DIR, 'gui_eval')       # 界面的所有输出都写这里


# ============================================================================
# 基础工具
# ============================================================================

def fmt(v, nd=4):
    """格式化指标。NaN（该类在数据里没出现过）显示成 --，不要显示成 nan。"""
    if v is None:
        return '--'
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '--'
    return '--' if np.isnan(v) else f'{v:.{nd}f}'


def np_to_pixmap(arr):
    """
    (H,W,3) uint8 -> QPixmap。

    QImage 构造时**不拷贝**传入的 buffer，array 被回收后画面就会花屏。
    所以这里先 ascontiguousarray 保证内存连续，再 .copy() 让 QImage 持有自己的副本。
    """
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def quad_image(img_chw, gt, pred):
    """
    把推理结果拼成一张横向长图：原图 | 真值 | 预测 | 错误图。

    全部是 metrics.py 里的纯 numpy 函数，不经过 matplotlib。
    本地图片没有真值时退化成 原图 | 预测 两联。
    """
    panels = [denormalize(np.asarray(img_chw))]
    if gt is not None:
        panels += [colorize(gt), colorize(pred), error_map(pred, gt)]
    else:
        panels += [colorize(pred)]
    return np.concatenate(panels, axis=1)


def load_infer_model(ckpt_path):
    """
    建模型 + 载权重。照抄 eval.py 的加载姿势。

    weights_only=False 是必须的 —— 断点里除了张量还有 history / config 这些
    普通 Python 对象，默认的 weights_only=True 会直接拒载。
    """
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    if C.CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, ckpt


def ckpt_summary(path):
    """读断点里的轮数和 best mIoU，只用于界面显示。读不动就返回问号。"""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        return ckpt.get('epoch', '?'), ckpt.get('best_miou', float('nan'))
    except Exception:
        return '?', float('nan')


def list_checkpoints():
    """checkpoints/*.pt，排除冒烟测试的产物。best.pt 排在最前面。"""
    paths = [p for p in glob.glob(os.path.join(C.CKPT_DIR, '*.pt'))
             if not os.path.basename(p).startswith('smoke_')]
    paths.sort(key=lambda p: (os.path.basename(p) != 'best.pt', os.path.basename(p)))
    return paths


# ============================================================================
# 后台线程
# ============================================================================
# 约定：同一时刻只允许一个 worker 在跑。两个线程各开 16 个 torch 线程会互相抢核，
# 而且界面照样卡。窗口层用 self.busy 统一管控。

class InferWorker(QThread):
    """单张图的滑窗推理。"""

    progress = pyqtSignal(int, int)          # 已完成窗口数 / 总窗口数
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, ckpt_path, cached, img, gt, key, window, stride, parent=None):
        super().__init__(parent)
        self.ckpt_path = ckpt_path
        self.cached = cached
        self.img = img
        self.gt = gt
        self.key = key
        self.window = window
        self.stride = stride
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            torch.set_num_threads(C.TORCH_THREADS)
            if self.cached is not None:
                model = self.cached
            else:
                model, _ = load_infer_model(self.ckpt_path)

            def cb(done_n, total):
                self.progress.emit(done_n, total)
                return not self._cancel          # 返回 False 会让 sliding_predict 中断

            pred = sliding_predict(model, self.img, self.window, self.stride, cb)
            if pred is None:
                self.done.emit({'cancelled': True})
                return

            self.done.emit({
                'cancelled': False,
                'model': model,
                'ckpt': self.ckpt_path,
                'pred': pred.numpy(),
                'img': self.img.numpy(),
                'gt': None if self.gt is None else self.gt.numpy(),
                'key': self.key,
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


class EvalWorker(QThread):
    """整个 split 的批量滑窗评测。"""

    progress = pyqtSignal(int, int)          # 已处理图数 / 总图数
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, ckpt_path, cached, split, limit, window, stride,
                 vis_n, parent=None):
        super().__init__(parent)
        self.ckpt_path = ckpt_path
        self.cached = cached
        self.split = split
        self.limit = limit
        self.window = window
        self.stride = stride
        self.vis_n = vis_n
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            torch.set_num_threads(C.TORCH_THREADS)
            if self.cached is not None:
                model = self.cached
            else:
                model, _ = load_infer_model(self.ckpt_path)

            keys = get_keys(self.split)
            if self.limit:
                keys = keys[:self.limit]

            # 逐张从磁盘读，Dataset 自身不缓存，500 张也不会把内存撑爆
            ds = LoveDADataset(keys, mode='full')
            dm = DomainMetrics()
            vis, n_done, cancelled = [], 0, False

            for i, key in enumerate(keys):
                if self._cancel:
                    cancelled = True
                    break

                img, gt, _ = ds[i]
                pred = sliding_predict(model, img, self.window, self.stride,
                                       lambda a, b: not self._cancel)
                if pred is None:
                    cancelled = True
                    break

                dm.update(pred, gt, key.split('/', 1)[0])
                n_done += 1

                if len(vis) < self.vis_n:
                    vis.append(dict(image=img.numpy(), gt=gt.numpy(),
                                    pred=pred.numpy(), title=key))
                self.progress.emit(n_done, len(keys))

            self.done.emit({'cancelled': cancelled, 'model': model, 'dm': dm,
                            'n': n_done, 'total': len(keys), 'vis': vis,
                            'split': self.split, 'window': self.window,
                            'stride': self.stride, 'ckpt': self.ckpt_path})
        except Exception:
            self.failed.emit(traceback.format_exc())


# ============================================================================
# 小部件
# ============================================================================

class ImageView(QLabel):
    """按窗口大小自适应缩放的图片显示区。缓存原图，resize 时只重算缩放。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(220)
        self.setStyleSheet('background:#1b1b1b; color:#888;')
        self.setText('（还没有结果）')

    def set_array(self, arr):
        self._pix = np_to_pixmap(arr)
        self._rescale()

    def clear(self):
        self._pix = None
        super().setPixmap(QPixmap())
        self.setText('（还没有结果）')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._pix is None:
            return
        self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation))


def make_class_table():
    """7 行 x 4 列的逐类指标表。"""
    t = QTableWidget(len(C.CLASS_NAMES), 4)
    t.setHorizontalHeaderLabels(['类别', 'IoU', 'Dice', 'GT 占比'])
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.NoSelection)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    return t


def fill_class_table(table, res):
    """
    用 SegMetrics.compute() 的结果填表。

    注意 res['iou'] 里，某个类如果在真值和预测里都没出现过就是 NaN，
    必须走 fmt() 显示成 --，直接 f'{v:.4f}' 会打印出 nan。
    """
    for c, name in enumerate(C.CLASS_NAMES):
        share = res['gt_pixels'][c] / max(res['total_pixels'], 1)
        cells = [name, fmt(res['iou'][c]), fmt(res['dice'][c]),
                 '--' if res['total_pixels'] == 0 else f'{share * 100:.2f}%']
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if col > 0:
                item.setTextAlignment(Qt.AlignCenter)
            table.setItem(c, col, item)


def _text_on(rgb):
    """
    在给定底色上选黑字还是白字。

    判据是 WCAG 的**相对亮度**（阈值 0.179），不是三通道均值。均值会被
    纯绿 [0,255,0] 骗过去 —— 它三通道均值只有 85，看着"暗"，其实亮得很，
    按均值判会得到"白字压纯绿"，等于看不见；相对亮度按人眼敏感度加权后
    它是 0.715，正确判成黑字。官方纯红 [255,0,0] 同理（均值 85 骗人），
    黑字 5.25:1 也明显好过白字 4.00:1。
    """
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    luma = float((c * (0.2126, 0.7152, 0.0722)).sum())
    return '#000' if luma > 0.179 else '#fff'


def make_legend():
    """类别配色图例条。四联图里每种颜色代表什么，看这一行就够了。"""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    for name, rgb in zip(C.CLASS_NAMES, PALETTE):
        lb = QLabel(f' {name} ')
        fg = _text_on(rgb)
        lb.setStyleSheet(f'background: rgb({rgb[0]},{rgb[1]},{rgb[2]});'
                         f'color: {fg}; border: 1px solid #999;')
        lay.addWidget(lb)
    err = QLabel(' 品红 ')
    err.setStyleSheet('background: rgb(255,0,255); color:#fff; border: 1px solid #999;')
    lay.addWidget(QLabel('|  错误图像素:'))
    lay.addWidget(err)
    lay.addStretch(1)
    return w


# ============================================================================
# 页签 0：单图推理演示
# ============================================================================

class TabInfer(QWidget):

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.worker = None
        self.cur_img = None          # (3,H,W) Tensor，已归一化
        self.cur_gt = None           # (H,W) Tensor，本地图片可能是 None
        self.cur_key = None

        self.combo = QComboBox()
        self.combo.addItems(get_keys('test'))
        self.combo.setMinimumWidth(220)
        # 换图时必须把上一张丢掉。否则「先打开本地图片、再从下拉框选一张、
        # 点开始推理」会拿下拉框的 key 配本地图片的像素，静默算出错误的指标。
        self.combo.currentIndexChanged.connect(self.reset_selection)

        btn_rand = QPushButton('随机一张')
        btn_open = QPushButton('打开本地图片…')
        btn_rand.clicked.connect(self.pick_random)
        btn_open.clicked.connect(self.open_local)

        self.spin_w = QSpinBox()
        self.spin_w.setRange(64, 2048)
        self.spin_w.setSingleStep(64)
        self.spin_w.setValue(C.SLIDING_WINDOW)
        self.spin_s = QSpinBox()
        self.spin_s.setRange(16, 2048)
        self.spin_s.setSingleStep(64)
        self.spin_s.setValue(C.SLIDING_STRIDE)

        self.btn_run = QPushButton('开始推理')
        self.btn_run.clicked.connect(self.run_infer)
        self.btn_cancel = QPushButton('取消')
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_cancel.setEnabled(False)

        self.bar = QProgressBar()
        self.bar.setFormat('%v / %m 个窗口')
        self.bar.setValue(0)

        top = QHBoxLayout()
        top.addWidget(QLabel('图片'))
        top.addWidget(self.combo, 1)
        top.addWidget(btn_rand)
        top.addWidget(btn_open)

        mid = QHBoxLayout()
        mid.addWidget(QLabel('窗口'))
        mid.addWidget(self.spin_w)
        mid.addWidget(QLabel('步长'))
        mid.addWidget(self.spin_s)
        mid.addWidget(self.btn_run)
        mid.addWidget(self.btn_cancel)
        mid.addWidget(self.bar, 1)

        self.view = ImageView()
        scroll = QScrollArea()
        scroll.setWidget(self.view)
        scroll.setWidgetResizable(True)

        self.summary = QLabel('mIoU --   mDice --   aAcc --')
        self.summary.setStyleSheet('font-size: 15px; font-weight: bold;')
        self.table = make_class_table()
        self.hint = QLabel('')

        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.summary)
        bl.addWidget(make_legend())
        bl.addWidget(self.hint)
        bl.addWidget(self.table, 1)

        split = QSplitter(Qt.Vertical)
        split.addWidget(scroll)
        split.addWidget(bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addLayout(mid)
        root.addWidget(split, 1)

        self.btn_save = QPushButton('保存这张对比图')
        self.btn_save.clicked.connect(self.save_figure)
        self.btn_save.setEnabled(False)
        mid.addWidget(self.btn_save)

        self._last_quad = None
        self._last_pred = None

    # ------------------------------------------------------------------
    def reset_selection(self):
        """丢掉当前缓存的图，下一次点「开始推理」会重新从下拉框读一张。"""
        self.cur_img = self.cur_gt = self.cur_key = None
        self._last_pred = self._last_quad = None
        self.view.clear()
        self.summary.setText('mIoU --   mDice --   aAcc --')
        self.table.clearContents()
        self.hint.setText('')
        self.btn_save.setEnabled(False)
        self.bar.setValue(0)

    def pick_random(self):
        import random
        self.combo.setCurrentIndex(random.randrange(self.combo.count()))

    def open_local(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择一张遥感影像', C.ROOT,
            '图片 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)')
        if not path:
            return
        try:
            import cv2
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise IOError('读不出来')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - np.asarray(C.NORM_MEAN, np.float32)) / np.asarray(C.NORM_STD, np.float32)
            self.cur_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose(2, 0, 1))).float()
            self.cur_gt = None
            self.cur_key = os.path.basename(path) + '（本地，无真值）'
        except Exception as e:
            QMessageBox.critical(self, '读图失败', f'{path}\n\n{e}')
            return

        self.view.clear()
        self.summary.setText('mIoU --   mDice --   aAcc --')
        self.table.clearContents()
        self.hint.setText(f'已载入本地图片 {os.path.basename(path)}：'
                          f'没有对应的真值掩码，只显示预测与「原图 | 预测」两联图。')
        self.btn_save.setEnabled(False)
        self._last_quad = None

    # ------------------------------------------------------------------
    def run_infer(self):
        """锁定当前图片尺寸，然后在后台跑滑窗。"""
        if self.cur_img is None or self.cur_key is None:
            # 从下拉框取一张测试集的图，走和 eval.py 完全相同的读取路径
            key = self.combo.currentText()
            try:
                ds = LoveDADataset([key], mode='full')
                self.cur_img, self.cur_gt, _ = ds[0]
                self.cur_key = key
            except Exception as e:
                QMessageBox.critical(self, '读图失败', f'{key}\n\n{e}')
                return
            self.hint.setText('')

        w = self.spin_w.value()
        s = min(self.spin_s.value(), w)          # 步长大于窗口会有空洞，夹住
        if s != self.spin_s.value():
            self.spin_s.setValue(s)

        self.win.set_busy(True)
        self.btn_cancel.setEnabled(True)
        self.view.clear()
        self.summary.setText('推理中…')
        self.table.clearContents()

        ckpt = self.win.ckpt_path()
        self.worker = InferWorker(ckpt, self.win.cached_model(ckpt),
                                  self.cur_img, self.cur_gt, self.cur_key, w, s)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self.btn_cancel.setEnabled(False)

    def on_progress(self, done_n, total):
        self.bar.setMaximum(total)
        self.bar.setValue(done_n)
        self.win.status(f'推理中 {done_n}/{total} 个窗口…')

    def on_done(self, res):
        self.win.set_busy(False)
        self.btn_cancel.setEnabled(False)
        if res.get('model') is not None:
            self.win.put_model(res['ckpt'], res['model'])

        if res.get('cancelled'):
            self.view.clear()
            self.summary.setText('已取消')
            self.win.status('推理已取消')
            return

        pred, gt = res['pred'], res['gt']
        quad = quad_image(res['img'], gt, pred)
        self._last_quad = quad
        self._last_pred = pred
        self.view.set_array(quad)
        self.btn_save.setEnabled(True)

        if gt is None:
            self.summary.setText('无真值，不计算指标')
            self.table.clearContents()
        else:
            m = SegMetrics()
            m.update(pred, gt)
            r = m.compute()
            self.summary.setText(f"mIoU {r['miou']:.4f}   mDice {r['mdice']:.4f}   "
                                 f"aAcc {r['aacc']:.4f}   oAcc {r['oacc']:.4f}")
            fill_class_table(self.table, r)

        self.win.status(f"完成：{res['key']}")

    def on_failed(self, msg):
        self.win.set_busy(False)
        self.btn_cancel.setEnabled(False)
        self.view.clear()
        self.summary.setText('推理失败')
        QMessageBox.critical(self, '推理失败', msg)
        self.win.status('推理失败')

    # ------------------------------------------------------------------
    def save_figure(self):
        """
        存对比图。

        有真值时走 metrics.save_prediction_grid —— 它带中文字幕，比界面上的
        无字长图更适合放进报告。这条路径是安全的：本文件全程没有 import pyplot，
        它内部那句 matplotlib.use('Agg') 只改 rcParams 的默认后端，
        动不了已经建好的 FigureCanvasQTAgg 实例（已实测：存完之后曲线画布仍能重绘）。

        没有真值时必须换一条路。曾经这里给 gt 塞了一块全零，结果错误图会把
        预测里所有非「背景」（下标 0，全零恰好是背景）的像素标成品红
        —— 实测占 85.7% 的面积。
        用户压根没提供真值，却会看到一张读起来像「全错」的图，这是凭空捏造。
        所以这种情况直接存界面上那张「原图 | 预测」两联图。
        """
        if self.cur_img is None or self._last_pred is None:
            return
        os.makedirs(OUT_GUI, exist_ok=True)
        name = self.cur_key.replace('/', '_').replace('（', '_').replace('）', '')
        path = os.path.join(OUT_GUI, f'infer_{name}_{time.strftime("%Y%m%d_%H%M%S")}.png')
        try:
            if self.cur_gt is None:
                np_to_pixmap(self._last_quad).save(path)
            else:
                save_prediction_grid(
                    [dict(image=self.cur_img.numpy(), gt=self.cur_gt.numpy(),
                          pred=self._last_pred, title=self.cur_key)],
                    path, max_rows=1)
            QMessageBox.information(self, '已保存', path)
        except Exception as e:
            QMessageBox.critical(self, '保存失败', str(e))


# ============================================================================
# 页签 1：测试集评测面板
# ============================================================================

class TabEval(QWidget):

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.worker = None
        self.last = None                 # 上一次完整跑完的结果

        self.combo_split = QComboBox()
        self.combo_split.addItems(['test', 'val', 'train'])
        self.combo_split.currentTextChanged.connect(self.update_estimate)

        self.combo_ckpt = QComboBox()
        for p in list_checkpoints():
            self.combo_ckpt.addItem(os.path.basename(p), p)
        self.combo_ckpt.currentIndexChanged.connect(self.update_estimate)
        self.lbl_ckpt = QLabel('')
        self.lbl_ckpt.setStyleSheet('color:#666;')

        self.spin_w = QSpinBox()
        self.spin_w.setRange(64, 2048)
        self.spin_w.setSingleStep(64)
        self.spin_w.setValue(C.SLIDING_WINDOW)
        self.spin_w.valueChanged.connect(self.update_estimate)

        self.spin_s = QSpinBox()
        self.spin_s.setRange(16, 2048)
        self.spin_s.setSingleStep(64)
        self.spin_s.setValue(C.SLIDING_STRIDE)
        self.spin_s.valueChanged.connect(self.update_estimate)

        self.spin_limit = QSpinBox()
        self.spin_limit.setRange(0, 5000)
        self.spin_limit.setSpecialValueText('全部')
        self.spin_limit.setValue(0)
        self.spin_limit.valueChanged.connect(self.update_estimate)

        self.btn_run = QPushButton('开始评测')
        self.btn_run.clicked.connect(self.run_eval)
        self.btn_cancel = QPushButton('取消')
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_cancel.setEnabled(False)

        self.bar = QProgressBar()
        self.bar.setFormat('%v / %m 张')

        self.chk_vis = QCheckBox('顺便存前 8 张可视化图')
        self.btn_save = QPushButton('保存文本报告')
        self.btn_save.clicked.connect(self.save_report)
        self.btn_save.setEnabled(False)

        self.lbl_eta = QLabel('')
        self.lbl_eta.setStyleSheet('color:#a60;')

        # ---- 汇总表：ALL / Rural / Urban 三行 ----
        self.tbl_domain = QTableWidget(3, 5)
        self.tbl_domain.setHorizontalHeaderLabels(
            ['域', 'mIoU', 'mDice', 'aAcc', 'oAcc'])
        self.tbl_domain.verticalHeader().setVisible(False)
        self.tbl_domain.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_domain.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl_domain.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_domain.setMaximumHeight(140)
        for r, key in enumerate(['ALL'] + C.DOMAINS):
            self.tbl_domain.setItem(r, 0, QTableWidgetItem(key))

        # ---- 逐类表：用下拉切换域 ----
        self.combo_domain = QComboBox()
        self.combo_domain.addItems(['ALL'] + C.DOMAINS)
        self.combo_domain.setEnabled(False)
        self.combo_domain.currentTextChanged.connect(self.refill_classes)
        self.tbl_class = make_class_table()
        self.res_domain = None

        left = QGridLayout()
        left.addWidget(QLabel('划分'), 0, 0)
        left.addWidget(self.combo_split, 0, 1)
        left.addWidget(QLabel('权重'), 1, 0)
        left.addWidget(self.combo_ckpt, 1, 1)
        left.addWidget(QLabel('窗口'), 2, 0)
        left.addWidget(self.spin_w, 2, 1)
        left.addWidget(QLabel('步长'), 3, 0)
        left.addWidget(self.spin_s, 3, 1)
        left.addWidget(QLabel('张数'), 4, 0)
        left.addWidget(self.spin_limit, 4, 1)

        right = QVBoxLayout()
        right.addWidget(self.lbl_ckpt)
        right.addWidget(self.lbl_eta)
        right.addWidget(self.chk_vis)
        right.addWidget(self.btn_save)
        right.addStretch(1)

        bar = QHBoxLayout()
        bar.addWidget(self.btn_run)
        bar.addWidget(self.btn_cancel)
        bar.addWidget(self.bar, 1)

        box = QGroupBox('参数')
        bl = QHBoxLayout(box)
        bl.addLayout(left)
        bl.addLayout(right, 1)

        root = QVBoxLayout(self)
        root.addWidget(box)
        root.addLayout(bar)
        root.addWidget(QLabel('分域总览'))
        root.addWidget(self.tbl_domain)
        head = QHBoxLayout()
        head.addWidget(QLabel('逐类明细   域'))
        head.addWidget(self.combo_domain)
        head.addStretch(1)
        root.addLayout(head)
        root.addWidget(self.tbl_class, 1)

        self.refresh_ckpt_label()
        self.update_estimate()

    # ------------------------------------------------------------------
    def refresh_ckpt_label(self):
        p = self.combo_ckpt.currentData()
        if not p:
            return
        ep, best = ckpt_summary(p)
        best_s = '?' if (isinstance(best, float) and np.isnan(best)) else f'{best:.4f}'
        self.lbl_ckpt.setText(f'断点内记录：第 {ep} 轮，验证集 best mIoU {best_s}'
                              f'（中心裁剪口径，非滑窗，仅供参考）')

    def update_estimate(self):
        n = self.spin_limit.value()
        if n == 0:
            try:
                n = len(get_keys(self.combo_split.currentText()))
            except Exception:
                n = 0
        per = len(_window_starts(1024, self.spin_w.value(),
                                min(self.spin_s.value(), self.spin_w.value()))) ** 2
        mins = n * per * SEC_PER_WINDOW / 60
        eta = f'约 {mins:.1f} 分钟' if mins < 90 else f'约 {mins / 60:.1f} 小时'
        self.lbl_eta.setText(f'{n} 张 x 每张 {per} 个窗口 -> {eta}')
        self.refresh_ckpt_label()

    # ------------------------------------------------------------------
    def run_eval(self):
        ckpt = self.combo_ckpt.currentData()
        if not ckpt:
            QMessageBox.warning(self, '没有权重', 'checkpoints 目录下没找到 .pt 文件。')
            return

        n = self.spin_limit.value()
        if n == 0:
            n = len(get_keys(self.combo_split.currentText()))
        if n > 60:
            per = len(_window_starts(1024, self.spin_w.value())) ** 2
            mins = n * per * SEC_PER_WINDOW / 60
            ok = QMessageBox.question(
                self, '确认',
                f'将评测 {n} 张，预计耗时 {mins:.0f} 分钟'
                f'（{mins / 60:.1f} 小时）。\n\n'
                f'期间请勿关闭窗口。要开始吗？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ok != QMessageBox.Yes:
                return

        w = self.spin_w.value()
        s = min(self.spin_s.value(), w)
        self.win.set_busy(True)
        self.btn_cancel.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.res_domain = None
        self.tbl_class.clearContents()
        self.bar.setValue(0)
        for r in range(3):
            for c in range(1, 5):
                self.tbl_domain.setItem(r, c, QTableWidgetItem('…'))

        self.worker = EvalWorker(ckpt, self.win.cached_model(ckpt),
                                 self.combo_split.currentText(),
                                 self.spin_limit.value(), w, s,
                                 8 if self.chk_vis.isChecked() else 0)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.win.status('正在停止…（等当前这张跑完）')

    def on_progress(self, done_n, total):
        self.bar.setMaximum(total)
        self.bar.setValue(done_n)
        self.win.status(f'评测中 {done_n}/{total} 张…')

    def on_done(self, res):
        self.win.set_busy(False)
        self.btn_cancel.setEnabled(False)
        if res.get('model') is not None:
            self.win.put_model(res['ckpt'], res['model'])

        dm = res['dm']
        # 注意：不要调 dm.report()，它即使 quiet=True 也会往 stdout 打整张表。
        # GUI 里只取数，走 groups[key].compute()。
        self.res_domain = {k: dm.groups[k].compute() for k in ['ALL'] + C.DOMAINS}

        for r, key in enumerate(['ALL'] + C.DOMAINS):
            d = self.res_domain[key]
            vals = ([d['miou'], d['mdice'], d['aacc'], d['oacc']]
                    if d['total_pixels'] > 0 else [None] * 4)
            for c, v in enumerate(vals, start=1):
                item = QTableWidgetItem(fmt(v))
                item.setTextAlignment(Qt.AlignCenter)
                self.tbl_domain.setItem(r, c, item)

        self.combo_domain.setEnabled(True)
        self.refill_classes(self.combo_domain.currentText())

        tag = '已取消' if res['cancelled'] else '完成'
        self.win.status(f"{tag}：{res['split']} 集 {res['n']}/{res['total']} 张")
        # 取消时结果是半份的，不许当成正式成绩存下来
        self.btn_save.setEnabled(not res['cancelled'])
        self.last = None if res['cancelled'] else res

        if res['vis']:
            os.makedirs(OUT_GUI, exist_ok=True)
            path = os.path.join(
                OUT_GUI, f"eval_{res['split']}_{time.strftime('%Y%m%d_%H%M%S')}.png")
            try:
                save_prediction_grid(res['vis'], path, max_rows=len(res['vis']))
                self.win.status(f'{tag}，可视化已写出 {path}')
            except Exception as e:
                QMessageBox.warning(self, '可视化保存失败', str(e))

        if res['cancelled']:
            QMessageBox.information(
                self, '已取消',
                f"已跑完 {res['n']}/{res['total']} 张，上表是这 {res['n']} 张的指标。\n\n"
                f"这是半份结果，「保存文本报告」已禁用。")

    def on_failed(self, msg):
        self.win.set_busy(False)
        self.btn_cancel.setEnabled(False)
        QMessageBox.critical(self, '评测失败', msg)
        self.win.status('评测失败')

    # ------------------------------------------------------------------
    def refill_classes(self, key):
        if self.res_domain is None:
            return
        fill_class_table(self.tbl_class, self.res_domain[key])

    def save_report(self):
        """按 outputs/sliding_test/report.txt 的版式写一份。"""
        if self.last is None:
            return
        os.makedirs(OUT_GUI, exist_ok=True)
        path = os.path.join(
            OUT_GUI, f"report_{self.last['split']}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        r = self.res_domain
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"权重: {self.last['ckpt']}\n")
            f.write(f"划分: {self.last['split']}   "
                    f"滑窗 {self.last['window']}/{self.last['stride']}\n")
            f.write(f"张数: {self.last['n']}\n\n")
            f.write(f"{'域':<8}{'mIoU':>10}{'mDice':>10}{'aAcc':>10}{'oAcc':>10}\n")
            for k in ['ALL'] + C.DOMAINS:
                f.write(f"{k:<8}{r[k]['miou']:>10.4f}{r[k]['mdice']:>10.4f}"
                        f"{r[k]['aacc']:>10.4f}{r[k]['oacc']:>10.4f}\n")
            for k in ['ALL'] + C.DOMAINS:
                f.write(f"\n[{k}] {'类别':<8}{'IoU':>10}{'Dice':>10}{'占比':>10}\n")
                for c, name in enumerate(C.CLASS_NAMES):
                    share = r[k]['gt_pixels'][c] / max(r[k]['total_pixels'], 1)
                    f.write(f"{name:<8}{fmt(r[k]['iou'][c]):>10}"
                            f"{fmt(r[k]['dice'][c]):>10}{share * 100:>9.2f}%\n")
        QMessageBox.information(self, '已保存', path)


# ============================================================================
# 页签 2：训练曲线查看
# ============================================================================

# (csv 列名, 显示名, 量级分组)
#
# 分组决定各自画在哪张子图里。'main' 是 0~1 的指标，可以共用一个 y 轴；
# 'aux' 的量级和它们差太远，混在一起会把主指标压成贴着底边的平线 ——
# 实测 train_loss(0.05~5.15) 和 val_miou(0.28~0.54) 同轴时，
# mIoU 曲线只占纵轴高度的 9.6%，趋势完全看不出来。所以 aux 各占一张子图。
METRICS = [
    ('val_miou', '验证 mIoU', 'main'),
    ('val_mdice', '验证 mDice', 'main'),
    ('val_aacc', '验证 aAcc', 'main'),
    ('val_oacc', '验证 oAcc', 'main'),
    ('val_miou_rural', 'Rural mIoU', 'main'),
    ('val_miou_urban', 'Urban mIoU', 'main'),
    ('val_iou_barren', '裸地 IoU', 'main'),
    ('train_loss', '训练 loss', 'aux'),
    ('lr', '学习率', 'aux'),
]


def read_log_csv(path):
    """
    读一个训练日志。返回 (rows, 是否含单列出来的那一类)。

    关于列名的历史包袱：这一列现在叫 val_iou_barren（裸地，下标 4）。
    类名错位时期它叫 val_iou_forest，那批旧日志里装的也是同一个下标 4
    的数据 —— 只是当时的名字是错的。所以**两个列名都要认**，
    否则旧日志会被当成「缺列」，曲线直接断掉。
    （列名到底取哪个不影响数值，见 config.CLASS_NAMES 上方注释。）

    更旧的纯 Dice 那一轮连这一列都没有（那时该类 IoU 恒为 0，加了列才看得出
    修复有没有生效）。缺列时该字段记 None，画图时跳过该点 —— **不能当成 0**，
    那会凭空造出一条从 0 开始的曲线。
    """
    COL = 'val_iou_barren'
    LEGACY = 'val_iou_forest'

    def pick(fields):
        for c in (COL, LEGACY):
            if c in fields:
                return c
        return None

    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []
        col = pick(fields)
        for raw in rd:
            v = raw.get(col) if col else None
            rows.append(dict(
                epoch=int(raw['epoch']),
                lr=float(raw['lr']),
                train_loss=float(raw['train_loss']),
                val_miou=float(raw['val_miou']),
                val_mdice=float(raw['val_mdice']),
                val_aacc=float(raw['val_aacc']),
                val_oacc=float(raw['val_oacc']),
                val_miou_rural=float(raw['val_miou_rural']),
                val_miou_urban=float(raw['val_miou_urban']),
                val_iou_barren=(None if v is None or v.strip() in ('', '--')
                                else float(v)),
            ))
    return rows, (col is not None)


class TabCurves(QWidget):

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.files = []          # [(path, rows, has_barren)]

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.NoSelection)
        self.list.itemChanged.connect(self.replot)
        self.list.setMaximumWidth(320)

        self.checks = []
        box = QGroupBox('指标')
        gl = QGridLayout(box)
        for i, (key, label, _grp) in enumerate(METRICS):
            cb = QCheckBox(label)
            cb.setChecked(key in ('val_miou', 'val_miou_rural', 'val_miou_urban'))
            cb.stateChanged.connect(self.replot)
            self.checks.append((key, cb))
            gl.addWidget(cb, i % 5, i // 5)

        self.fig = Figure(figsize=(8, 5), dpi=100, layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.fig)

        btn_reload = QPushButton('重新扫描 logs/')
        btn_reload.clicked.connect(self.reload)
        btn_save = QPushButton('导出图片')
        btn_save.clicked.connect(self.export)
        btns = QHBoxLayout()
        btns.addWidget(btn_reload)
        btns.addWidget(btn_save)
        btns.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(QLabel('日志文件（勾选要画的）'))
        left.addWidget(self.list, 1)
        left.addLayout(btns)

        right = QVBoxLayout()
        right.addWidget(box)
        right.addWidget(self.canvas, 1)

        root = QHBoxLayout(self)
        root.addLayout(left)
        root.addLayout(right, 1)

        self.reload()

    # ------------------------------------------------------------------
    def reload(self):
        """
        扫描 logs/train_*.csv。

        续训会新建日志文件（第 1~15 轮、16~22 轮、23~30 轮分散在三个文件里），
        所以必须按 epoch 合并，不能只看最新那个。
        """
        paths = sorted(glob.glob(os.path.join(C.LOG_DIR, 'train_*.csv')))
        self.list.blockSignals(True)
        self.list.clear()
        self.files = []
        for p in paths:
            try:
                rows, has_barren = read_log_csv(p)
            except Exception:
                continue
            if not rows:
                continue
            self.files.append((p, rows, has_barren))
            it = QListWidgetItem(
                f"{os.path.basename(p)}   ({len(rows)} 轮, "
                f"{'新' if has_barren else '旧·无裸地列'})")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            # 默认只勾 CE 修复之后的那批：旧那批裸地恒为 0，混在一起曲线会被拉花
            it.setCheckState(Qt.Checked if has_barren else Qt.Unchecked)
            self.list.addItem(it)
        self.list.blockSignals(False)
        self.replot()

    def _merged(self):
        """把所有勾上的文件按 epoch 合并成一条序列。"""
        merged = {}
        for i, (path, rows, _) in enumerate(self.files):
            if self.list.item(i).checkState() != Qt.Checked:
                continue
            for r in rows:
                merged[r['epoch']] = r
        return [merged[e] for e in sorted(merged)]

    def replot(self, *_):
        rows = self._merged()
        self.fig.clear()

        if not rows:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, '没有勾选任何日志文件', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw_idle()
            return

        _setup_cjk_font()          # 复用 metrics.py 的中文字体探测

        state = {key: cb.isChecked() for key, cb in self.checks}
        main = [(k, lb) for k, lb, g in METRICS if state.get(k) and g == 'main']
        aux = [(k, lb) for k, lb, g in METRICS if state.get(k) and g == 'aux']

        epochs = [r['epoch'] for r in rows]

        # 一张子图放一组量级相同的量。'aux' 的各占一张 —— lr(1e-3)、
        # train_loss(0~5)、mIoU(0~1) 三者差好几个数量级，挤一根轴上什么都看不清。
        panels = ([('指标', main)] if main else []) + [(lb, [(k, lb)]) for k, lb in aux]

        if not panels:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, '没有勾选任何指标', ha='center', va='center')
            ax.set_xticks([]); ax.set_yticks([])
            self.canvas.draw_idle()
            return

        n = len(panels)
        axes = []
        for i, (ylabel, series) in enumerate(panels):
            ax = self.fig.add_subplot(n, 1, i + 1, sharex=axes[0] if axes else None)
            axes.append(ax)
            for key, label in series:
                # 缺列的老日志里 val_iou_barren 是 None，只跳过这些点，
                # 不能当成 0 —— 那会凭空造出一条从 0 起步的曲线
                xs = [e for e, r in zip(epochs, rows) if r.get(key) is not None]
                ys = [r[key] for r in rows if r.get(key) is not None]
                if xs:
                    ax.plot(xs, ys, marker='o', markersize=3, linewidth=1.6,
                            label=label)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
        axes[-1].set_xlabel('轮次')

        # 里程碑竖线画在每张子图上，横向对齐便于对照
        for ax in axes:
            # warmup 结束的位置：第 5 轮裸地 IoU 掉到 0.042 就是在这儿，
            # 标出来以后不会再把 warmup 的瞬时震荡误读成模型出了问题。
            ax.axvline(C.WARMUP_EPOCHS, color='#888', linestyle=':', linewidth=1)
            # best 轮：从 best.pt 里读，不靠曲线自己找
            try:
                ep, _ = ckpt_summary(os.path.join(C.CKPT_DIR, 'best.pt'))
                if isinstance(ep, int) and epochs and min(epochs) <= ep <= max(epochs):
                    ax.axvline(ep, color='#c33', linestyle='--', linewidth=1)
            except Exception:
                pass

        axes[0].annotate('warmup 结束', (C.WARMUP_EPOCHS, axes[0].get_ylim()[0]),
                         fontsize=8, color='#666', rotation=90, va='bottom')

        # best 的星标打在它真正所属的那张图上（主指标那张）
        try:
            ep, best = ckpt_summary(os.path.join(C.CKPT_DIR, 'best.pt'))
            if main and isinstance(ep, int) and epochs and min(epochs) <= ep <= max(epochs):
                axes[0].plot([ep], [best], marker='*', color='#c33', markersize=12)
                axes[0].annotate(f'best 第{ep}轮 {best:.4f}', (ep, best),
                                 fontsize=8, color='#c33',
                                 xytext=(4, 8), textcoords='offset points')
        except Exception:
            pass

        self.canvas.draw_idle()

    def export(self):
        os.makedirs(OUT_GUI, exist_ok=True)
        path = os.path.join(OUT_GUI, f'curves_{time.strftime("%Y%m%d_%H%M%S")}.png')
        self.fig.savefig(path, dpi=110)
        QMessageBox.information(self, '已保存', path)


# ============================================================================
# 主窗口
# ============================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('LoveDA UNet3+ 可视化界面')
        self.resize(1280, 860)

        # 缓存的推理模型。载一次 80MB 的断点要一两秒，同一个权重反复推理时不该重载。
        # 必须连着路径一起缓存 —— 用户在下拉框里换了权重却仍用旧模型，
        # 指标会 silently 对不上，这种错最难查。
        self.model = None
        self.model_path = None

        self.busy = False

        self.tabs = QTabWidget()
        self.tab_infer = TabInfer(self)
        self.tab_eval = TabEval(self)
        self.tab_curves = TabCurves(self)
        self.tabs.addTab(self.tab_infer, '单图推理演示')
        self.tabs.addTab(self.tab_eval, '测试集评测面板')
        self.tabs.addTab(self.tab_curves, '训练曲线查看')
        self.setCentralWidget(self.tabs)

        self.setStatusBar(QStatusBar())
        self.status('就绪。当前权重 ' + os.path.basename(self.ckpt_path()))

    # ------------------------------------------------------------------
    def ckpt_path(self):
        """当前选中的权重。评测页决定，单图页跟着用同一个。"""
        p = self.tab_eval.combo_ckpt.currentData()
        return p or os.path.join(C.CKPT_DIR, 'best.pt')

    def cached_model(self, path):
        """路径对得上才给缓存，否则返回 None 让 worker 重新加载。"""
        return self.model if (self.model is not None and self.model_path == path) else None

    def put_model(self, path, model):
        self.model, self.model_path = model, path

    def status(self, msg):
        self.statusBar().showMessage(msg)

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        """
        关窗口时如果还有任务在跑，先拦一下。

        不处理的话：最后一个窗口关掉 -> app.exec_() 返回 -> sys.exit() 时
        QThread 还在跑，Qt 会打印 "QThread: Destroyed while thread is still
        running" 然后整个进程直接崩。跑了 50 分钟的评测就这么没了，连个提示都没有。
        所以这里先问，确认后发取消信号并等它把当前那张跑完再退。
        """
        if not self.busy:
            event.accept()
            return

        ok = QMessageBox.question(
            self, '后台任务还在跑',
            '评测/推理还没结束，现在关掉会中断它（已完成的部分不会保存）。\n\n'
            '确定要退出吗？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ok != QMessageBox.Yes:
            event.ignore()
            return

        self.status('正在停止后台任务…')
        for tab in (self.tab_infer, self.tab_eval):
            w = getattr(tab, 'worker', None)
            if w is not None and w.isRunning():
                w.cancel()
                w.wait(60000)          # 等它跑完手上这一张，别硬杀
        event.accept()

    def set_busy(self, busy):
        """
        全局互斥：同一时刻只允许一个后台任务。

        两个线程各开 16 个 torch 线程会互相抢核，界面反而更卡；
        所以跑起来就把两个页签的启动按钮都禁掉。
        """
        self.busy = busy
        self.tab_infer.btn_run.setEnabled(not busy)
        self.tab_eval.btn_run.setEnabled(not busy)
        self.tab_eval.combo_ckpt.setEnabled(not busy)
        # setOverrideCursor 是压栈、restore 是弹栈，必须严格成对
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()


def install_excepthook(win):
    """
    Qt 槽函数里抛出的异常默认会被吞掉或直接 abort 进程，
    装个钩子至少让错误弹出来、能复制。
    """
    def hook(etype, value, tb):
        msg = ''.join(traceback.format_exception(value))
        sys.__excepthook__(etype, value, tb)
        QMessageBox.critical(win, '未处理的异常', msg)
    sys.excepthook = hook


def main():
    # 保险丝：torch 必须在 QApplication 之前加载，否则段错误（见文件顶部注释）。
    # 正常跑 python gui.py 时必然成立；这句是防着以后有人把 import torch 挪进函数，
    # 那种改法会得到进程直接消失、没有任何报错的现象。
    assert 'torch' in sys.modules, \
        "torch 必须在创建 QApplication 之前 import，否则会段错误（见文件顶部注释）"

    # 高 DPI 必须在 QApplication 构造之前设置
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    C.ensure_dirs()
    os.makedirs(OUT_GUI, exist_ok=True)
    build_splits(verbose=False)          # 缺 splits.json 时生成，已有则静默读取

    app = QApplication(sys.argv)
    app.setFont(QFont('Microsoft YaHei', 10))
    _setup_cjk_font()                    # matplotlib 的中文，曲线图要用

    win = MainWindow()
    install_excepthook(win)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
