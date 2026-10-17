# -*- coding: utf-8 -*-
"""LoveDA UNet3+ 在线演示

    python app.py                      # 本机 7860 端口
    GRADIO_SERVER_PORT=8080 python app.py
"""

import os
import time

import gradio as gr
import torch
from PIL import Image

import config as C
from inference import infer, load_model
from metrics import PALETTE

torch.set_num_threads(int(os.environ.get('TORCH_THREADS', min(8, os.cpu_count() or 4))))

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get('CKPT_PATH', os.path.join(C.CKPT_DIR, 'best.pt'))
SAMPLES = sorted(
    os.path.join(ROOT, 'samples', f)
    for f in os.listdir(os.path.join(ROOT, 'samples'))
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
) if os.path.isdir(os.path.join(ROOT, 'samples')) else []

MAX_BATCH = 12

LEGEND = '<div class="legend">' + ''.join(
    f'<span><i style="background:rgb({r},{g},{b})"></i>{name}</span>'
    for name, (r, g, b) in zip(C.CLASS_NAMES, PALETTE)
) + '</div>'

CSS = """
.gradio-container {max-width: 1500px !important; margin: auto;}
.legend {display:flex; flex-wrap:wrap; gap:4px 20px; font-size:14px; margin:2px 0 12px;}
.legend span {display:inline-flex; align-items:center; gap:6px;}
.legend i {width:16px; height:16px; border:1px solid #888; border-radius:3px; display:inline-block;}
"""


def predict(files, quality, progress=gr.Progress()):
    if not files:
        raise gr.Error('请先上传图片，或点击左侧示例图片')
    if isinstance(files, str):
        files = [files]
    if len(files) > MAX_BATCH:
        raise gr.Error(f'一次最多处理 {MAX_BATCH} 张图片')

    stride = 512 if quality.startswith('快速') else 256
    n = len(files)
    gallery, rows, failed = [], [], []
    t0 = time.time()

    for i, path in enumerate(files):
        name = os.path.basename(path)

        def cb(done, total, i=i, name=name):
            progress((i + done / total) / n,
                     desc=f'第 {i + 1}/{n} 张 · 滑窗 {done}/{total} · {name}')

        try:
            with Image.open(path) as im:
                r = infer(im, CKPT, stride=stride, progress_cb=cb)
            gallery.append((r['overlay'], f'{name} · 叠加'))
            gallery.append((r['colored'], f'{name} · 预测'))
            rows.append([name] + [s[2] for s in r['stats']])
        except Exception as e:
            failed.append(f'{name}（{e}）')
            rows.append([name] + ['—'] * C.NUM_CLASSES)

    elapsed = time.time() - t0
    status = f'**共 {n} 张 · 用时 {elapsed:.1f} 秒 · 平均 {elapsed / n:.1f} 秒/张**'
    if failed:
        status += '\n\n⚠️ 失败：' + '；'.join(failed)
    return gallery, rows, status


with gr.Blocks(title='LoveDA UNet3+ 语义分割') as demo:
    gr.Markdown(
        '# LoveDA UNet3+ 语义分割\n'
        '上传一张或多张遥感影像，模型逐像素分为 7 类。支持批量检测，结果如下。'
    )
    gr.HTML(LEGEND)

    with gr.Row():
        with gr.Column(scale=3, min_width=320):
            files = gr.Files(label='上传图片（可多选）',
                             file_count='multiple', file_types=['image'])
            preview = gr.Gallery(label='原图预览', columns=3, height=200,
                                 object_fit='contain', interactive=False,
                                 show_label=False)
            files.change(lambda fs: fs or [], files, preview)
            quality = gr.Radio(['标准（512/256）', '快速（512/512）'],
                               value='标准（512/256）', label='滑窗模式')
            btn = gr.Button('开始检测', variant='primary')

        with gr.Column(scale=7):
            gallery = gr.Gallery(label='分割结果', columns=4, height=460,
                                 object_fit='contain', preview=False)
            stats = gr.Dataframe(headers=['文件'] + C.CLASS_NAMES,
                                 datatype=['str'] * (C.NUM_CLASSES + 1),
                                 max_height=240, wrap=True,
                                 label='各类像素占比', interactive=False)
            status = gr.Markdown()

    btn.click(predict, [files, quality], [gallery, stats, status])

    if SAMPLES:
        gr.Examples(
            examples=[[[p], '快速（512/512）'] for p in SAMPLES],
            inputs=[files, quality], outputs=[gallery, stats, status],
            fn=predict, run_on_click=True, label='示例图片（点击即检测）',
        )


if __name__ == '__main__':
    load_model(CKPT)
    demo.queue(max_size=8)
    demo.launch(server_name=os.environ.get('GRADIO_SERVER_NAME', '0.0.0.0'),
                server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7860')),
                css=CSS)
