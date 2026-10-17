# -*- coding: utf-8 -*-
"""LoveDA UNet3+ 在线演示

    python app.py                      # 本机 7860 端口
    GRADIO_SERVER_PORT=8080 python app.py
"""

import os

import gradio as gr
import torch

import config as C
from inference import infer, load_model

torch.set_num_threads(int(os.environ.get('TORCH_THREADS', min(8, os.cpu_count() or 4))))

CKPT = os.environ.get('CKPT_PATH', os.path.join(C.CKPT_DIR, 'best.pt'))


def predict(image, quality, progress=gr.Progress()):
    if image is None:
        raise gr.Error('请先上传一张图片')
    stride = 512 if quality.startswith('快速') else 256

    def cb(done, total):
        progress(done / total, desc=f'滑窗推理 {done}/{total}')

    try:
        r = infer(image, CKPT, stride=stride, progress_cb=cb)
    except Exception as e:
        raise gr.Error(f'推理失败: {e}')

    return (r['overlay'], r['colored'], r['original'], r['stats'],
            f"耗时 {r['elapsed']:.1f} 秒")


with gr.Blocks(title='LoveDA UNet3+ 语义分割') as demo:
    gr.Markdown('# LoveDA UNet3+ 语义分割\n'
                '上传一张遥感影像，模型逐像素分为 7 类：背景 / 建筑 / 道路 / 水体 / 裸地 / 林地 / 耕地。')
    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image(type='pil', label='上传影像')
            quality = gr.Radio(['标准（512/256）', '快速（512/512）'],
                               value='标准（512/256）', label='滑窗模式')
            btn = gr.Button('开始分割', variant='primary')
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab('叠加图'):
                    out_overlay = gr.Image(label='原图 + 预测', interactive=False)
                with gr.Tab('预测分割'):
                    out_mask = gr.Image(label='预测', interactive=False)
                with gr.Tab('原图'):
                    out_orig = gr.Image(label='原图', interactive=False)
            stats = gr.Dataframe(headers=['类别', '像素数', '占比'],
                                 datatype=['str', 'number', 'str'],
                                 label='类别统计', interactive=False)
            info = gr.Markdown()
    btn.click(predict, [img, quality],
              [out_overlay, out_mask, out_orig, stats, info])


if __name__ == '__main__':
    load_model(CKPT)
    demo.queue(max_size=8)
    demo.launch(server_name=os.environ.get('GRADIO_SERVER_NAME', '0.0.0.0'),
                server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7860')))
