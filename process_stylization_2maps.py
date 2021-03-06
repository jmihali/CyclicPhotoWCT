"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
import torch


class ReMapping:
    def __init__(self):
        self.remapping = []

    def process(self, seg):
        new_seg = seg.copy()
        for k, v in self.remapping.items():
            new_seg[seg == k] = v
        return new_seg


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def memory_limit_image_resize(cont_img):
    # prevent too small or too big images
    MINSIZE=256
    MAXSIZE=960
    orig_width = cont_img.width
    orig_height = cont_img.height
    if max(cont_img.width,cont_img.height) < MINSIZE:
        if cont_img.width > cont_img.height:
            cont_img.thumbnail((int(cont_img.width*1.0/cont_img.height*MINSIZE), MINSIZE), Image.BICUBIC)
        else:
            cont_img.thumbnail((MINSIZE, int(cont_img.height*1.0/cont_img.width*MINSIZE)), Image.BICUBIC)
    if min(cont_img.width,cont_img.height) > MAXSIZE:
        if cont_img.width > cont_img.height:
            cont_img.thumbnail((MAXSIZE, int(cont_img.height*1.0/cont_img.width*MAXSIZE)), Image.BICUBIC)
        else:
            cont_img.thumbnail(((int(cont_img.width*1.0/cont_img.height*MAXSIZE), MAXSIZE)), Image.BICUBIC)
    print("Resize image: (%d,%d)->(%d,%d)" % (orig_width, orig_height, cont_img.width, cont_img.height))
    return cont_img.width, cont_img.height


def cyclic_stylization(stylization_module, smoothing_module, content_image_path, style_image_path, content_seg_path,
                style_seg_path, stylized_image_path, reversed_image_path, cuda, save_intermediate, no_post, do_smoothing=False, cont_seg_remapping=None,
                styl_seg_remapping=None):

    # Load image
    with torch.no_grad():
        cont_img = Image.open(content_image_path).convert('RGB')
        styl_img = Image.open(style_image_path).convert('RGB')

        new_cw, new_ch = memory_limit_image_resize(cont_img)
        new_sw, new_sh = memory_limit_image_resize(styl_img)
        cont_pilimg = cont_img.copy()
        cw = cont_pilimg.width
        ch = cont_pilimg.height
        try:
            cont_seg = Image.open(content_seg_path)
            styl_seg = Image.open(style_seg_path)
            cont_seg.resize((new_cw,new_ch),Image.NEAREST)
            styl_seg.resize((new_sw,new_sh),Image.NEAREST)
        except:
            cont_seg = []
            styl_seg = []

        cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
        styl_img = transforms.ToTensor()(styl_img).unsqueeze(0)

        if cuda:
            cont_img = cont_img.cuda(0)
            styl_img = styl_img.cuda(0)
            stylization_module.cuda(0)

        # cont_img = Variable(cont_img, volatile=True)cont_seg
        # styl_img = Variable(styl_img, volatile=True)

        cont_seg = np.asarray(cont_seg)
        styl_seg = np.asarray(styl_seg)
        if cont_seg_remapping is not None:
            cont_seg = cont_seg_remapping.process(cont_seg)
        if styl_seg_remapping is not None:
            styl_seg = styl_seg_remapping.process(styl_seg)

        with Timer("Elapsed time in stylization: %f"):
            stylized_img, reversed_img = stylization_module.transform(cont_img, styl_img, cont_seg, styl_seg)
        if ch != new_ch or cw != new_cw:
            print("De-resize image: (%d,%d)->(%d,%d)" % (new_cw, new_ch, cw, ch))
            stylized_img = nn.functional.upsample(stylized_img, size=(ch, cw), mode='bilinear')
            reversed_img = nn.functional.upsample(reversed_img, size=(ch, cw), mode='bilinear')
        utils.save_image(stylized_img.data.cpu().float(), stylized_image_path, nrow=1, padding=0)
        utils.save_image(reversed_img.data.cpu().float(), reversed_image_path, nrow=1, padding=0)

        if not do_smoothing:
            return stylized_img, reversed_img
        else: # in case you also want to do the smoothing:
            with Timer("Elapsed time in propagation: %f"):
                stylized_smoothed_img = smoothing_module.process(stylized_image_path, content_image_path)
                reversed_smoothed_img = smoothing_module.process(reversed_image_path, content_image_path)

            stylized_smoothed_img.save(stylized_image_path.replace('.jpg', '') + '_smoothed.jpg')
            reversed_smoothed_img.save(reversed_image_path.replace('.jpg', '') + '_smoothed.jpg')
            return stylized_smoothed_img, reversed_smoothed_img
