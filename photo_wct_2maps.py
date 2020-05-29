import torch
import torch.nn as nn
from photo_wct import PhotoWCT

class CyclicPhotoWCT(nn.Module):
    def __init__(self):
        super(CyclicPhotoWCT, self).__init__()
        self.fw = PhotoWCT()
        self.bw = PhotoWCT()

    def transform(self, cont_img, styl_img, cont_seg, styl_seg):
        stylized_img = self.fw.transform(cont_img, styl_img, cont_seg, styl_seg)
        reversed_img = self.bw.transform(stylized_img, cont_img, cont_seg, styl_seg)
        return stylized_img, reversed_img

    def forward(self, *input):
        pass
