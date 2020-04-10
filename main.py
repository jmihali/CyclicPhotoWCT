from __future__ import print_function
import torch
import process_stylization
from photo_wct import PhotoWCT
from photo_smooth import Propagator

# Load model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load('./PhotoWCTModels/photo_wct.pth'))
p_pro = Propagator()
cuda=0

content_image_path = './images/content1.png'
style_image_path = './images/style1.png'
output_image_path = './results/example3.png'

if cuda==1:
    p_wct.cuda(0)

process_stylization.stylization(
    stylization_module=p_wct,
    smoothing_module=p_pro,
    content_image_path=content_image_path,
    style_image_path=style_image_path,
    content_seg_path=[],
    style_seg_path=[],
    output_image_path=output_image_path,
    cuda=cuda,
    save_intermediate=True,
    no_post=True,
    do_smoothing=False #added by me, to include or exclude smoothing part
)
