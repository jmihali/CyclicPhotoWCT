from __future__ import print_function
import torch
import process_stylization_2maps
from perceptual_loss import mse_loss_images, content_loss_images
import torch
from photo_wct_2maps import CyclicPhotoWCT
from photo_smooth import Propagator


# cyclic model using two mappers

def run_cyclic_photoWCT(photo_wct_path, content_image_path, style_image_path, stylized_image_path, reversed_image_path,
                        cuda, do_smoothing=False):
    if cuda:
        p_wct.load_state_dict(torch.load(photo_wct_path))
    else:
        p_wct.load_state_dict(torch.load(photo_wct_path, map_location=torch.device('cpu')))

    process_stylization_2maps.cyclic_stylization(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        content_image_path=content_image_path,
        style_image_path=style_image_path,
        content_seg_path=[],
        style_seg_path=[],
        stylized_image_path=stylized_image_path,
        reversed_image_path=reversed_image_path,
        cuda=cuda,
        save_intermediate=True,
        no_post=True,
        do_smoothing=do_smoothing  # added by me, to include or exclude smoothing part
    )


def run_experiment(p_wct_path, message, content_image_path, style_image_path, output_path, cuda, do_smoothing=False):
    print("Network with", message)
    stylized_image_path = output_path + '/stylized_%s.jpg' % message
    reversed_image_path = output_path + '/reversed_%s.jpg' % message
    print('Content image', content_image_path)
    print('Style image', style_image_path)
    run_cyclic_photoWCT(p_wct_path, content_image_path, style_image_path, stylized_image_path, reversed_image_path,
                        cuda, do_smoothing)
    print('Stylized image', stylized_image_path)
    print('Reversed image', reversed_image_path)
    print("MSE loss between content and reversed image:",
          mse_loss_images(content_image_path, reversed_image_path).item())
    print("Content loss between content and reversed image:",
          content_loss_images(content_image_path, reversed_image_path).item())
    print("=" * 15)


# Load model
p_wct = CyclicPhotoWCT()
p_pro = Propagator(beta=0.7)
cuda = torch.cuda.is_available()
do_smoothing = False  # change this flag if you want to apply or no the smoothing module

if cuda:
    p_wct.cuda(0)

p_wct_paths = ['./PhotoWCTModels/cyclic_photo_wct_2maps.pth']
content_image_paths = ['./images/opernhaus.jpg', './images/forest_summer.jpg', './images/pyramid_egypt.jpg']
style_image_paths = ['./images/schonbrunn.jpg', './images/forest_winter.jpg', './images/pyramid_maya.jpg']
output_paths = ['./results/opernhaus', './results/forest', './results/pyramid']

for content_image_path, style_image_path, output_path in zip(content_image_paths, style_image_paths, output_paths):
    for p_wct_path, message in zip(p_wct_paths, ['2_mappers']):
        run_experiment(p_wct_path, message, content_image_path, style_image_path, output_path, cuda, do_smoothing)
    print('='*15)
    print('='*15)