from __future__ import print_function
import torch
import process_stylization
from photo_wct import PhotoWCT
from photo_smooth import Propagator
from torch.nn import MSELoss


# Cyclic model using two photoWCTs


def run_cyclic_photoWCT(photo_wct_path, content_image_path, style_image_path, stylized_image_path, reversed_image_path, cuda):
    if cuda==1:
        p_wct.load_state_dict(torch.load(photo_wct_path))
    else:
        p_wct.load_state_dict(torch.load(photo_wct_path, map_location=torch.device('cpu')))

    print("Forward Stylization")
    # forward stylization
    process_stylization.stylization(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        content_image_path=content_image_path,
        style_image_path=style_image_path,
        content_seg_path=[],
        style_seg_path=[],
        output_image_path=stylized_image_path,
        cuda=cuda,
        save_intermediate=True,
        no_post=True,
        do_smoothing=False #added by me, to include or exclude smoothing part
    )

    print("Reverse Stylization")
    # reversed stylzation
    process_stylization.stylization(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        content_image_path=stylized_image_path,
        style_image_path=content_image_path,
        content_seg_path=[],
        style_seg_path=[],
        output_image_path=reversed_image_path,
        cuda=cuda,
        save_intermediate=True,
        no_post=True,
        do_smoothing=False #added by me, to include or exclude smoothing part
    )

p_wct = PhotoWCT()
p_pro = Propagator()
cuda=0

if cuda==1:
    p_wct.cuda(0)


content_image_path = './images/opernhaus.jpg'
style_image_path = './images/schonbrunn.jpg'

p_wct_path = './PhotoWCTModels/photo_wct.pth'


print("No tuning: ")
run_cyclic_photoWCT(p_wct_path, content_image_path, style_image_path,
                    './results/opernhaus/stylized_notuning.png', './results/opernhaus/reversed_notuning.png', cuda)
print("=" * 15)

loss = MSELoss()
for cnt in [1, 10, 20, 30, 40]:
    p_wct_path = './PhotoWCTModels/cyclic_photo_wct_c%d.pth'%cnt
    stylized_image_path = './results/opernhaus/stylized%d.png' % cnt
    reversed_image_path = './results/opernhaus/reversed%d.png' % cnt

    print("Number of training epochs : %d" % cnt)

    run_cyclic_photoWCT(p_wct_path, content_image_path, style_image_path, stylized_image_path, reversed_image_path, cuda)

    print("="*15)
