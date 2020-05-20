import os
image_dir = os.getcwd() + '/Images/'
model_dir = os.getcwd() + '/Models/'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image
from torch.autograd import Variable



# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


# In[3]:


# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)


vgg = VGG()
vgg.load_state_dict(torch.load('./models/vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()


# my implementation of the content loss function as an child class of nn.MSELoss
class ContentLoss(nn.MSELoss):
    def __init__(self, content_layers = ['r42'], content_weights = [1]):
        super(ContentLoss, self).__init__()
        self.content_weights = content_weights
        self.content_layers = content_layers
        self.content_loss_fns = [nn.MSELoss()] * len(content_layers)
        if torch.cuda.is_available():
            self.content_loss_fns = [loss_fn.cuda() for loss_fn in self.content_loss_fns]

    def forward(self, input_image, content_image):
        # compute optimization targets
        content_targets = [A.detach() for A in vgg(content_image, self.content_layers)]
        out = vgg(input_image, self.content_layers)
        layer_losses = [self.content_weights[a] * self.content_loss_fns[a](A, content_targets[a]) for a, A in enumerate(out)]
        return sum(layer_losses)


class WeightedMseContentLoss(nn.MSELoss): # my implementation of the weighted mse & content loss function as an child class of nn.MSELoss

    def __init__(self, content_layers = ['r42'], content_weights = [1], mse_loss_weight=100, content_loss_weight=1):
        super(WeightedMseContentLoss, self).__init__()
        self.content_weights = content_weights
        self.content_layers = content_layers
        self.content_loss_fns = [nn.MSELoss()] * len(content_layers)
        self.mse_loss_weight = mse_loss_weight
        self.content_loss_weight = content_loss_weight
        if torch.cuda.is_available():
            self.content_loss_fns = [loss_fn.cuda() for loss_fn in self.content_loss_fns]

    def forward(self, input_image, content_image):
        # compute optimization targets
        content_targets = [A.detach() for A in vgg(content_image, self.content_layers)]
        out = vgg(input_image, self.content_layers)
        layer_losses = [self.content_weights[a] * self.content_loss_fns[a](A, content_targets[a]) for a, A in enumerate(out)]
        return self.content_loss_weight*sum(layer_losses) + self.mse_loss_weight*nn.MSELoss()(input_image, content_image)



def mse_loss_images(img1_path, img2_path): # takes the path of two images and determines their mse loss
    img1 = ToTensor()(Image.open(img1_path))  # unsqueeze to add artificial first dimension
    img2 = ToTensor()(Image.open(img2_path))  # unsqueeze to add artificial first dimension
    img1 = Variable(img1.unsqueeze(0))
    img2 = Variable(img2.unsqueeze(0))
    return nn.MSELoss()(img1, img2)

def content_loss_images(img1_path, img2_path, content_layers=['r42']): # takes the path of two images and determines their content loss
    img1 = ToTensor()(Image.open(img1_path))
    img2 = ToTensor()(Image.open(img2_path))
    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
    img1 = Variable(img1.unsqueeze(0))
    img2 = Variable(img2.unsqueeze(0))
    img1_fm = vgg(img1, content_layers)[0]
    img2_fm = vgg(img2, content_layers)[0]
    return nn.MSELoss()(img1_fm, img2_fm)

