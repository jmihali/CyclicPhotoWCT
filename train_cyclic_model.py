import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from photo_wct import PhotoWCT
from perceptual_loss import PerceptualLoss
import numpy as np


# function that sets parameters of a model to req_grad = False
def set_parameter_requires_grad(model, set_value):
    for param in model.parameters():
        param.requires_grad = set_value # true or false


def train_model(model, data_loaders, criterion, optimizer, epochs=40, checkpt_epoch = 1):
    print("Epochs = %d "%epochs)
    print("Checkpoint every %d epochs"%checkpt_epoch)
    cnt = 1
    for epoch in range(epochs):
        print('Epoch %d / %d' % (epoch, epochs-1))
        print('-'*15)

        for phase in ['train']: # only training phase needed for now
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for x in data_loaders[phase]:
                split_size_or_sections = x[0].shape[0]//2
                if x[0].shape[0] % 2 != 0 :
                    split_size_or_sections += 1
                content_img_batch, style_img_batch = torch.split(x[0], split_size_or_sections=split_size_or_sections, dim=0)
                content_img_batch, style_img_batch = content_img_batch.to(device), style_img_batch.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    loss = 0
                    for content_img, style_img in zip(content_img_batch, style_img_batch):
                        content_img = content_img.reshape(1, 3, 224, 224)
                        style_img = style_img.reshape(1, 3, 224, 224)
                        content_img, style_img = content_img.to(device), style_img.to(device)

                        stylized_img = model.transform(content_img, style_img, np.asarray([]), np.asarray([]))
                        reversed_img = model.transform(stylized_img, content_img, np.asarray([]), np.asarray([]))

                        # Implemet perceptual loss:

                        loss += criterion(reversed_img, content_img)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            # implement some kind of print for loss?

        # save the model
        if (epoch+1) % checkpt_epoch == 0:
            save_dir = './PhotoWCTModels/'
            torch.save(model.state_dict(), save_dir + 'cyclic_photo_wct_c%d.pth' % cnt)
            print("Checkpoint save")
            cnt += 1





image_transforms = {
    'train': transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])]),
    'val': transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])}


# root_dir = '/scratch_net/biwidl217/jmihali/coco1/'
root_dir = '/home/jmihali/coco_dummy/'

train = datasets.ImageFolder(root_dir, image_transforms['train'])
# val = datasets.ImageFolder(root_dir, image_transforms['val'])  # no validation set needed for now

# prepare dataloaders

data_generator = {
    'train' : train,
    #'val': val
}

batch_size = 32
data_loader = {
    k: torch.utils.data.DataLoader(data_generator[k], batch_size=batch_size,
                               shuffle=True) for k in ['train']} #removed 'val'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# define & initialize model
model = PhotoWCT()
model.load_state_dict(torch.load('./PhotoWCTModels/photo_wct.pth'))

# set encoders requires_grad=False, i.e. freeze them during training
set_parameter_requires_grad(model.e1, False)
set_parameter_requires_grad(model.e2, False)
set_parameter_requires_grad(model.e3, False)
set_parameter_requires_grad(model.e4, False)


# transfer model to GPU if you have one
model.to(device)

# set criterion to reconstruction loss & define optimizer
criterion = PerceptualLoss() # initially, let's try with mse loss
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

print('Learning rate = %f'%lr)
train_model(model, data_loader, criterion, optimizer)


# save the model
save_dir = './PhotoWCTModels/'
torch.save(model.state_dict(), save_dir+'cyclic_photo_wct.pth')