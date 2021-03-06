import os
import torch
from torch.nn import MSELoss
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from photo_wct import PhotoWCT
from perceptual_loss import ContentLoss, WeightedMseContentLoss
import numpy as np
from time import perf_counter


# function that sets parameters of a model to req_grad = False
def set_parameter_requires_grad(model, set_value):
    for param in model.parameters():
        param.requires_grad = set_value # true or false


def train_model(model, data_loader, criterion, optimizer, epochs=20, checkpt_epoch = 5):
    print("Epochs = %d "%epochs)
    print("Checkpoint every %d epochs"%checkpt_epoch)
    cnt = 1
    for epoch in range(epochs):
        print('Epoch %d / %d' % (epoch, epochs-1))
        print('-'*15)

        model.train()

        for data in data_loader:
            split_size_or_sections = data[0].shape[0]//2
            if data[0].shape[0] % 2 != 0 :
                split_size_or_sections += 1
            content_img_batch, style_img_batch = torch.split(data[0], split_size_or_sections=split_size_or_sections, dim=0)
            content_img_batch, style_img_batch = content_img_batch.to(device), style_img_batch.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                loss = 0
                for content_img, style_img in zip(content_img_batch, style_img_batch):
                    content_img = content_img.reshape(1, 3, 224, 224)
                    style_img = style_img.reshape(1, 3, 224, 224)
                    content_img, style_img = content_img.to(device), style_img.to(device)

                    stylized_img = model.transform(content_img, style_img, np.asarray([]), np.asarray([]))
                    reversed_img = model.transform(stylized_img, content_img, np.asarray([]), np.asarray([]))

                    loss += criterion(reversed_img, content_img)

                loss.backward()
                optimizer.step()

        # save checkpoint of the model
        if (epoch+1) % checkpt_epoch == 0:
            save_dir = './PhotoWCTModels/'
            torch.save(model.state_dict(), save_dir + 'cyclic_photo_wct_c%d.pth' % cnt)
            print("Checkpoint save")
            cnt += 1


image_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])



root_dir = 'dataset/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_data = datasets.ImageFolder(root_dir, image_transform)
data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

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
criterion = WeightedMseContentLoss(content_loss_weight=1, mse_loss_weight=1700)
#criterion = ContentLoss() # MSELoss() # alternatives to the above dissimilarity constraint

lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)

print("Criterion: MSE & Perceptual loss 'r42', weights 1700 & 1 respectively")
print('Optimizer: Adam | Learning rate = %f'%lr)

t0 = perf_counter()
train_model(model, data_loader, criterion, optimizer)
t1 = perf_counter()
time_elapsed = t1 - t0
t_hrs = time_elapsed // 3600
t_min = (time_elapsed - t_hrs*3600) // 60
t_sec = (time_elapsed - t_hrs*3600 - t_min*60)

print("Time elapsed = %d hours, %d minutes and %d seconds (%d seconds)" % (t_hrs, t_min, t_sec, time_elapsed))
# save the model
save_dir = './PhotoWCTModels/'
torch.save(model.state_dict(), save_dir+'cyclic_photo_wct_2maps.pth')
