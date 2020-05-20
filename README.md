# Cyclic PhotoWCT

## Semester Project Spring 2020

## ETH Zürich - Computer Vision Lab
![GitHub Logo](/images/ethzlogo.png)

###Author: Joan Mihali

This project is based on the work of Li et. al [A Closed-form Solution to Photorealistic Image Stylization](https://arxiv.org/abs/1802.06474) and on its [code](https://github.com/NVIDIA/FastPhotoStyle). 

To run the code you need to download the pytorch VGG19-Model from [Simonyan and Zisserman, 2014](https://arxiv.org/abs/1409.1556) and the pretrained PhotoWCT and CyclicPhotoWCT models by running: 

`sh download_models.sh`

To run a stylization process, run 'main_cyclic.py'.

To train your own model, download a dataset and store the images in ./dataset/train/ and run 'train_cyclic_model.py'. You can download the dataset that I used to train these models from [here](http://images.cocodataset.org/zips/val2017.zip). 
