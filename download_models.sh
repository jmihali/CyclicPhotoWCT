#!/bin/bash
mkdir models
cd models
wget -c --no-check-certificate https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth
cd ..

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1682C6gnIsjhk_-Y5leTjXa8ERXFTs8iB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1682C6gnIsjhk_-Y5leTjXa8ERXFTs8iB" -O PhotoWCTModels.zip && rm -rf /tmp/cookies.txt

unzip PhotoWCTModels.zip
rm PhotoWCTModels.zip

