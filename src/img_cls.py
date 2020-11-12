import numpy as np
import json
from PIL import Image
# import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models
from torchvision import transforms


class BaseTransform():
    '''
    前処理クラス
    '''
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),  # 短い辺の長さがresizeの大きさになる
            transforms.CenterCrop(resize),  # 画像中央をresize × resizeで切り取り
            transforms.ToTensor(),  # Torchテンソルに変換
            transforms.Normalize(mean, std)  # 色情報の標準化
        ])

    def __call__(self, img):
        return self.base_transform(img)


class Predictor():
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predcit_label_name = self.class_index[str(maxid)][1]
        return predcit_label_name


def main():
    net = models.vgg16(pretrained=True)
    net.eval()

    label_json = json.load(open('./data/imagenet_class_index.json', 'r'))
    predictor = Predictor(label_json)

    img_path = './data/dog.jpeg'
    img = Image.open(img_path)

    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = BaseTransform(resize, mean, std)
    img_trans = transform(img)

    inputs = img_trans.unsqueeze_(0)
    out = net(inputs)
    result = predictor.predict_max(out)

    print(result)


if __name__ == '__main__':
    main()
