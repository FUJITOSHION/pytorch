# パッケージのimport
import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms


# 乱数のシード設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# gpuの設定
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ImageTransform():
    '''
    画像の前処理クラス
    '''

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


def make_path_list(phase='train'):
    rootpath = './data/hymenoptera_data/'
    target_path = osp.join(rootpath + phase + '/**/*.jpg')
    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


class Dataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # file list
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or val

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == 'val':
            label = img_path[28:32]

        if label == 'ants':
            label = 0

        elif label == 'bees':
            label = 1

        return img_transformed, label


def train_model(net, dl_dict, criterion, optim, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {} / {}'.format(epoch + 1, num_epochs))
        print('-'*20)

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(dl_dict[phase]):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dl_dict[phase].dataset)
            epoch_acc = epoch_corrects.double()/len(dl_dict[phase].dataset)

            print('{} Loss: {:4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


if __name__ == '__main__':
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_list = make_path_list(phase='train')
    val_list = make_path_list(phase='val')

    train_dataset = Dataset(train_list, transform=ImageTransform(
        size, mean, std), phase='train')

    val_dataset = Dataset(val_list, transform=ImageTransform(
        size, mean, std), phase='val')

    # dataloader
    batch_size = 32
    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dl = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    dl_dict = {'train': train_dl, 'val': val_dl}

    # network

    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    net.train()

    # 損失関数
    criterion = nn.CrossEntropyLoss()

    # paramの設定
    params_to_update = []

    update_param_names = ['classifier.6.weight', 'classifier.6.bias']

    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False

    # 損失関数
    optimizer = optim.SGD(params=params_to_update, lr=1e-4, momentum=0.9)

    # 学習
    num_epochs = 2
    train_model(net, dl_dict, criterion, optimizer, num_epochs=num_epochs)
