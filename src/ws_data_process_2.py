# coding=utf-8
# for Weak Supervised Learning Algorithm
import os
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# path statement
root = os.getcwd()
dataset = os.path.join(root, 'dataset')
dataset_NTIRE = os.path.join(dataset, 'NTIRE2020')
train_x = os.path.join(dataset_NTIRE, 'train_x')
train_y = os.path.join(dataset_NTIRE, 'train_y')
train_z = os.path.join(dataset_NTIRE, 'train_z')
test = os.path.join(dataset_NTIRE, 'test')

seg_root = os.path.join(root, 'data_ws')
seg_train = os.path.join(seg_root, 'train')
seg_train_x = os.path.join(seg_train, 'train_x')
seg_train_y = os.path.join(seg_train, 'train_y')
seg_train_z = os.path.join(seg_train, 'train_z')
fake_y = os.path.join(seg_train, 'fake_y')
seg_test = os.path.join(seg_root, 'test')
seg_test_ori = os.path.join(seg_test, 'oringin')
seg_test_step1 = os.path.join(seg_test, 'step1')
seg_test_step2 = os.path.join(seg_test, 'step2')


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def get_seg_train_txt():
    txt_name = 'seg_train_path2.txt'
    pathx = []
    pathy = []
    pathfy = []
    pathz = []
    for file in os.listdir(seg_train_x):
        if ('flr' not in file) and ('ftd' not in file) and ('rt' not in file):
            if ".png" in file:
                pathx_add = os.path.join(seg_train_x, file)
                pathx.append(pathx_add)
    for file in os.listdir(seg_train_y):
        if ('flr' not in file) and ('ftd' not in file) and ('rt' not in file):
            if ".png" in file:
                pathy_add = os.path.join(seg_train_y, file)
                pathy.append(pathy_add)
    for file in os.listdir(fake_y):
        if ('flr' not in file) and ('ftd' not in file) and ('rt' not in file):
            if ".png" in file:
                pathfy_add = os.path.join(fake_y, file)
                pathfy.append(pathfy_add)
    for file in os.listdir(seg_train_z):
        if ('flr' not in file) and ('ftd' not in file) and ('rt' not in file):
            if ".png" in file:
                pathz_add = os.path.join(seg_train_z, file)
                pathz.append(pathz_add)
    f = open(txt_name, "w")
    for i in range(min(len(pathx), len(pathy), len(pathfy), len(pathz))):
        print(pathx[i], pathy[i], pathfy[i], pathz[i], file=f)
    f.close()
    print("successfully train txt-ed")


def get_seg_test_txt():
    txt_name = 'seg_test_path2.txt'
    path = []
    for file in os.listdir(seg_test_step1):
        if ('flr' not in file) and ('ftd' not in file) and ('rt' not in file):
            path_add = os.path.join(seg_test_step1, file)
            path.append(path_add)
    f = open(txt_name, "w")
    for i in range(len(path)):
        print(path[i], file=f)
    f.close()
    print("successfully test txt-ed")


def rgb_loader(imgpath):
    img = Image.open(imgpath)
    return img


class MyTrainDataset(Dataset):
    def __init__(self, txt_path, transform=None, loader=rgb_loader):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1], words[2], words[3]))
            self.imgs = imgs
            self.transform = transform
            self.loader = loader

    def __getitem__(self, index):
        x, y, y_fake, z = self.imgs[index]
        x = self.loader(x)
        y = self.loader(y)
        y_fake = self.loader(y_fake)
        z = self.loader(z)
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
            y_fake = self.transform(y_fake)
            z = self.transform(z)
        return x, y, y_fake, z

    def __len__(self):
        return len(self.imgs)


class MyTestDataset(Dataset):
    def __init__(self, txt_path, transform=None, loader=rgb_loader):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0]))
            self.imgs = imgs
            self.transform = transform
            self.loader = loader

    def __getitem__(self, index):
        img = self.imgs[index]
        img = self.loader(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


def load_seg_data(batch_size=32):
    transform_ = Compose([ToTensor()])

    train_data = MyTrainDataset(txt_path='seg_train_path2.txt',
                                transform=transform_)
    test_data = MyTestDataset(txt_path='seg_test_path2.txt',
                              transform=transform_)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, shuffle=False)
    return train_loader, test_loader


def check_data(is_train=True):
    tr_data = MyTrainDataset(txt_path='seg_train_path2.txt')
    ts_data = MyTestDataset(txt_path='seg_test_path2.txt')

    if is_train:
        img1 = tr_data[0][0]
        img2 = tr_data[0][1]
        img3 = tr_data[0][2]
        img4 = tr_data[0][3]
        plt.figure()
        plt.subplot(221)
        plt.imshow(img1)
        plt.subplot(222)
        plt.imshow(img2)
        plt.subplot(223)
        plt.imshow(img3)
        plt.subplot(224)
        plt.imshow(img4)
        plt.show()
    else:
        img1 = ts_data[0]
        plt.figure()
        plt.subplot(111)
        plt.imshow(img1)
        plt.show()


if __name__ == '__main__':
    get_seg_train_txt()
    get_seg_test_txt()

    check_data(is_train=True)
    check_data(is_train=False)
    print("data process end")
