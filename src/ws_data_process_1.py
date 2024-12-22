# coding=utf-8
# for Weak Supervised Learning Algorithm
import os
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
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


def enlarge_dataset(path):
    for file in os.listdir(path):
        if '.png' in file:
            read_path = os.path.join(path, file)
            img1 = Image.open(read_path)

            out1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            out2 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            out3 = img1.transpose(Image.ROTATE_90)
            out4 = img1.transpose(Image.ROTATE_180)
            out5 = img1.transpose(Image.ROTATE_270)

            out1_path = os.path.join(path, 'flr_' + file)
            out1.save(out1_path)
            out2_path = os.path.join(path, 'ftd_' + file)
            out2.save(out2_path)
            out3_path = os.path.join(path, 'rt90_' + file)
            out3.save(out3_path)
            out4_path = os.path.join(path, 'rt180_' + file)
            out4.save(out4_path)
            out5_path = os.path.join(path, 'rt270_' + file)
            out5.save(out5_path)
    print("data successfully augment")


def segment_train(upscale=4, lr_size=32):
    make_dirs(seg_train_x)
    make_dirs(seg_train_y)
    make_dirs(seg_train_z)

    lr_stride = lr_size
    hr_size = lr_size * upscale
    for file in os.listdir(train_y):
        print(file)
        read_path = os.path.join(train_y, file)
        y = Image.open(read_path)
        y = np.array(y)
        h, w, _ = y.shape
        read_path = os.path.join(train_z, file)
        z = Image.open(read_path)
        z = np.array(z)

        for i in range(0, h - lr_size, lr_stride):
            for j in range(0, w - lr_size, lr_stride):
                numx = i // lr_stride
                numy = j // lr_stride
                if (numx+1)*(numy+1) <= 120:
                    # y segment
                    sub_input = y[i:i + lr_size, j:j + lr_size]
                    sub_input = Image.fromarray(sub_input.astype("uint8"))

                    filename = str(file.split('.')[0]) + '_' + str(numx) + '_' + str(numy) + '.png'
                    y_name = os.path.join(seg_train_y, filename)
                    sub_input.save(y_name)

                    # z segment
                    upi = i * upscale
                    upj = j * upscale
                    sub_label = z[upi:upi + hr_size, upj:upj + hr_size]
                    sub_label = Image.fromarray(sub_label.astype("uint8"))
                    filename = str(file.split('.')[0]) + '_' + str(numx) + '_' + str(numy) + '.png'
                    z_name = os.path.join(seg_train_z, filename)
                    sub_label.save(z_name)
                else:
                    pass

    for file in os.listdir(train_x):
        read_path = os.path.join(train_x, file)
        x = Image.open(read_path)
        x = np.array(x)
        h, w, _ = x.shape
        for i in range(0, h - lr_size, lr_stride):
            for j in range(0, w - lr_size, lr_stride):
                numx = i // lr_stride
                numy = j // lr_stride
                if (numx+1)*(numy+1) <= 120:
                    # x segment
                    sub_input = x[i:i + lr_size, j:j + lr_size]
                    sub_input = Image.fromarray(sub_input.astype("uint8"))

                    filename = str(file.split('.')[0]) + '_' + str(numx) + '_' + str(numy) + '.png'
                    x_name = os.path.join(seg_train_x, filename)
                    sub_input.save(x_name)
                else:
                    pass
    print("train data successfully seg-ed")


def segment_test():
    make_dirs(seg_test_ori)
    make_dirs(seg_test_step1)
    make_dirs(seg_test_step2)

    for file in os.listdir(test):
        read_path = os.path.join(test, file)
        img = Image.open(read_path)
        savepath = os.path.join(seg_test_ori, file)
        img.save(savepath)
    print("test data successfully seg-ed")


def get_seg_train_txt():
    txt_name = 'seg_train_path1.txt'
    pathx = []
    pathy = []
    for file in os.listdir(seg_train_x):
        if ('flr' not in file) and ('ftd'not in file) and ('rt' not in file):
            pathx_add = os.path.join(seg_train_x, file)
            pathx.append(pathx_add)
    for file in os.listdir(seg_train_y):
        if ('flr' not in file) and ('ftd'not in file) and ('rt' not in file):
            pathy_add = os.path.join(seg_train_y, file)
            pathy.append(pathy_add)
    f = open(txt_name, "w")
    for i in range(len(pathy)):
        print(pathx[i], pathy[i], file=f)
    f.close()
    print("successfully train txt-ed")


def get_seg_test_txt():
    txt_name = 'seg_test_path1.txt'
    path_test = []
    for file in os.listdir(seg_test_ori):
        path_test_add = os.path.join(seg_test_ori, file)
        path_test.append(path_test_add)
    f = open(txt_name, "w")
    for i in range(len(path_test)):
        print(path_test[i], file=f)
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
            imgs.append((words[0], words[1]))
            self.imgs = imgs
            self.transform = transform
            self.loader = loader

    def __getitem__(self, index):
        x, y = self.imgs[index]
        x = self.loader(x)
        y = self.loader(y)
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

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
#     transform_ = Compose([ToTensor(), Normalize((.5, .5, .5), (.5, .5, .5))])

    train_data = MyTrainDataset(txt_path='seg_train_path1.txt',
                                transform=transform_)
    test_data = MyTestDataset(txt_path='seg_test_path1.txt',
                              transform=transform_)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, shuffle=False)
    return train_loader, test_loader


def check_data(is_train=True):
    tr_data = MyTrainDataset(txt_path='seg_train_path1.txt')
    ts_data = MyTestDataset(txt_path='seg_test_path1.txt')

    if is_train:
        img1 = tr_data[0][0]
        img2 = tr_data[0][1]
        plt.figure()
        plt.subplot(211)
        plt.imshow(img1)
        plt.subplot(212)
        plt.imshow(img2)
        plt.show()
    else:
        img1 = ts_data[0]
        plt.figure()
        plt.subplot(111)
        plt.imshow(img1)
        plt.show()


def remove_file(path):
    for file in os.listdir(path):
        # if 'flr' in file:
        #     file_path = os.path.join(path, file)
        #     if os.path.exists(file_path):
        #         os.remove(file_path)
        #     else:
        #         print('no such file')
        file_path = os.path.join(path, file)
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print('no such file')
    print("successfully removed")

    
if __name__ == '__main__':
    # remove_file(train_x)
    # remove_file(train_y)
    # remove_file(train_z)

    # enlarge_dataset(train_x)
    # enlarge_dataset(train_y)
    # enlarge_dataset(train_z)

    # remove_file(seg_train_x)
    # remove_file(seg_train_y)
    # remove_file(seg_train_z)

    segment_train(upscale=4, lr_size=32)
    segment_test()

    get_seg_train_txt()
    get_seg_test_txt()

    # check_data(is_train=True)
    # check_data(is_train=False)
    print("data process end")
