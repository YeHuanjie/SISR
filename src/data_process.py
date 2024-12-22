# coding=utf-8
import os
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# path statement
root = os.getcwd()
dataset = os.path.join(root, 'dataset')
dataset_small = os.path.join(dataset, 'dataset_small')
train = os.path.join(dataset_small, 'train')
test = os.path.join(dataset_small, 'test')

train_hr = os.path.join(train, 'hr')
train_lr = os.path.join(train, 'lr')
test_hr = os.path.join(test, 'hr')
test_lr = os.path.join(test, 'lr')

seg_root = os.path.join(root, 'data')
seg_train = os.path.join(seg_root, 'train')
seg_test = os.path.join(seg_root, 'test')

seg_train_hr = os.path.join(seg_train, 'hr')
seg_train_lr = os.path.join(seg_train, 'lr')
seg_test_hr = os.path.join(seg_test, 'hr')
seg_test_lr = os.path.join(seg_test, 'lr')


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def mod_crop(img, factor=4):
    if len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    h = h - h % factor
    w = w - w % factor
    if len(img.shape) == 3:
        image = img[0:h, 0:w, :]
    else:
        image = img[0:h, 0:w]
    return image


def crop(path, upscale=4):
    for file in os.listdir(path):
        read_path = os.path.join(path, file)
        img1 = Image.open(read_path)
        img1 = np.array(img1)
        img1 = mod_crop(img1, upscale)
        img1 = Image.fromarray(img1.astype("uint8"))
        save_path = os.path.join(path, file)
        img1.save(save_path)
    print("crop finished")


def lr_downsample(lr_path, upscale=4):
    # downsample lr image
    for file in os.listdir(lr_path):
        read_path = os.path.join(lr_path, file)
        img = Image.open(read_path)
        h, w = img.size
        img1 = img.resize((h // upscale, w // upscale), Image.ANTIALIAS)
        save_path = os.path.join(lr_path, file)
        img1.save(save_path)
    print("downsample finished")


def enlarge_dataset(path):
    for file in os.listdir(path):
        read_path = os.path.join(path, file)
        img1 = Image.open(read_path)
        h, w = img1.size

        out1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        out2 = img1.transpose(Image.FLIP_TOP_BOTTOM)
        out3 = img1.transpose(Image.ROTATE_90)
        out4 = img1.transpose(Image.ROTATE_180)
        out5 = img1.transpose(Image.ROTATE_270)

        out6 = img1.resize((round(h * 0.9), round(w * 0.9)), Image.ANTIALIAS)
        out7 = img1.resize((round(h * 0.8), round(w * 0.8)), Image.ANTIALIAS)
        out8 = img1.resize((round(h * 0.7), round(w * 0.7)), Image.ANTIALIAS)
        out9 = img1.resize((round(h * 0.6), round(w * 0.6)), Image.ANTIALIAS)

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

        out6_path = os.path.join(path, 'rs9_' + file)
        out6.save(out6_path)
        out7_path = os.path.join(path, 'rs8_' + file)
        out7.save(out7_path)
        out8_path = os.path.join(path, 'rs7_' + file)
        out8.save(out8_path)
        out9_path = os.path.join(path, 'rs6_' + file)
        out9.save(out9_path)


def segment(is_train=True, upscale=4, lr_size=17):
    lr_stride = lr_size
    hr_size = lr_size * upscale
    if is_train:
        hr_path = train_hr
        lr_path = train_lr
    else:
        hr_path = test_hr
        lr_path = test_lr
    make_dirs(seg_train_hr)
    make_dirs(seg_train_lr)
    make_dirs(seg_test_hr)
    make_dirs(seg_test_lr)
    for file in os.listdir(lr_path):
        if ('flr' not in file) and ('ftd' not in file) and ('rt' not in file) and ('rs' not in file):
            read_path = os.path.join(lr_path, file)
            lr = Image.open(read_path)
            lr = np.array(lr)
            h, w, _ = lr.shape

            read_path = os.path.join(hr_path, file)
            hr = Image.open(read_path)
            hr = np.array(hr)

            for x in range(0, h - lr_size, lr_stride):
                for y in range(0, w - lr_size, lr_stride):
                    # lr segment
                    sub_input = lr[x:x + lr_size, y:y + lr_size]
                    sub_input = Image.fromarray(sub_input.astype("uint8"))
                    if is_train:
                        filename = str(file.split('.')[0]) + '_' + str(x // lr_stride) \
                                   + '_' + str(y // lr_stride) + '.png'
                        input_name = os.path.join(seg_train_lr, filename)
                    else:
                        filename = str(file.split('.')[0]) + '_' + str(x // lr_stride) \
                                   + '_' + str(y // lr_stride) + '.png'
                        input_name = os.path.join(seg_test_lr, filename)
                    sub_input.save(input_name)

                    # hr segment
                    upx = x * upscale
                    upy = y * upscale
                    sub_label = hr[upx:upx + hr_size, upy:upy + hr_size]
                    sub_label = Image.fromarray(sub_label.astype("uint8"))
                    if is_train:
                        filename = str(file.split('.')[0]) + '_' + str(x // lr_stride) \
                                   + '_' + str(y // lr_stride) + '.png'
                        label_name = os.path.join(seg_train_hr, filename)
                    else:
                        filename = str(file.split('.')[0]) + '_' + str(x // lr_stride) \
                                   + '_' + str(y // lr_stride) + '.png'
                        label_name = os.path.join(seg_test_hr, filename)
                    sub_label.save(label_name)
        else:
            pass
    print("successfully seg-ed")


def get_seg_txt(is_train=True):
    if is_train:
        hr_path = seg_train_hr
        lr_path = seg_train_lr
        txt_name = 'seg_train_path.txt'
    else:
        hr_path = seg_test_hr
        lr_path = seg_test_lr
        txt_name = 'seg_test_path.txt'
    path0 = []
    path1 = []
    for file in os.listdir(lr_path):
        path0_add = os.path.join(lr_path, file)
        path0.append(path0_add)
    for file in os.listdir(hr_path):
        path1_add = os.path.join(hr_path, file)
        path1.append(path1_add)
    f = open(txt_name, "w")
    for i in range(len(path0)):
        print(path0[i], path1[i], file=f)
    f.close()
    print("successfully txt-ed")


def y_loader(imgpath):
    img = Image.open(imgpath).convert('YCbCr')
    y, _, _ = img.split()
    return y


def rgb_loader(imgpath):
    img = Image.open(imgpath)
    return img


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None, loader=rgb_loader):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        label = self.loader(label)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.imgs)


def load_seg_data(is_train=True, batch_size=32):
    transform_ = Compose([ToTensor()])
    target_transform_ = Compose([ToTensor()])
    # transform_ = Compose([ToTensor(),
    #                       Normalize((.5, .5, .5), (.5, .5, .5))])
    # target_transform_ = Compose([ToTensor(),
    #                              Normalize((.5, .5, .5), (.5, .5, .5))])

    train_data = MyDataset(txt_path='seg_train_path.txt',
                           transform=transform_, target_transform=target_transform_)
    test_data = MyDataset(txt_path='seg_test_path.txt',
                          transform=transform_, target_transform=target_transform_)
    if is_train:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return train_loader
    else:
        valid_loader = DataLoader(test_data, shuffle=False)
        return valid_loader


def check_data(is_train=True):
    tr_data = MyDataset(txt_path='seg_train_path.txt')
    ts_data = MyDataset(txt_path='seg_test_path.txt')

    if is_train:
        img1 = tr_data[0][0]
        img2 = tr_data[0][1]
    else:
        img1 = ts_data[0][0]
        img2 = ts_data[0][1]

    plt.figure()
    plt.subplot(211)
    plt.imshow(img1)
    plt.subplot(212)
    plt.imshow(img2)
    plt.show()


def remove_file(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print('no such file')
    print("successfully removed")


if __name__ == '__main__':
    # crop(train_hr, upscale=4)
    # crop(train_lr, upscale=4)
    crop(test_hr, upscale=4)
    crop(test_lr, upscale=4)
    #
    # lr_downsample(train_lr, upscale=4)
    lr_downsample(test_lr, upscale=4)
    #
    # enlarge_dataset(train_hr)
    # enlarge_dataset(train_lr)

    # remove_file(seg_train_hr)
    # remove_file(seg_train_lr)
    # remove_file(seg_test_hr)
    # remove_file(seg_test_lr)

    segment(is_train=True, upscale=4, lr_size=32)
    segment(is_train=False, upscale=4, lr_size=32)

    get_seg_txt(is_train=True)
    get_seg_txt(is_train=False)

    check_data(is_train=True)
    check_data(is_train=False)
    print("data process end")
