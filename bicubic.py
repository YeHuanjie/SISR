# coding=utf-8
import os
from PIL import Image
import shutil
import cv2
import IQA

upscale_factor = 4

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


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


save_root = 'bicubic'
save_hr = os.path.join(save_root, 'origin')
make_dirs(save_hr)
# bicubic algorithm: no need to train, only need to test
# origin image
for file in os.listdir(test_hr):
    read_path = os.path.join(test_hr, file)
    save_path = os.path.join(save_hr, file)
    shutil.copyfile(read_path, save_path)


save_lr = os.path.join(save_root, 'bicubic')
save_input = os.path.join(save_root, 'input')
make_dirs(save_lr)
make_dirs(save_input)
for file in os.listdir(test_lr):
    read_path = os.path.join(test_lr, file)
    img = Image.open(read_path)
    save_input_img = os.path.join(save_input, 'input_' + file)
    img.save(save_input_img)
    # input image
    h, w = img.size
    img1 = img.resize((h * upscale_factor, w * upscale_factor), Image.ANTIALIAS)
    save_bicubic_img = os.path.join(save_lr, 'bicubic_' + file)
    img1.save(save_bicubic_img)
    # bicubic image
