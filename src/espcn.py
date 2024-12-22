#%%
import os
import torch
from math import log10
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
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


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


class ESPCN(torch.nn.Module):
    def __init__(self, upscale=4):
        super(ESPCN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=9//2),
            torch.nn.PReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=5//2),
            torch.nn.PReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=1 * upscale * upscale, kernel_size=5, stride=1, padding=5//2),
        )
        self.pixel_shuffle = torch.nn.PixelShuffle(upscale)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, mean=0, std=0.001)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ESPCNTrainer(object):
    def __init__(self, flags, train_loader, test_loader):
        super(ESPCNTrainer, self).__init__()
        self.model = None
        self.modelname = flags.modelname
        self.upscale = flags.upscale_factor
        self.lr = flags.lr
        self.epochs = flags.epochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = train_loader
        self.test_loader = test_loader

    def bulid_model(self):
        make_dirs(self.modelname)
        self.model = ESPCN(upscale=self.upscale).cuda()
        self.model.apply(weight_init)
        self.criterion = torch.nn.MSELoss().cuda()

        params = []
        conv1_param = dict(self.model.conv1.named_parameters())
        for key, value in conv1_param.items():
            if 'bias' not in key:
                params += [{'params': [value], 'lr': self.lr}]
            else:
                params += [{'params': [value], 'lr': self.lr * 0.1}]

        conv2_param = dict(self.model.conv2.named_parameters())
        for key, value in conv2_param.items():
            if 'bias' not in key:
                params += [{'params': [value], 'lr': self.lr}]
            else:
                params += [{'params': [value], 'lr': self.lr * 0.1}]

        conv3_param = dict(self.model.conv3.named_parameters())
        for key, value in conv3_param.items():
            if 'bias' not in key:
                params += [{'params': [value], 'lr': self.lr}]
            else:
                params += [{'params': [value], 'lr': self.lr * 0.1}]

        self.optimizer = torch.optim.Adam(params=params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[80, 100], gamma=0.1)

    def save_model(self, num):
        model_out_path = os.path.join(self.modelname, self.modelname + '_params_' + str(num) + '.pkl')
        torch.save(self.model.state_dict(), model_out_path)

    def train(self):
        self.model.train()
        running_loss = 0
        for batch_num, (data, label) in enumerate(self.train_loader):
            data, label = data.cuda(), label.cuda()
            output = self.model(data)
            self.optimizer.zero_grad()
            loss = self.criterion(output.float(), label.float())
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        train_loss = running_loss / len(self.train_loader)
        print("this epoch train loss : {} ".format(train_loss))

        f_path = os.path.join(self.modelname, self.modelname + '_loss.txt')
        f = open(f_path, 'a+')
        print("this epoch train loss : {} ".format(train_loss), file=f)
        f.close()
        return train_loss

    def test_seg(self):
        self.model.eval()
        running_loss = 0
        with torch.no_grad():
            for batch_num, (data, label) in enumerate(self.test_loader):
                data, label = data.cuda(), label.cuda()
                output = self.model(data)
                MSE_ = self.criterion(output.float(), label.float())
                running_loss += MSE_.item()
            test_loss = running_loss / len(self.test_loader)
            if test_loss < 1e-5:
                test_psnr = 100
            else:
                test_psnr = 10 * log10(1 / test_loss)
            print("this epoch test loss : {}".format(test_loss))
            print("***Average PSNR: {} dB".format(test_psnr))

            f_path = os.path.join(self.modelname, self.modelname + '_loss.txt')
            f = open(f_path, 'a+')
            print("this epoch test loss : {}".format(test_loss), file=f)
            print("***Average PSNR: {} dB".format(test_psnr), file=f)
            f.close()

    def test(self):
        self.model = torch.nn.DataParallel(self.model)
        save_sr = os.path.join(self.modelname, self.modelname)
        make_dirs(save_sr)
        # ./espcn/espcn
        with torch.no_grad():
            for file in os.listdir(test_lr):
                read_path = os.path.join(test_lr, file)
                img = Image.open(read_path)
                img = img.convert("YCbCr")
                y, _, _ = img.split()

                inputs = transforms.ToTensor()(y).view(1, -1, img.size[1], img.size[0])
                para_savepath = os.path.join(self.modelname, self.modelname + '_params_' + str(self.epochs) + '.pkl')
                self.model.load_state_dict(torch.load(para_savepath))
                inputs = inputs.cuda()

                output = self.model(inputs)
                output_img = output.cpu().clone()
                output_img = output_img.squeeze(0)
                output_img[output_img < 0] = 0
                output_img[output_img > 1] = 1
                output_img = transforms.ToPILImage()(output_img)
                output_h, output_w = output_img.size
                output_img = np.array(output_img)

                img = img.convert("RGB")
                h, w = img.size
                img = img.resize((h * self.upscale, w * self.upscale), Image.ANTIALIAS)
                img = img.convert("YCbCr")
                _, cb, cr = img.split()

                cb = np.array(cb)
                cr = np.array(cr)

                merge_img = np.zeros([output_w, output_h, 3])
                merge_img[:, :, 0] = output_img
                merge_img[:, :, 1] = cr
                merge_img[:, :, 2] = cb
                # cv2 : Y Cr Cb
                merge_img = merge_img.astype('uint8')
                merge_img = cv2.cvtColor(merge_img, cv2.COLOR_YCR_CB2BGR)

                sr_name = os.path.join(save_sr, self.modelname + '_' + file)
                cv2.imwrite(sr_name, merge_img)

    def run(self, mode='train'):
        all_loss = []
        if mode == 'train':
            self.bulid_model()
            for epoch in range(1, self.epochs + 1):
                print("\n ==> Epoch {}".format(epoch))

                f_path = os.path.join(self.modelname, self.modelname + '_loss.txt')
                f = open(f_path, 'a+')
                print(" ==> Epoch {}".format(epoch), file=f)
                f.close()

                train_loss = self.train()
                all_loss.append(train_loss)
                # seg img test
                self.test_seg()
                self.scheduler.step(epoch)
                if epoch % 10 == 0:
                    self.save_model(num=epoch)
                else:
                    pass
                if epoch == 90 or epoch == 95 or epoch >= 100:
                    plt.plot(all_loss)
                    plt_path = os.path.join(self.modelname, self.modelname + '_loss_' + str(epoch) + '.png')
                    plt.savefig(plt_path)
                else:
                    pass
        if mode == 'test':
            self.bulid_model()
            # full img test
            self.test()
        if mode == 'train and test':
            self.bulid_model()
            for epoch in range(1, self.epochs + 1):
                print("\n ==> Epoch {}".format(epoch))

                f_path = os.path.join(self.modelname, self.modelname + '_loss.txt')
                f = open(f_path, 'a+')
                print(" ==> Epoch {}".format(epoch), file=f)
                f.close()

                train_loss = self.train()
                all_loss.append(train_loss)
                # seg img test
                self.test_seg()
                self.scheduler.step(epoch)
                if epoch % 10 == 0:
                    self.save_model(num=epoch)
                else:
                    pass
                if epoch == 90 or epoch == 95 or epoch >= 100:
                    plt.plot(all_loss)
                    plt_path = os.path.join(self.modelname, self.modelname + '_loss_' + str(epoch) + '.png')
                    plt.savefig(plt_path)
                else:
                    pass
            self.bulid_model()
            # full img test
            self.test()
