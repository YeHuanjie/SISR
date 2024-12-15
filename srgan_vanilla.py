#%%
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import ganloss
from math import log10
import rdn
import rrdb

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


# class ResBlock(torch.nn.Module):
#     def __init__(self, channels):
#         # Res
#
#         super(ResBlock, self).__init__()
#         self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.prelu = torch.nn.PReLU()
#         self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.prelu(out)
#         out = self.conv2(out)
#         out = out + x
#         return out


class ResBlock(torch.nn.Module):
    def __init__(self, channels):
        # Res + BN

        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.prelu = torch.nn.PReLU()
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        return out


class UpscaleBlock(torch.nn.Module):
    def __init__(self, channels, upscale):
        # Upscale

        super(UpscaleBlock, self).__init__()
        self.conv = torch.nn.Conv2d(channels, channels * upscale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = torch.nn.PixelShuffle(upscale)
        self.prelu = torch.nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self, in_channels=3, upscale=4, num_feature=64, num_layer=16):
        super(Generator, self).__init__()
        # EDSR + tanh

        self.num_layer = num_layer
        # part1: extract features
        self.feature = torch.nn.Conv2d(in_channels, num_feature, kernel_size=9, padding=9 // 2)
        self.prelu = torch.nn.PReLU()

        # part2: res block
        self.resblock = torch.nn.ModuleList([ResBlock(num_feature)])
        for _ in range(self.num_layer - 1):
            self.resblock.append(ResBlock(num_feature))

        # part3: fusion
        self.fusion = torch.nn.Conv2d(num_feature, num_feature, kernel_size=3, padding=3 // 2)
        # 1x1

        self.bn = torch.nn.BatchNorm2d(num_feature)

        # part4: upsample
        self.upsample = UpscaleBlock(num_feature, upscale)

        # part5: out
        self.out = torch.nn.Conv2d(num_feature, in_channels, kernel_size=9, padding=9 // 2)

        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        feature = self.feature(x)
        out = feature
        for i in range(self.num_layer):
            out = self.resblock[i](out)
        out = self.fusion(out)
        out = self.bn(out)
        out = self.upsample(out)
        out = self.out(out)
        out = self.tanh(out)
        return (out + 1) / 2


# class Generator(torch.nn.Module):
#     def __init__(self, upscale, in_channels, num_features, growth_rate, num_blocks, num_layers):
#         # RDN + tanh
#         super(Generator, self).__init__()
#         self.rdn = rdn.RDN(upscale, in_channels, num_features, growth_rate, num_blocks, num_layers)
#         self.tanh = torch.nn.Tanh()
#
#     def forward(self, x):
#         x = self.rdn(x)
#         x = self.tanh(x)
#         return (x + 1) / 2


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=3, num_feature=64):
        super(Discriminator, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, num_feature, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(num_feature, num_feature, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_feature),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(num_feature, 2 * num_feature, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(2 * num_feature),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(2 * num_feature, 2 * num_feature, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(2 * num_feature),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(2 * num_feature, 4 * num_feature, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(4 * num_feature),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(4 * num_feature, 4 * num_feature, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(4 * num_feature),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(4 * num_feature, 8 * num_feature, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(8 * num_feature),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv2d(8 * num_feature, 8 * num_feature, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(8 * num_feature),
            torch.nn.LeakyReLU(0.2),

            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(8 * num_feature, 16 * num_feature, kernel_size=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(16 * num_feature, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class SRGANTrainer(object):
    def __init__(self, flags, train_loader, test_loader):
        super(SRGANTrainer, self).__init__()
        self.G = None
        self.D = None
        self.modelname = flags.modelname
        self.upscale = flags.upscale_factor
        self.lr = flags.lr
        self.epochs = flags.epochs
        self.criterion = None

        self.mse = None
        self.bce = None

        self.Goptimizer = None
        self.Doptimizer = None
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.num_features = flags.srgan_num_features
        self.num_layers = flags.srgan_num_layers

    def bulid_model(self):
        make_dirs(self.modelname)

        self.G = Generator(3, self.upscale, self.num_features, self.num_layers).cuda()
        self.D = Discriminator(3, self.num_features).cuda()

        self.criterion = ganloss.GeneratorLoss().cuda()

        self.mse = torch.nn.MSELoss().cuda()
        self.bce = torch.nn.BCELoss().cuda()

        self.Goptimizer = torch.optim.Adam(params=self.G.parameters(), lr=self.lr)
        self.Doptimizer = torch.optim.Adam(params=self.D.parameters(), lr=self.lr * 0.1)

    def save_model(self, num):
        G_out_path = os.path.join(self.modelname, self.modelname + '_G_params_' + str(num) + '.pkl')
        D_out_path = os.path.join(self.modelname, self.modelname + '_D_params_' + str(num) + '.pkl')
        torch.save(self.G.state_dict(), G_out_path)
        torch.save(self.D.state_dict(), D_out_path)

    def pre_train(self):
        self.G.train()
        for batch_num, (data, target) in enumerate(self.train_loader):
            data, real_img = data.cuda(), target.cuda()
            fake_img = self.G(data)
            self.G.zero_grad()
            loss = self.mse(fake_img, real_img)
            loss.backward()
            self.Goptimizer.step()

    # def train(self):
    #     self.G.train()
    #     self.D.train()
    #     g_loss = 0
    #     d_loss = 0
    #     g_score = 0
    #     d_score = 0
    #     for batch_num, (data, target) in enumerate(self.train_loader):
    #         # step1: Update D network: maximize D(x)-1-D(G(z))
    #         data, real_img = data.cuda(), target.cuda()
    #
    #         fake_img = self.G(data)
    #         self.D.zero_grad()
    #
    #         real_out = self.D(real_img)
    #         fake_out = self.D(fake_img)
    #         # real_out = self.D(real_img).mean()
    #         # fake_out = self.D(fake_img).mean()
    #
    #         real_labels = 0.9 * torch.ones_like(real_out).cuda()
    #         fake_labels = torch.zeros_like(fake_out).cuda()
    #
    #         d_loss = self.bce(real_out, real_labels) + self.bce(fake_out, fake_labels)
    #         fake_out = fake_out.mean()
    #         # d_loss = 1 - real_out + fake_out
    #
    #         d_loss.backward(retain_graph=True)
    #         # 保留梯度
    #         self.Doptimizer.step()
    #
    #         # step2: Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
    #         self.G.zero_grad()
    #         g_loss = self.criterion(fake_out, fake_img, real_img)
    #         g_loss.backward()
    #         self.Goptimizer.step()
    #         fake_img = self.G(data)
    #
    #         fake_out = self.D(fake_img)
    #         # fake_out = self.D(fake_img).mean()
    #
    #         d_loss = self.bce(real_out, real_labels) + self.bce(fake_out, fake_labels)
    #         fake_out = fake_out.mean()
    #         real_out = real_out.mean()
    #         # d_loss = 1 - real_out + fake_out
    #
    #         g_loss = self.criterion(fake_out, fake_img, real_img)
    #
    #         g_loss += g_loss.item()
    #         d_loss += d_loss.item()
    #         d_score += real_out.item()
    #         g_score += fake_out.item()
    #     train_g_loss = g_loss / len(self.train_loader)
    #     train_d_loss = d_loss / len(self.train_loader)
    #     train_d_score = d_score / len(self.train_loader)
    #     train_g_score = g_score / len(self.train_loader)
    #     print("this epoch G_loss: %.6f, D_loss: %.6f, D(x): %.6f, D(G(X)): %.6f"
    #           % (train_g_loss, train_d_loss, train_d_score, train_g_score))
    #
    #     f_path = os.path.join(self.modelname, self.modelname + '_loss.txt')
    #     f = open(f_path, 'a+')
    #     print("this epoch G_loss: %.6f, D_loss: %.6f, D(x): %.6f, D(G(X)): %.6f"
    #           % (train_g_loss, train_d_loss, train_g_score, train_d_score), file=f)
    #     f.close()
    #     return train_g_loss, train_d_loss, train_g_score, train_d_score

    def train(self):
        self.G.train()
        self.D.train()
        g_loss = 0
        d_loss = 0
        g_score = 0
        d_score = 0
        for batch_num, (data, target) in enumerate(self.train_loader):
            # step1: Update D network: maximize D(x)-1-D(G(z))
            data, real_img = data.cuda(), target.cuda()

            fake_img = self.G(data)

            real_out = self.D(real_img)
            fake_out = self.D(fake_img)

            real_labels = 0.9 * torch.ones_like(real_out).cuda()
            fake_labels = torch.zeros_like(fake_out).cuda()

            self.D.zero_grad()
            d_loss = self.bce(real_out, real_labels) + self.bce(fake_out, fake_labels)
            d_loss.backward(retain_graph=True)
            self.Doptimizer.step()
            self.G.zero_grad()
            g_loss = self.criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            self.Goptimizer.step()

            fake_out = fake_out.mean()
            real_out = real_out.mean()

            g_loss += g_loss.item()
            d_loss += d_loss.item()
            d_score += real_out.item()
            g_score += fake_out.item()
        train_g_loss = g_loss / len(self.train_loader)
        train_d_loss = d_loss / len(self.train_loader)
        train_d_score = d_score / len(self.train_loader)
        train_g_score = g_score / len(self.train_loader)
        print("this epoch G_loss: %.6f, D_loss: %.6f, D(x): %.6f, D(G(X)): %.6f"
              % (train_g_loss, train_d_loss, train_d_score, train_g_score))

        f_path = os.path.join(self.modelname, self.modelname + '_loss.txt')
        f = open(f_path, 'a+')
        print("this epoch G_loss: %.6f, D_loss: %.6f, D(x): %.6f, D(G(X)): %.6f"
              % (train_g_loss, train_d_loss, train_g_score, train_d_score), file=f)
        f.close()
        # return train_g_loss, train_d_loss, train_g_score, train_d_score

    def test_seg(self):
        self.G.eval()
        self.D.eval()
        running_loss = 0
        with torch.no_grad():
            for batch_num, (data, label) in enumerate(self.test_loader):
                data, label = data.cuda(), label.cuda()
                output = self.G(data)
                MSE_ = self.mse(output.float(), label.float())
                running_loss += MSE_.item()
            test_loss = running_loss / len(self.test_loader)
            if test_loss < 1e-5:
                test_psnr = 100
            else:
                test_psnr = 10 * log10(1 / test_loss)

            print("this epoch test MSE loss : {}".format(test_loss))
            print("this epoch test psnr : {}".format(test_psnr))

            f_path = os.path.join(self.modelname, self.modelname + '_loss.txt')
            f = open(f_path, 'a+')
            print("this epoch test MSE loss : {}".format(test_loss), file=f)
            print("this epoch test psnr : {}".format(test_psnr), file=f)
            f.close()

    def test(self, num):
        self.G.eval()
        self.D.eval()
        save_sr = os.path.join(self.modelname, self.modelname)
        make_dirs(save_sr)
        # ./srgan/srgan
        with torch.no_grad():
            for file in os.listdir(test_lr):
                read_path = os.path.join(test_lr, file)
                img = Image.open(read_path)

                inputs = transforms.ToTensor()(img).view(1, -1, img.size[1], img.size[0])
                para_savepath = os.path.join(self.modelname, self.modelname + '_G_params_' + str(num) + '.pkl')
                self.G.load_state_dict(torch.load(para_savepath))
                inputs = inputs.cuda()

                output = self.G(inputs)
                output_img = output.cpu().clone()
                output_img = output_img.squeeze(0)
                output_img[output_img < 0] = 0
                output_img[output_img > 1] = 1
                output_img = transforms.ToPILImage()(output_img)

                sr_name = os.path.join(save_sr, self.modelname + '_' + file)
                output_img.save(sr_name)

    def run(self, mode='train'):
        if mode == 'train':
            self.bulid_model()

            for init_epoch in range(1, 5):
                self.pre_train()
                self.test_seg()

            for epoch in range(1, self.epochs + 1):
                print("\n ==> Epoch {}".format(epoch))

                f_path = os.path.join(self.modelname, self.modelname + '_loss.txt')
                f = open(f_path, 'a+')
                print(" ==> Epoch {}".format(epoch), file=f)
                f.close()

                self.train()
                self.test_seg()
                self.save_model(num=epoch)

        if mode == 'test':
            self.bulid_model()
            self.test(num=self.epochs)

        if mode == 'train and test':
            self.bulid_model()

            for init_epoch in range(1, 5):
                self.pre_train()
                self.test_seg()

            for epoch in range(1, self.epochs + 1):
                print("\n ==> Epoch {}".format(epoch))

                f_path = os.path.join(self.modelname, self.modelname + '_loss.txt')
                f = open(f_path, 'a+')
                print(" ==> Epoch {}".format(epoch), file=f)
                f.close()

                self.train()
                self.test_seg()
                self.save_model(num=epoch)
                self.test(num=epoch)
