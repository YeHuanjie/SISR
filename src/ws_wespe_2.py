# coding=utf-8
from torchvision.models.vgg import vgg16
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import srgan_rdn
import ganloss
from math import log10


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
fake_fake_y = os.path.join(seg_train, 'fake_fake_y')


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = torch.cat([x, out], 1)
        return out


class RDB(torch.nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        # growth rate: number of feature map
        # in 3 out 64
        # in 3+64 out 64
        # in 3+64+64 out 64
        # ......
        self.layers = torch.nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate)
                                            for i in range(num_layers)])

        # local feature fusion
        self.lff = torch.nn.Conv2d(in_channels + growth_rate * num_layers, in_channels, kernel_size=1)
        # 1x1 conv

    def forward(self, x):
        out = self.layers(x)
        out = self.lff(out)
        out = x + out
        return out
        # local residual learning


class UPRDN(torch.nn.Module):
    # four parts
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(UPRDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # part1: shallow feature extraction
        self.sfe1 = torch.nn.Conv2d(num_channels, self.G0, kernel_size=3, padding=3 // 2)
        self.sfe2 = torch.nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)

        # part2: residual dense blocks
        self.rdbs = torch.nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G0, self.G, self.C))

        # part3: dense feature fusion
        self.gff = torch.nn.Sequential(
            torch.nn.Conv2d(self.G0 * self.D, self.G0, kernel_size=1),
            # 1x1 conv
            torch.nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # part4: up_sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([torch.nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     torch.nn.PixelShuffle(2)])
            self.upscale = torch.nn.Sequential(*self.upscale)
        else:
            self.upscale = torch.nn.Sequential(
                torch.nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                torch.nn.PixelShuffle(scale_factor)
            )

        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2),
            torch.nn.Tanh())

    def forward(self, x):
        # part1: shallow feature extraction
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        # part2: residual dense blocks
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        # part3: dense feature fusion
        x = self.gff(torch.cat(local_features, 1)) + sfe1

        # part4: up_sampling
        x = self.upscale(x)

        x = self.output(x)
        return x


class DOWNRDN(torch.nn.Module):
    # four parts
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(DOWNRDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # part1: shallow feature extraction
        self.sfe1 = torch.nn.Conv2d(num_channels, self.G0, kernel_size=3, padding=3 // 2)
        self.sfe2 = torch.nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)

        # part2: residual dense blocks
        self.rdbs = torch.nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G0, self.G, self.C))

        # part3: dense feature fusion
        self.gff = torch.nn.Sequential(
            torch.nn.Conv2d(self.G0 * self.D, self.G0, kernel_size=1),
            # 1x1 conv
            torch.nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # part4: down_sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([torch.nn.Conv2d(self.G0, self.G0, kernel_size=4, stride=2, padding=1)])
            self.upscale = torch.nn.Sequential(*self.upscale)
        else:
            self.upscale = torch.nn.Sequential(
                torch.nn.Conv2d(self.G0, self.G0, kernel_size=4, stride=2, padding=1),
            )

        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2),
            torch.nn.Tanh())

    def forward(self, x):
        # part1: shallow feature extraction
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        # part2: residual dense blocks
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        # part3: dense feature fusion
        x = self.gff(torch.cat(local_features, 1)) + sfe1

        # part4: up_sampling
        x = self.upscale(x)

        x = self.output(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            torch.nn.InstanceNorm2d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        out = self.conv_layers(x)
        return out


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class TVLoss(torch.nn.Module):
    def __init__(self, tv_weight=10):
        super(TVLoss, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class WESPE2Trainer(object):
    def __init__(self, flags, train_loader, test_loader):
        super(WESPE2Trainer, self).__init__()
        self.modelname = flags.modelname
        self.upscale = flags.upscale_factor
        self.lr = flags.lr
        self.epochs = flags.epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.Gg = None
        self.Gf = None
        self.D2 = None

        self.mse_criterion = None
        self.l1_criterion = None
        self.bce_criterion = None
        self.tv = None

        self.G_optimizer = None
        self.D_optimizer = None

        # self.srgan_g_save = flags.srgan_g_save

        self.num_features = flags.num_features
        self.growth_rate = flags.growth_rate
        self.num_blocks = flags.num_blocks
        self.num_layers = flags.num_layers

    def bulid_model(self):
        make_dirs(self.modelname)
        self.Gg = UPRDN(self.upscale, 3, self.num_features,
                        self.growth_rate, self.num_blocks, self.num_layers).cuda()
        self.Gf = DOWNRDN(self.upscale, 3, self.num_features,
                          self.growth_rate, self.num_blocks, self.num_layers).cuda()
        self.Gg.apply(weight_init)
        self.Gf.apply(weight_init)
        self.D2 = Discriminator(in_channels=3).cuda()
        self.D2.apply(weight_init)

        self.mse_criterion = torch.nn.MSELoss()
        self.l1_criterion = torch.nn.L1Loss()
        self.bce_criterion = torch.nn.BCELoss()
        self.tv = TVLoss()

        g_params = list(self.Gg.parameters()) + list(self.Gf.parameters())
        d_params = list(self.D2.parameters())

        self.G_optimizer = torch.optim.Adam(lr=self.lr, params=g_params, betas=(0.9, 0.999))
        self.D_optimizer = torch.optim.Adam(lr=self.lr, params=d_params, betas=(0.9, 0.999))

    def adjust_learning_rate(self, epoch):
        for param_group in self.G_optimizer.param_groups:
            param_group['lr'] = self.lr * (0.5 ** (epoch // 50))
        for param_group in self.D_optimizer.param_groups:
            param_group['lr'] = self.lr * (0.5 ** (epoch // 50))

    def save_model(self, num):
        path1 = os.path.join(self.modelname, self.modelname + '2_G_g_' + str(num) + '.pkl')
        path2 = os.path.join(self.modelname, self.modelname + '2_G_f_' + str(num) + '.pkl')
        path3 = os.path.join(self.modelname, self.modelname + '2_D_2_' + str(num) + '.pkl')

        torch.save(self.Gg.state_dict(), path1)
        torch.save(self.Gf.state_dict(), path2)
        torch.save(self.D2.state_dict(), path3)

    def train(self, device):
        gen_ad = 0
        gen_cyc = 0
        gen_idt = 0
        gen_tv = 0

        gen_loss = 0
        dis_loss = 0
        for batch_num, (x, y, y_fake, z) in enumerate(self.train_loader):
            x, y, y_fake, z = x.cuda(), y.cuda(), y_fake.cuda(), z.cuda()

            # generate
            z_fake = self.Gg(y_fake)
            y_fake_fake = self.Gf(z_fake)

            # ------- train discriminator -------- #
            self.D_optimizer.zero_grad()
            real_out = self.D2(z)
            fake_out = self.D2(z_fake)
            real_labels = torch.ones_like(real_out).cuda()
            fake_labels = torch.zeros_like(fake_out).cuda()

            dis_loss = self.mse_criterion(real_out, real_labels) + self.mse_criterion(fake_out, fake_labels)

            # train D
            dis_loss.backward(retain_graph=True)
            self.D_optimizer.step()

            # ------- train generator ------- #
            self.G_optimizer.zero_grad()

            # gan loss
            real_out = self.D2(z)
            fake_out = self.D2(z_fake)

            gen_ad = self.mse_criterion(fake_out, real_labels)
            gen_ad.backward(retain_graph=True)

            # tv loss
            # gen_tv = 0.5 * self.tv(z_fake)
            # gen_tv.backward(retain_graph=True)
            gen_tv = 0

            # idt loss
            z_idt = self.Gg(y)
            gen_idt = 20 * self.l1_criterion(z_idt, z)
            gen_idt.backward(retain_graph=True)
            gen_idt = 0

            # cyc loss
            gen_cyc = 20 * self.l1_criterion(y_fake_fake, x)
            gen_cyc.backward(retain_graph=True)

            # train G
            gen_loss = gen_ad + gen_cyc + gen_idt + gen_tv
            self.G_optimizer.step()

            gen_ad += 1 * gen_ad
            gen_cyc += 1 * gen_cyc
            gen_idt += 1 * gen_idt
            gen_tv += 1 * gen_tv

            gen_loss += gen_loss.item()
            dis_loss += dis_loss.item()

        train_g_ad = gen_ad
        train_g_cyc = gen_cyc
        train_g_idt = gen_idt
        train_g_tv = gen_tv

        train_g_loss = gen_loss
        train_d_loss = dis_loss
        print("this epoch ad_loss: %.4f, cyc_loss: %.4f, idt_loss: %.4f, tv_loss: %.4f, "
              % (train_g_ad, train_g_cyc, train_g_idt, train_g_tv))
        print("this epoch G_loss: %.4f, D_loss: %.4f" % (train_g_loss, train_d_loss))

        f_path = os.path.join(self.modelname, self.modelname + '_loss_2.txt')
        f = open(f_path, 'a+')
        print("this epoch ad_loss: %.4f, cyc_loss: %.4f, idt_loss: %.4f, tv_loss: %.4f, "
              % (train_g_ad, train_g_cyc, train_g_idt, train_g_tv), file=f)
        print("this epoch G_loss: %.4f, D_loss: %.4f" % (train_g_loss, train_d_loss), file=f)
        f.close()

    def test_seg(self, num):
        make_dirs(fake_fake_y)
        with torch.no_grad():
            for file in os.listdir(fake_y):
                read_path = os.path.join(fake_y, file)
                img = Image.open(read_path)

                inputs = transforms.ToTensor()(img).view(1, -1, img.size[1], img.size[0])
                path = os.path.join(self.modelname, self.modelname + '2_G_g_' + str(num) + '.pkl')
                self.Gg.load_state_dict(torch.load(path))
                inputs = inputs.cuda()

                output = self.Gg(inputs)
                output_img = output.cpu().clone()
                output_img = output_img.squeeze(0)
                output_img[output_img < 0] = 0
                output_img[output_img > 1] = 1
                output_img = transforms.ToPILImage()(output_img)

                fake_file_name = os.path.join(fake_fake_y, 'fake_' + file)
                output_img.save(fake_file_name)
            print("finish")

    def test(self, num):
        make_dirs(seg_test_step2)
        with torch.no_grad():
            for file in os.listdir(seg_test_step1):
                read_path = os.path.join(seg_test_step1, file)
                img = Image.open(read_path)

                inputs = transforms.ToTensor()(img).view(1, -1, img.size[1], img.size[0])
                path = os.path.join(self.modelname, self.modelname + '2_G_g_' + str(num) + '.pkl')
                self.Gg.load_state_dict(torch.load(path))
                inputs = inputs.cuda()

                output = self.Gg(inputs)
                output_img = output.cpu().clone()
                output_img = output_img.squeeze(0)
                output_img[output_img < 0] = 0
                output_img[output_img > 1] = 1
                output_img = transforms.ToPILImage()(output_img)

                fake_file_name = os.path.join(seg_test_step2, 'fake_' + file)
                output_img.save(fake_file_name)
            print("finish")

    def run(self):
        device = torch.device('cuda')
        self.bulid_model()

        for epoch in range(1, self.epochs + 1):
            self.adjust_learning_rate(epoch)
            print("\n ==> Epoch {}".format(epoch))

            f_path = os.path.join(self.modelname, self.modelname + '_loss_2.txt')
            f = open(f_path, 'a+')
            print(" ==> Epoch {}".format(epoch), file=f)
            f.close()

            self.train(device=device)
            self.save_model(num=epoch)

        self.test_seg(num=self.epochs)
        self.test(num=self.epochs)
