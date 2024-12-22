# coding=utf-8
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import functools

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


class UnetBlock(torch.nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=torch.nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == torch.nn.InstanceNorm2d
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = torch.nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                   stride=2, padding=1, bias=use_bias)
        downrelu = torch.nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = torch.nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            # upconv = torch.nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                                   kernel_size=4, stride=2,
            #                                   padding=1)
            #
            # upconv = torch.nn.Sequential(
            #     torch.nn.Upsample(scale_factor=2),
            #     torch.nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
            # )
    
            upconv = torch.nn.Sequential(
                torch.nn.Conv2d(inner_nc * 2, outer_nc * 2 * 2, kernel_size=3, stride=1, padding=1),
                torch.nn.PixelShuffle(2),
            )
        
            down = [downconv]
            up = [uprelu, upconv, torch.nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            # upconv = torch.nn.ConvTranspose2d(inner_nc, outer_nc,
            #                                   kernel_size=4, stride=2,
            #                                   padding=1, bias=use_bias)
            #
            # upconv = torch.nn.Sequential(
            #     torch.nn.Upsample(scale_factor=2),
            #     torch.nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1)
            # )

            upconv = torch.nn.Sequential(
                torch.nn.Conv2d(inner_nc, outer_nc * 2 * 2, kernel_size=3, stride=1, padding=1),
                torch.nn.PixelShuffle(2),
            )
    
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            # upconv = torch.nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                                   kernel_size=4, stride=2,
            #                                   padding=1, bias=use_bias)
            #
            # upconv = torch.nn.Sequential(
            #     torch.nn.Upsample(scale_factor=2),
            #     torch.nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
            # )

            upconv = torch.nn.Sequential(
                torch.nn.Conv2d(inner_nc * 2, outer_nc * 2 * 2, kernel_size=3, stride=1, padding=1),
                torch.nn.PixelShuffle(2),
            )
    
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [torch.nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(channels, channels, 3, bias=True),
            torch.nn.InstanceNorm2d(channels),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(channels, channels, 3, bias=True),
            torch.nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.layers(x)


class ResGenerator(torch.nn.Module):
    def __init__(self, repeat_num=9):
        super(ResGenerator, self).__init__()

        layers = list()
        layers.append(torch.nn.ReflectionPad2d(3))
        layers.append(torch.nn.Conv2d(3, 64, 7))
        layers.append(torch.nn.InstanceNorm2d(64))
        layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.Conv2d(64, 128, 3, stride=2, padding=1))
        layers.append(torch.nn.InstanceNorm2d(128))
        layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.Conv2d(128, 256, 3, stride=2, padding=1))
        layers.append(torch.nn.InstanceNorm2d(256))
        layers.append(torch.nn.ReLU(inplace=True))

        for _ in range(repeat_num):
            layers.append(ResidualBlock(256))

        # layers.append(torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1))
        # layers.append(torch.nn.InstanceNorm2d(128))
        # layers.append(torch.nn.ReLU(inplace=True))
        #
        # layers.append(torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1))
        # layers.append(torch.nn.InstanceNorm2d(64))
        # layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.Conv2d(256, 128*2*2, 3, stride=1, padding=1))
        layers.append(torch.nn.PixelShuffle(2))
        layers.append(torch.nn.InstanceNorm2d(128))
        layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.Conv2d(128, 64*2*2, 3, stride=1, padding=1))
        layers.append(torch.nn.PixelShuffle(2))
        layers.append(torch.nn.InstanceNorm2d(64))
        layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.ReflectionPad2d(3))
        layers.append(torch.nn.Conv2d(64, 3, 7))
        layers.append(torch.nn.Tanh())

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class UnetGenerator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=torch.nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        # add the innermost layer
        for i in range(num_downs - 5):
            # add intermediate layers with ngf * 8 filters
            unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None,
                                   submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetBlock(ngf * 4, ngf * 8, input_nc=None,
                               submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock(ngf * 2, ngf * 4, input_nc=None,
                               submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock(ngf, ngf * 2, input_nc=None,
                               submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetBlock(output_nc, ngf, input_nc=input_nc,
                               submodule=unet_block, outermost=True, norm_layer=norm_layer)
        # add the outermost layer

    def forward(self, x):
        return self.model(x)


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
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.tv_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class WESPETrainer(object):
    def __init__(self, flags, train_loader, test_loader):
        super(WESPETrainer, self).__init__()
        self.modelname = flags.modelname
        self.upscale = flags.upscale_factor
        self.lr = flags.lr
        self.epochs = flags.epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.Gg = None
        self.Gf = None
        self.D1 = None

        self.mse_criterion = None
        self.l1_criterion = None
        self.bce_criterion = None
        self.tv = None

        self.G_optimizer = None
        self.D_optimizer = None

    def bulid_model(self):
        make_dirs(self.modelname)
        # self.Gg = UnetGenerator(3, 3, 7, 64, use_dropout=False).cuda()
        # self.Gf = UnetGenerator(3, 3, 7, 64, use_dropout=False).cuda()
        self.Gg = ResGenerator(9).cuda()
        self.Gf = ResGenerator(9).cuda()
        self.Gg.apply(weight_init)
        self.Gf.apply(weight_init)

        self.D1 = Discriminator(in_channels=3).cuda()
        self.D1.apply(weight_init)

        self.mse_criterion = torch.nn.MSELoss()
        self.l1_criterion = torch.nn.L1Loss()
        self.bce_criterion = torch.nn.BCELoss()
        self.tv = TVLoss()

        g_params = list(self.Gg.parameters()) + list(self.Gf.parameters())
        d_params = list(self.D1.parameters())

#         self.G_optimizer = torch.optim.Adam(lr=self.lr, params=g_params, betas=(0.5, 0.999))
#         self.D_optimizer = torch.optim.Adam(lr=self.lr, params=d_params, betas=(0.5, 0.999))
        self.G_optimizer = torch.optim.Adam(lr=self.lr, params=g_params, betas=(0.9, 0.999))
        self.D_optimizer = torch.optim.Adam(lr=self.lr, params=d_params, betas=(0.9, 0.999))

    def adjust_learning_rate(self, epoch):
        for param_group in self.G_optimizer.param_groups:
            param_group['lr'] = self.lr * (0.5 ** (epoch // 50))
        for param_group in self.D_optimizer.param_groups:
            param_group['lr'] = self.lr * (0.5 ** (epoch // 50))

    def save_model(self, num):
        path1 = os.path.join(self.modelname, self.modelname + '1_G_g_' + str(num) + '.pkl')
        path2 = os.path.join(self.modelname, self.modelname + '1_G_f_' + str(num) + '.pkl')
        path3 = os.path.join(self.modelname, self.modelname + '1_D_1_' + str(num) + '.pkl')

        torch.save(self.Gg.state_dict(), path1)
        torch.save(self.Gf.state_dict(), path2)
        torch.save(self.D1.state_dict(), path3)

    # def train(self, device):
    #     gen_ad = 0
    #     gen_cyc = 0
    #     gen_idt = 0
    #     gen_tv = 0
    #
    #     gen_loss = 0
    #     dis_loss = 0
    #     for batch_num, (x, y) in enumerate(self.train_loader):
    #         x, y = x.cuda(), y.cuda()
    #         batch_size = x.size()[0]
    #
    #         # generate
    #         y_fake = self.Gg(x)
    #         x_fake = self.Gf(y_fake)
    #
    #         f_y = self.Gf(y)
    #         gf_y = self.Gg(f_y)
    #
    #         # ------- train generator ------- #
    #         self.G_optimizer.zero_grad()
    #
    #         # gan loss
    #         real_out = self.D1(y)
    #         fake_out = self.D1(y_fake)
    #         real_labels = torch.ones_like(real_out).cuda()
    #         fake_labels = torch.zeros_like(fake_out).cuda()
    #
    #         gen_ad = 0.5 * self.mse_criterion(fake_out, real_labels)
    #
    #         # cyc loss
    #         gen_cyc = self.l1_criterion(x_fake, x) + self.l1_criterion(gf_y, y)
    #
    #         # idt loss
    #         y_idt = self.Gg(y)
    #         x_idt = self.Gf(x)
    #         gen_idt = self.l1_criterion(y_idt, y) + self.l1_criterion(x_idt, x)
    #
    #         # tv loss
    #         gen_tv = self.tv(y_fake)
    #
    #         # total loss
    #         gen_loss = 1 * gen_ad + 10 * gen_cyc + 5 * gen_idt + 0.5 * gen_tv
    #
    #         # train G
    #         gen_loss.backward(retain_graph=True)
    #         self.G_optimizer.step()
    #
    #         # ------- train discriminator -------- #
    #         self.D_optimizer.zero_grad()
    #         real_out = self.D1(y)
    #         fake_out = self.D1(y_fake)
    #
    #         dis_loss = 0.5 * self.mse_criterion(real_out, real_labels) \
    #                    + 0.5 * self.mse_criterion(fake_out, fake_labels)
    #
    #         # train D
    #         dis_loss.backward()
    #         self.D_optimizer.step()
    #
    #         gen_ad += 1 * gen_ad
    #         gen_cyc += 10 * gen_cyc
    #         gen_idt += 5 * gen_idt
    #         gen_tv += 0.5 * gen_tv
    #
    #         gen_loss += gen_loss.item()
    #         dis_loss += dis_loss.item()
    #
    #     train_g_ad = gen_ad
    #     train_g_cyc = gen_cyc
    #     train_g_idt = gen_idt
    #     train_g_tv = gen_tv
    #
    #     train_g_loss = gen_loss
    #     train_d_loss = dis_loss
    #     print("this epoch ad_loss: %.4f, cyc_loss: %.4f, idt_loss: %.4f, tv_loss: %.4f, "
    #           % (train_g_ad, train_g_cyc, train_g_idt, train_g_tv))
    #     print("this epoch G_loss: %.4f, D_loss: %.4f" % (train_g_loss, train_d_loss))
    #
    #     f_path = os.path.join(self.modelname, self.modelname + '_loss_1.txt')
    #     f = open(f_path, 'a+')
    #     print("this epoch ad_loss: %.4f, cyc_loss: %.4f, idt_loss: %.4f, tv_loss: %.4f, "
    #           % (train_g_ad, train_g_cyc, train_g_idt, train_g_tv), file=f)
    #     print("this epoch G_loss: %.4f, D_loss: %.4f" % (train_g_loss, train_d_loss), file=f)
    #     f.close()

    def train(self, device):
        gen_ad = 0
        gen_cyc = 0
        gen_idt = 0
        gen_tv = 0

        gen_loss = 0
        dis_loss = 0
        for batch_num, (x, y) in enumerate(self.train_loader):
            x, y = x.cuda(), y.cuda()

            # generate
            y_fake = self.Gg(x)
            x_fake = self.Gf(y_fake)

            f_y = self.Gf(y)
            gf_y = self.Gg(f_y)

            # ------- train discriminator -------- #
            self.D_optimizer.zero_grad()
            real_out = self.D1(y)
            fake_out = self.D1(y_fake)
            real_labels = torch.ones_like(real_out).cuda()
            fake_labels = torch.zeros_like(fake_out).cuda()

            dis_loss = self.mse_criterion(real_out, real_labels) + self.mse_criterion(fake_out, fake_labels)

            # train D
            dis_loss.backward(retain_graph=True)
            self.D_optimizer.step()

            # ------- train generator ------- #
            self.G_optimizer.zero_grad()

            # gan loss
            real_out = self.D1(y)
            fake_out = self.D1(y_fake)

            gen_ad = self.mse_criterion(fake_out, real_labels)
            gen_ad.backward(retain_graph=True)

            # tv loss
            # gen_tv = 0.1 * self.tv(y_fake)
            # gen_tv.backward(retain_graph=True)
            gen_tv = 0

            # idt loss
            y_idt = self.Gg(y)
            x_idt = self.Gf(x)
            gen_idt = 50 * self.l1_criterion(y_idt, y) + 50 * self.l1_criterion(x_idt, x)
            gen_idt.backward(retain_graph=True)

            # cyc loss
            gen_cyc = 50 * self.l1_criterion(x_fake, x) + 50 * self.l1_criterion(gf_y, y)
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

        f_path = os.path.join(self.modelname, self.modelname + '_loss_1.txt')
        f = open(f_path, 'a+')
        print("this epoch ad_loss: %.4f, cyc_loss: %.4f, idt_loss: %.4f, tv_loss: %.4f, "
              % (train_g_ad, train_g_cyc, train_g_idt, train_g_tv), file=f)
        print("this epoch G_loss: %.4f, D_loss: %.4f" % (train_g_loss, train_d_loss), file=f)
        f.close()
        
    def test_x(self, num):
        make_dirs(fake_y)
        with torch.no_grad():
            for file in os.listdir(seg_train_x):
                if ".png" in file:
                    read_path = os.path.join(seg_train_x, file)
                    img = Image.open(read_path)

                    inputs = transforms.ToTensor()(img).view(1, -1, img.size[1], img.size[0])
                    path = os.path.join(self.modelname, self.modelname + '1_G_g_' + str(num) + '.pkl')
                    self.Gg.load_state_dict(torch.load(path))
                    inputs = inputs.cuda()

                    output = self.Gg(inputs)
                    output_img = output.cpu().clone()
                    output_img = output_img.squeeze(0)
                    output_img[output_img < 0] = 0
                    output_img[output_img > 1] = 1
                    output_img = transforms.ToPILImage()(output_img)

                    fake_file_name = os.path.join(fake_y, 'fake_' + file)
                    output_img.save(fake_file_name)
            print("finish")

    def test_ori(self, num):
        make_dirs(seg_test_step1)
        with torch.no_grad():
            for file in os.listdir(seg_test_ori):
                if ".png" in file:
                    read_path = os.path.join(seg_test_ori, file)
                    img = Image.open(read_path)

                    inputs = transforms.ToTensor()(img).view(1, -1, img.size[1], img.size[0])
                    path = os.path.join(self.modelname, self.modelname + '1_G_g_' + str(num) + '.pkl')
                    self.Gg.load_state_dict(torch.load(path))
                    inputs = inputs.cuda()

                    output = self.Gg(inputs)
                    output_img = output.cpu().clone()
                    output_img = output_img.squeeze(0)
                    output_img[output_img < 0] = 0
                    output_img[output_img > 1] = 1
                    output_img = transforms.ToPILImage()(output_img)

                    fake_file_name = os.path.join(seg_test_step1, 'fake_' + file)
                    output_img.save(fake_file_name)
            print("finish")

    def run(self):
        device = torch.device('cuda')
        self.bulid_model()

        for epoch in range(1, self.epochs + 1):
            self.adjust_learning_rate(epoch)
            print("\n ==> Epoch {}".format(epoch))

            f_path = os.path.join(self.modelname, self.modelname + '_loss_1.txt')
            f = open(f_path, 'a+')
            print(" ==> Epoch {}".format(epoch), file=f)
            f.close()

            self.train(device=device)
            self.save_model(num=epoch)

            self.test_ori(num=epoch)
        
        self.test_x(num=self.epochs)
#         self.test_ori(num=17)

