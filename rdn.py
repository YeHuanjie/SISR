#%%
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from math import log10

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


class RDN(torch.nn.Module):
    # four parts
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN, self).__init__()
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

        self.output = torch.nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

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


class RDNTrainer(object):
    def __init__(self, flags, train_loader, test_loader):
        super(RDNTrainer, self).__init__()
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
        self.num_features = flags.rdn_num_features
        self.growth_rate = flags.rdn_growth_rate
        self.num_blocks = flags.rdn_num_blocks
        self.num_layers = flags.rdn_num_layers

    def bulid_model(self):
        make_dirs(self.modelname)
        # self.model = torch.nn.DataParallel(RDN(self.upscale, 3, self.num_features,
        #                                        self.growth_rate, self.num_blocks, self.num_layers)).cuda()
        self.model = RDN(self.upscale, 3, self.num_features, self.growth_rate, self.num_blocks, self.num_layers).cuda()
        self.criterion = torch.nn.L1Loss().cuda()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

    def save_model(self, num):
        model_out_path = os.path.join(self.modelname, self.modelname + '_params_' + str(num) + '.pkl')
        torch.save(self.model.state_dict(), model_out_path)

    def adjust_learning_rate(self, epoch):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr * (0.5 ** (epoch // 200))

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
                criterion = torch.nn.MSELoss()
                MSE_ = criterion(output.float(), label.float())
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
            f.close()

    def test(self):
        self.model = torch.nn.DataParallel(self.model)
        save_sr = os.path.join(self.modelname, self.modelname)
        make_dirs(save_sr)
        # ./rdn/rdn
        with torch.no_grad():
            for file in os.listdir(test_lr):
                read_path = os.path.join(test_lr, file)
                img = Image.open(read_path)

                inputs = transforms.ToTensor()(img).view(1, -1, img.size[1], img.size[0])
                para_savepath = os.path.join(self.modelname, self.modelname + '_params_' + str(self.epochs) + '.pkl')
                self.model.load_state_dict(torch.load(para_savepath))
                inputs = inputs.cuda()

                output = self.model(inputs)
                output_img = output.cpu().clone()
                output_img = output_img.squeeze(0)
                output_img[output_img < 0] = 0
                output_img[output_img > 1] = 1
                output_img = transforms.ToPILImage()(output_img)

                sr_name = os.path.join(save_sr, self.modelname + '_' + file)
                output_img.save(sr_name)

    def run(self, mode='train'):
        all_loss = []
        if mode == 'train':
            self.bulid_model()
            for epoch in range(1, self.epochs + 1):
                self.adjust_learning_rate(epoch)
                print("\n ==> Epoch {}".format(epoch))

                f_path = os.path.join(self.modelname, self.modelname + '_loss.txt')
                f = open(f_path, 'a+')
                print(" ==> Epoch {}".format(epoch), file=f)
                f.close()

                train_loss = self.train()
                all_loss.append(train_loss)
                # seg img test
                self.test_seg()
                if epoch % 10 == 0:
                    self.save_model(num=epoch)
                else:
                    pass
                if epoch == self.epochs:
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
                self.adjust_learning_rate(epoch)
                print("\n ==> Epoch {}".format(epoch))

                f_path = os.path.join(self.modelname, self.modelname + '_loss.txt')
                f = open(f_path, 'a+')
                print(" ==> Epoch {}".format(epoch), file=f)
                f.close()

                train_loss = self.train()
                all_loss.append(train_loss)
                # seg img test
                self.test_seg()
                if epoch % 10 == 0:
                    self.save_model(num=epoch)
                else:
                    pass
                if epoch == self.epochs:
                    plt.plot(all_loss)
                    plt_path = os.path.join(self.modelname, self.modelname + '_loss_' + str(epoch) + '.png')
                    plt.savefig(plt_path)
                else:
                    pass
            self.bulid_model()
            # full img test
            self.test()
