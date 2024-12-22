# coding=utf-8
import torch
from thop import profile
from thop import clever_format


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


model = RDN(2, 3, 8, 8, 4, 2)
input_torch = torch.randn(1, 3, 360, 240)
flops, params = profile(model, inputs=(input_torch, ))
flops, params = clever_format([flops, params], "%.3f")
