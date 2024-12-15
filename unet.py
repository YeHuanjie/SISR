# coding=utf-8
import torch
import functools
from rcan import RCAB


class UnetBlock(torch.nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=torch.nn.BatchNorm2d, use_dropout=False,
                 up='Upsample'):
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
            up (string)         -- upsample methods -- whether 'PixelShuffle' or 'ConvTranspose' or 'Upsample'
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
            if up == 'ConvTranspose':
                upconv = torch.nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                                  kernel_size=4, stride=2,
                                                  padding=1)
            elif up == 'Upsample':
                upconv = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='nearest'),
                    torch.nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
                )
            else:
                upconv = torch.nn.Sequential(
                    torch.nn.Conv2d(inner_nc * 2, outer_nc * 2 * 2, kernel_size=3, stride=1, padding=1),
                    torch.nn.PixelShuffle(2),
                )
            down = [downconv]
            up = [RCAB(inner_nc * 2), uprelu, upconv, torch.nn.Tanh()]
            # up = [uprelu, upconv, torch.nn.Tanh()]
            model = down + [submodule] + up

        elif innermost:
            if up == 'ConvTranspose':
                upconv = torch.nn.ConvTranspose2d(inner_nc, outer_nc,
                                                  kernel_size=4, stride=2,
                                                  padding=1, bias=use_bias)
            elif up == 'Upsample':
                upconv = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='nearest'),
                    torch.nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1)
                )
            else:
                upconv = torch.nn.Sequential(
                    torch.nn.Conv2d(inner_nc, outer_nc * 2 * 2, kernel_size=3, stride=1, padding=1),
                    torch.nn.PixelShuffle(2),
                )
            down = [downrelu, downconv]
            up = [RCAB(inner_nc), uprelu, upconv, upnorm]
            # up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            if up == 'ConvTranspose':
                upconv = torch.nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                                  kernel_size=4, stride=2,
                                                  padding=1, bias=use_bias)
            elif up == 'Upsample':
                upconv = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='nearest'),
                    torch.nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
                )
            else:
                upconv = torch.nn.Sequential(
                    torch.nn.Conv2d(inner_nc * 2, outer_nc * 2 * 2, kernel_size=3, stride=1, padding=1),
                    torch.nn.PixelShuffle(2),
                )
            down = [downrelu, downconv, downnorm]
            up = [RCAB(inner_nc * 2), uprelu, upconv, upnorm]
            # up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [torch.nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # add skip connections
            return torch.cat([x, self.model(x)], 1)


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

        self.head = torch.nn.Conv2d(3, 3, 3, padding=3 // 2)
        self.tail = torch.nn.Conv2d(3, 3, 3, padding=3 // 2)

    def forward(self, x):
        # y = self.head(x)
        # out = self.model(y)
        # out = y + out
        # out = self.tail(out)
        # return out

        out = self.model(x)
        return out
