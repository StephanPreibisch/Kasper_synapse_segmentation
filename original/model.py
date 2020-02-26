import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Two 2D convolutional layer: 2D conv + batch norm + ReLu 
    """
    def __init__(self, in_ch, out_ch, pad=0, bias=False):
        """
        Args:
        in_ch: number of input channels
        out_ch: number of output channels
        pad: amount of zero padding added to the sides of input
        bias: if add a learnable bias to the output
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=pad, bias=bias),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=pad, bias=bias),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace=True))

    def forward(self, img):
        out = self.conv_block(img)
        return out


class UpBlock(nn.Module):
    """
    Up block: Up-conv + crop + ConvBlock
    """
    def __init__(self, in_ch, out_ch, pad=0, bias=False):
        """
        Args:
        in_ch: number of input channels
        out_ch: number of output channels
        pad: amount of zero padding added to the sides of input
        bias: if add a learnable bias to the output
        """
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_ch=in_ch, out_ch=out_ch, pad=pad, bias=bias)

    def forward(self, img, down_layer):
        """
        Args:
        down_layer: layer in down-sampling to crop from
        """
        up = self.upsample(img)
        # Cropping down_layer
        _, _, h, w = down_layer.shape
        target_sz = up.shape[2:]
        low_x = (h-target_sz[0]) // 2
        high_x = low_x + target_sz[0]
        low_y = (w-target_sz[1]) // 2
        high_y = low_y + target_sz[1]
        cropped = down_layer[:, :, low_x:high_x, low_y:high_y]
        out = torch.cat([cropped, up], dim=1)
        out = self.conv_block(out)
        return out


class UNet(nn.Module):
    """
    2D UNet
    """
    def __init__(self, in_channels=1, base_filters=16, out_channels=1, depth=3, pad=0, bias=False):
        """
        Args:
        in_channels: number of input channels
        base_filters: numer of base filters
        out_channels: number of output channels
        depth: UNet depth
        pad: amount of zero padding added to the sides of input
        bias: if add a learnable bias to the output
        """
        super().__init__()
        self.down_modules = nn.ModuleList()
        for i in range(depth):
            self.down_modules.append(
                ConvBlock(in_ch=in_channels, out_ch=2**i*base_filters, pad=pad, bias=bias))
            in_channels = 2**i*base_filters
        self.up_modules = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.up_modules.append(
                UpBlock(in_ch=in_channels, out_ch=2**i*base_filters, pad=0, bias=False))
            in_channels = 2**i*base_filters
        self.final_module = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    
    def forward(self, img):
        down_layers = []
        for i, down in enumerate(self.down_modules):
            img = down(img)
            if i != len(self.down_modules) -1:
                down_layers.append(img)
                img = F.max_pool2d(img, 2)
        for i, up in enumerate(self.up_modules):
            img = up(img, down_layers[-1-i])
        img = self.final_module(img)
        img = torch.sigmoid(img)
        return img


if __name__ == "__main__":
    model = UNet()
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())