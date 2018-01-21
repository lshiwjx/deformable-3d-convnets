import torch
import torch.nn as nn

from .deform3d.deform_conv3d_modules import ConvOffset3d


class BottleBlock(nn.Module):
    def __init__(self, channel_in, channel_mid, channel_out, downsample, group=1, stride=1, deform=False, dgroup=1):
        super(BottleBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, channel_mid, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(channel_mid)
        if deform == False:
            self.conv2 = nn.Conv3d(channel_mid, channel_mid, kernel_size=3, groups=group, padding=1, bias=False,
                                   stride=stride)
        else:
            self.conv_off = nn.Conv3d(channel_mid, dgroup * 3 * 27, kernel_size=3, stride=stride,
                                      padding=1, bias=True)
            self.conv2 = ConvOffset3d(channel_mid, channel_mid, kernel_size=3, groups=group, padding=1, bias=False,
                                      stride=stride, dilation=1, channel_per_group=channel_mid // dgroup)
        self.bn2 = nn.BatchNorm3d(channel_mid)
        self.conv3 = nn.Conv3d(channel_mid, channel_out, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(channel_out)
        self.relu = nn.ReLU(inplace=True)
        # self.down = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.downsample = downsample
        self.ratio = channel_out // channel_in
        self.deform = deform

        if self.deform:
            nn.init.constant(self.conv_off.weight.data, 0)
            nn.init.constant(self.conv_off.bias.data, 0)
        nn.init.kaiming_normal(self.conv1.weight.data, mode='fan_out')
        nn.init.kaiming_normal(self.conv2.weight.data, mode='fan_out')
        nn.init.kaiming_normal(self.conv3.weight.data, mode='fan_out')

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.deform:
            off = self.conv_off(out)
            out = self.conv2(out, off)
        else:
            out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        out = self.relu(out)

        if self.deform:
            return out, off
        else:
            return out, None


class ResNetXt1013d(nn.Module):
    def __init__(self, class_num, mode):
        super(ResNetXt1013d, self).__init__()
        self.mode = mode
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        downsample0 = nn.Sequential(
            nn.Conv3d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(256)
        )
        self.layer10 = BottleBlock(64, 128, 256, downsample0, 32)
        self.layer11 = BottleBlock(256, 128, 256, None, 32)
        self.layer12 = BottleBlock(256, 128, 256, None, 32)

        downsample1 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(512)
        )
        self.layer20 = BottleBlock(256, 256, 512, downsample1, 32, stride=2)
        self.layer21 = BottleBlock(512, 256, 512, None, 32, deform=False, dgroup=1)
        self.layer22 = BottleBlock(512, 256, 512, None, 32, deform=False, dgroup=1)
        self.layer23 = BottleBlock(512, 256, 512, None, 32, deform=False, dgroup=1)

        downsample2 = nn.Sequential(
            nn.Conv3d(512, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(1024)
        )
        self.layer30 = BottleBlock(512, 512, 1024, downsample2, 32, stride=2)
        self.layer31 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer32 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer33 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer34 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer35 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer36 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer37 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer38 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer39 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer310 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer311 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer312 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer313 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer314 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer315 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer316 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer317 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer318 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer319 = BottleBlock(1024, 512, 1024, None, 32)
        self.layer320 = BottleBlock(1024, 512, 1024, None, 32, deform=False, dgroup=1)
        self.layer321 = BottleBlock(1024, 512, 1024, None, 32, deform=False, dgroup=1)
        self.layer322 = BottleBlock(1024, 512, 1024, None, 32, deform=False, dgroup=1)

        downsample3 = nn.Sequential(
            nn.Conv3d(1024, 2048, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(2048)
        )
        self.layer40 = BottleBlock(1024, 1024, 2048, downsample3, 32, stride=2, deform=True, dgroup=1)
        self.layer41 = BottleBlock(2048, 1024, 2048, None, 32, deform=True, dgroup=1)
        self.layer42 = BottleBlock(2048, 1024, 2048, None, 32, deform=True, dgroup=1)

        self.dropout = nn.Dropout(inplace=True)
        self.fc = nn.Linear(2048, class_num)
        self.layers = []

        # init weight
        nn.init.kaiming_normal(self.conv1.weight.data, mode='fan_out')
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant(m.weight.data, 1)
                nn.init.constant(m.bias.data, 0)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x, _ = self.layer10(x)
        x, _ = self.layer11(x)
        x, _ = self.layer12(x)

        x, _ = self.layer20(x)
        x, _ = self.layer21(x)
        x, _ = self.layer22(x)
        x, _ = self.layer23(x)
        x, _ = self.layer30(x)
        x, _ = self.layer31(x)
        x, _ = self.layer32(x)
        x, _ = self.layer33(x)
        x, _ = self.layer34(x)
        x, _ = self.layer35(x)
        x, _ = self.layer36(x)
        x, _ = self.layer37(x)
        x, _ = self.layer38(x)
        x, _ = self.layer39(x)
        x, _ = self.layer310(x)
        x, _ = self.layer311(x)
        x, _ = self.layer312(x)
        x, _ = self.layer313(x)
        x, _ = self.layer314(x)
        x, _ = self.layer315(x)
        x, _ = self.layer316(x)
        x, _ = self.layer317(x)
        x, _ = self.layer318(x)
        x, _ = self.layer319(x)
        x, _ = self.layer320(x)
        x, _ = self.layer321(x)
        x, _ = self.layer322(x)

        x, _ = self.layer40(x)
        x, _ = self.layer41(x)
        x, _ = self.layer42(x)

        x = x.view(x.size(0), 2048, -1)
        x = torch.mean(x, 2)
        x = self.dropout(x)

        x = self.fc(x)
        return x, self.layers