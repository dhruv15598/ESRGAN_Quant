import torch
import torch.nn as nn
# from torchinfo import summary


class RD_block(nn.Module):
    def __init__(self, channels, growth_channels, residual_beta):
        super(RD_block, self).__init__()
        self.residual_beta = residual_beta
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, kernel_size=3,
                               stride=1, padding=1)
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, kernel_size=3,
                               stride=1, padding=1)

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.identity = nn.Identity()

    def forward(self, x):
        temp = x
        out1 = self.activation(self.conv1(x))
        out2 = self.activation(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.activation(self.conv3(torch.cat([x, out1, out2, ], 1)))
        out4 = self.activation(self.conv4(torch.cat([x, out1, out2, out3, ], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out6 = torch.mul(out5, self.residual_beta)
        out = torch.add(out6, temp)
        return out


class RRD_block(nn.Module):
    def __init__(self, channels, growth_channels, residual_beta):
        self.residual_beta = residual_beta
        super(RRD_block, self).__init__()
        self.block1 = RD_block(channels, growth_channels, residual_beta)
        self.block2 = RD_block(channels, growth_channels, residual_beta)
        self.block3 = RD_block(channels, growth_channels, residual_beta)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = torch.mul(out3, self.residual_beta)
        out = torch.add(out4, x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, upscale_factor):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))


class RRDBNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels, growth_channels, upscale_factor, residual_beta):
        super(RRDBNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=1, padding=1)
        self.res_block = nn.Sequential(*[RRD_block(channels, growth_channels, residual_beta) for _ in range(23)])

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1)

        self.upsample = nn.Sequential(
            UpsampleBlock(channels, upscale_factor), UpsampleBlock(channels, upscale_factor),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        self.conv4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

    def forward(self, x):
        temp = x
        out1 = self.conv1(x)
        out2 = self.conv2(self.res_block(out1))
        out3 = torch.add(out2, out1)
        out4 = self.upsample(out3)
        out5 = self.conv3(out4)
        out = self.conv4(out5)

        out = torch.clamp_(out, 0.0, 1.0)
        return out


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.disc = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
#             nn.LeakyReLU(negative_slope=0.2),
#
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(negative_slope=0.2),
#
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(negative_slope=0.2),
#
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(negative_slope=0.2),
#
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(negative_slope=0.2),
#
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(negative_slope=0.2),
#
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(negative_slope=0.2),
#
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.AdaptiveAvgPool2d((6, 6)),
#             nn.Flatten(),
#             nn.Linear(512 * 6 * 6, 1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(1024, 1),
#
#         )
#
#     def forward(self, x):
#         return self.disc(x)

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 64 x 64
            nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 32 x 32
            nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 8 x 8
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # ex
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out
#############################################
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        m.weight.data *= 0.1
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gen = RRDBNet(3, 3, 64, 32, 2, 0.2).to(device)
# # # gen_opt = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.9, 0.999))
# # # gen_model = gen.apply(weights_init)
# summary(gen, input_size=(16, 3, 64, 64))
# # # dis = Discriminator().to(device)
# # # summary(dis, input_size=(16, 3, 256, 256))

#############################################