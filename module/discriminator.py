import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.lerelu = nn.LeakyReLU()

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        return self.lerelu(x)


class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.args = list(args)

    def forward(self, input):
        batch_size = input.shape[0]
        return input.reshape([batch_size] + self.args)


class Discriminator(nn.Module):
    '''Discriminate eyeglasses patterns
    '''

    def __init__(self, n_channels=3):
        super(Discriminator, self).__init__()
        self.head_block = nn.Sequential(*[
            nn.Conv2d(n_channels, 20, 3, 2, 1),
            nn.LeakyReLU(),
        ])
        self.deconv_list = nn.Sequential(*[
            ConvBlock(20, 40, 3, 2, 1),
            ConvBlock(40, 80, 3, 2, 1),
            ConvBlock(80, 160, 3, 2, 1),
        ])
        self.end = nn.Sequential(*[
            Reshape(7040),
            nn.Linear(7040, 1),
            nn.Sigmoid()
        ])

    def forward(self, input):
        x = self.head_block(input)
        x = self.deconv_list(x)
        return self.end(x)


if __name__ == '__main__':

    from torchsummary import summary
    d = Discriminator().cuda()
    summary(d, (3, 64, 176))
