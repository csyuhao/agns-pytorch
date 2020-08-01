import torch.nn as nn


class DeConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding=2):
        super(DeConvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.deconv(input)
        x = self.bn(x)
        return self.relu(x)


class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.args = list(args)

    def forward(self, input):
        batch_size = input.shape[0]
        return input.reshape([batch_size] + self.args)


class Generator(nn.Module):
    '''Generating eyeglasses patterns
    '''

    def __init__(self, z_dim=25):
        super(Generator, self).__init__()
        self.head_block = nn.Sequential(*[
            nn.Linear(z_dim, 7040),
            nn.BatchNorm1d(7040),
            nn.ReLU(),
            Reshape(160, 4, 11)
        ])
        self.deconv_list = nn.Sequential(*[
            DeConvBlock(160, 80, 6, 2, 2),
            DeConvBlock(80, 40, 6, 2, 2),
            DeConvBlock(40, 20, 6, 2, 2)
        ])
        self.end = nn.Sequential(*[
            nn.ConvTranspose2d(20, 3, 6, 2, 2),
            nn.Tanh(),
        ])

    def forward(self, input):
        x = self.head_block(input)
        x = self.deconv_list(x)
        return self.end(x)


if __name__ == '__main__':

    from torchsummary import summary
    g = Generator().cuda()
    summary(g, (25,))
