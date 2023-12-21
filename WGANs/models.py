import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()

        # in the paper they don't use the batchnorm at the earlier layers
        self.net = nn.Sequential(
            # Input: N x channel_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, kernel_size=4, stride=2, padding=1), # 16 x 16
            self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1), # 8 x 8
            self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1), # 4 x 4
            nn.Conv2d(features_d*8, out_channels=1, kernel_size=4, stride=2, padding=0), # the output is only 1 channel, N x 1 x 1 x 1
            nn.Linear(1 * 1 * 1, 1) # N x 1 x 1 x 1 -> 1, the output of the critic
        )


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False # bias is set to False because we want to use BatchNorm
            ),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, channel_img, feature_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, feature_g*16, kernel_size=4, stride=1, padding=0), # N x feature_g*16 x 4 x 4
            self._block(feature_g*16, feature_g*8, kernel_size=4, stride=2, padding=1), # 8 x 8
            self._block(feature_g*8, feature_g*4, kernel_size=4, stride=2, padding=1), # 16 x 16
            self._block(feature_g*4, feature_g*2, kernel_size=4, stride=2, padding=1), # 32 x 32
            nn.ConvTranspose2d( # for input to the discriminator, 64 x 64
                feature_g*2,
                channel_img,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh() # good for image generation output ranges: [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), # why ReLU? because based on paper
        )
    
    def forward(self, x):
        return self.net(x)
    
def initialize_weights(models):
    for m in models.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 3, 3, 64, 64
    z_dim = 100
    X = torch.randn((N, in_channels, H, W))
    z = torch.randn((N, z_dim, 1, 1))

    disc = Discriminator(channels_img=in_channels, features_d=8)
    initialize_weights(disc)
    assert disc(X).shape == (N, 1, 1, 1)

    gen = Generator(z_dim=z_dim, channel_img=in_channels, feature_g=8)
    initialize_weights(gen)
    assert gen(z).shape == (N, in_channels, H, W)

    print("Success")
