import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        # in the paper they don't use the batchnorm at the earlier layers
        self.net = nn.Sequential(
            # Input: N x channel_img x 64 x 64
            nn.Conv2d(channels_img+1, features_d, kernel_size=4, stride=2, padding=1), # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, kernel_size=4, stride=2, padding=1), # 16 x 16
            self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1), # 8 x 8
            self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1), # 4 x 4
            nn.Conv2d(features_d*8, out_channels=1, kernel_size=4, stride=2, padding=0), # the output is only 1 channel, N x 1 x 1 x 1
            nn.Sigmoid() # N x 1 x 1 x 1
        )

        # CEIL([(F + 2P - K) / S] + 1)
        # Embed
        self.embed = nn.Embedding(num_classes, img_size*img_size)


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
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x, labels):
        embeddings = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embeddings], dim=1)
        return self.net(x)
    
class Generator(nn.Module):
    def __init__(self, 
                 z_dim, 
                 channel_img, 
                 feature_g,
                 num_classes,
                 img_size,
                 embed_size):
        super(Generator, self).__init__()
        
        self.img_size = img_size
        self.net = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim + embed_size, feature_g*16, kernel_size=4, stride=1, padding=0), # N x feature_g*16 x 4 x 4
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

        self.embed = nn.Embedding(num_classes, embed_size)

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
    
    def forward(self, x, labels):
        # latent vector z: N x Noise Dim x 1 x 1
        embeddings = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embeddings], dim=1)
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
    N, in_channels, H, W = 1, 3, 64, 64
    z_dim = 100
    X = torch.randn((N, in_channels, H, W))
    z = torch.randn((N, z_dim, 1, 1))

    disc = Discriminator(channels_img=in_channels, features_d=8, num_classes=10, img_size=64)
    initialize_weights(disc)
    assert disc(X, torch.tensor([1])).shape == (N, 1, 1, 1)

    gen = Generator(z_dim=z_dim, channel_img=in_channels, feature_g=8, num_classes=10, img_size=64, embed_size=100)
    initialize_weights(gen)
    assert gen(z, torch.tensor([1])).shape == (N, in_channels, H, W)

    print("Success")
