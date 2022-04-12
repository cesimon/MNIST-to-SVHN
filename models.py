import torch.nn as nn
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=0),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=0),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks, output_channels):
        super(GeneratorResNet, self).__init__()

        # The same model is used to generate SVHN from MNIST and MNIST from SVHN.
        # Pictures from SVHN have 3 channels, and the ones from MNIST have only one.
        # Therefore, when creating the generator, we use the shape of future input images
        # to adapt the number of channels given as input to the first convolution layer.
        input_channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, out_features, 7, stride=1, padding=0),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        # We use the output_channels variable to set the desired number of channels for the output,
        # because the same model is used to generate SVHN from MNIST and MNIST from SVHN.
        # Pictures from SVHN have 3 channels, and the ones from MNIST have only one.
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_features, output_channels, 7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Output shape of image discriminator (PatchGAN)
        self.output_shape = (1, 5, 5)

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=2)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 512, stride=1),
            nn.Conv2d(512, 1, 4, stride=1, padding=2)
        )

    def forward(self, img):
        return self.model(img)

# https://github.com/tkhkaeio/CyCADA/blob/b6f7795d7d80e788500ffee8e2d2527a62bf4b87/models/networks.py#L722
class LeNet(nn.Module): #for 32
    def __init__(self, input_nc):
        super(LeNet, self).__init__()

        sequence = [
            nn.Conv2d(input_nc, 20, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),

            nn.Flatten(),

            nn.Linear(50*5*5, 500),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(500, 10)
        ]
        self.net = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        self.out = self.net(input)
        return self.out
