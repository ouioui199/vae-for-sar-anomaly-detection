from typing import List, Sequence

import torch
from torch import nn, Tensor
from torchcvnn import nn as c_nn

from models.base_model import BaseVAE


class Encoder_block(nn.Sequential):
    def __init__(
            self, 
            in_ch: int, 
            out_ch: int,
            kernel_size: int = 4,
            stride: Sequence[int] = (2, 2),
            padding: Sequence[int] = (1, 1)
        ) -> None:
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        bn = nn.BatchNorm2d(out_ch)
        lr = nn.LeakyReLU(0.1)
        super().__init__(conv, bn, lr)


class ConvEncoder(nn.Sequential):
    def __init__(
            self, 
            im_ch: int, 
            c1f: int = 32, 
            nz: int = 128, 
            patch_size: int = 64
        ) -> None:
        super().__init__()
        fc_in_features = int(((patch_size / 16) ** 2) * int(c1f / 2))

        e1 = Encoder_block(im_ch, c1f)
        e2 = Encoder_block(c1f, c1f * 2)
        e3 = Encoder_block(c1f * 2, c1f * 4)
        e4 = Encoder_block(c1f * 4, c1f * 4)
        conv1 = nn.Conv2d(c1f * 4, int(c1f / 2), 1) # 1d conv
        flat = nn.Flatten()
        fc = nn.Linear(fc_in_features, nz)
        
        super().__init__(e1, e2, e3, e4, conv1, flat, fc)


class Decoder_block(nn.Sequential):
    def __init__(
            self,
            in_ch: int,
            out_ch: int
        ) -> None:
        upsample = nn.UpsamplingNearest2d(scale_factor=2)
        conv = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        bn = nn.BatchNorm2d(out_ch)
        lr = nn.LeakyReLU(0.1)

        super().__init__(upsample, conv, bn, lr)
    

class ConvDecoder(nn.Module):
    def __init__(
            self,
            im_ch: int,
            c1f: int = 32,
            nz: int = 128,
            patch_size: int = 64
        ) -> None:
        super().__init__()

        fc_in_features = int(((patch_size / 16) ** 2) * int(c1f / 2))

        self.patche_size = patch_size
        self.c1f = c1f

        self.fc = nn.Linear(nz, fc_in_features)
        self.decoder = nn.Sequential(
            nn.Conv2d(int(c1f / 2), c1f * 4, 1),
            Decoder_block(c1f * 4, c1f * 4),
            Decoder_block(c1f * 4, c1f * 2),
            Decoder_block(c1f * 2, c1f),
            Decoder_block(c1f, c1f),
            nn.Conv2d(c1f, im_ch, 1),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc(x)
        out = out.view(-1, int(self.c1f / 2), int(self.patche_size / 16), int(self.patche_size / 16))
        out = self.decoder(out)
        return out
    

class Discriminator(nn.Sequential):
    def __init__(self, nz: int = 128) -> None:
        lin1 = nn.Linear(nz, int(nz / 2))
        lin2 = nn.Linear(int(nz / 2), int(nz / 4))
        lin3 = nn.Linear(int(nz / 4), int(nz / 8))
        lin4 = nn.Linear(int(nz / 8), 1)

        relu = nn.ReLU()
        drop = nn.Dropout(p=0.2)
        act = nn.Sigmoid()
        super().__init__(lin1, drop, relu, lin2, drop, relu, lin3, drop, relu, lin4, act)


class ConvLRVAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool = False) -> None:
        super().__init__()
        # self.residual = residual
        # self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        # self.up = nn.UpsamplingNearest2d(scale_factor=2)
            # c_nn.UpsampleFFT(scale_factor=2),
        self.conv_lr = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        # x = self.up(x)
        # if self.residual:
        #     residual = self.skip(x)
        #     return self.conv_lr(x) + residual
        return self.conv_lr(x)


class VanillaVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size: int,
                 hidden_dims: List = [16, 32, 64, 128, 128],
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        in_channels_ = in_channels
        self.latent_dim = latent_dim

        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, 3, stride=2, padding=1),
                    # nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        fc_features = hidden_dims[-1] * (input_size // (2 ** len(hidden_dims))) ** 2
        modules.append(nn.Flatten())
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(fc_features, latent_dim)
        self.fc_var = nn.Linear(fc_features, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, fc_features)
        
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                ConvLRVAE(hidden_dims[i], hidden_dims[i + 1], residual=True)
            )

        self.decoder_in_features = hidden_dims[0]
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            # nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], in_channels_, 3, padding=1),
            nn.Tanh()
        )

    def encode(self, input: Tensor) -> Sequence[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        B, D = result.shape
        H = W = int((D / self.decoder_in_features) ** 0.5)
        result = result.view(B, self.decoder_in_features, H, W)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor) -> Sequence[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    

class VanillaVAE2(VanillaVAE):
    def __init__(self,
        in_channels: int,
        latent_dim: int,
        input_size: int,
        hidden_dims: List = [16, 32, 64, 128, 128],
    ) -> None:
        super(VanillaVAE, self).__init__()
        in_channels_ = in_channels
        self.latent_dim = latent_dim

        modules = []
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(h_dim, h_dim, 3, padding=1),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(2, 2)
                )
            )
            in_channels = h_dim

        fc_features = hidden_dims[-1] * (input_size // (2 ** len(hidden_dims))) ** 2
        modules.append(nn.Flatten())
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(fc_features, latent_dim)
        self.fc_var = nn.Linear(fc_features, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, fc_features)
        
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                ConvLRVAE(hidden_dims[i], hidden_dims[i + 1], residual=True)
            )

        self.decoder_in_features = hidden_dims[0]
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], in_channels_, 3, padding=1),
            nn.Sigmoid()
        )


""" convolution + leaky relu """
class Conv_LR(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, depth: int = 1) -> None:
        super().__init__()

        self.depth = depth
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=(1, 1))
        self.leakyRelu  = nn.LeakyReLU(0.1)
        if depth > 1:
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=(1,1))
        
    def forward(self, x: Tensor) -> Tensor:
        
        if self.depth == 1:
            return self.leakyRelu(self.conv1(x))
        
        x = self.leakyRelu(self.conv1(x))
        for _ in range(self.depth - 1):
            x = self.leakyRelu(self.conv2(x))

        return x
    

""" One layer of the encoder"""
class Encoder(nn.Module):

    def __init__(self, in_c: int, out_c: int, depth: int = 1) -> None:
        super().__init__()

        self.CL = Conv_LR(in_c, out_c, depth)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.CL(x)
        x = self.pool(x)
        
        return x


""" One layer of the decoder"""
class Decoder(nn.Module):

    def __init__(self, in_c: int, skip_c: int, out_c: int) -> None:
        super().__init__()

        self.CL = Conv_LR(in_c + skip_c, out_c, depth=2)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.CL(x)
        return x


""" Last layer of the decoder """    
class Out(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c1: int, out_c2: int, out_c3: int = 1) -> None:
        super().__init__()

        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.CL1 = Conv_LR(in_c + skip_c, out_c1)
        self.CL2 = Conv_LR(out_c1, out_c2)
        self.C = nn.Conv2d(out_c2, out_c3, 3, padding=(1, 1))
        
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.CL1(x)
        x = self.CL2(x)
        x = self.C(x)
        return x
    
    
""" The UNet model """    
class UNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        """ Encoder """
        self.e1 = Encoder(1, 48, depth=2)
        self.e2 = Encoder(48, 48)

        """ Bottleneck """
        self.b = Conv_LR(48, 48)
        
        """ Decoder """
        self.d1 = Decoder(48, 48, 96)
        self.d2 = Decoder(96, 48, 96)
        
        """ Out """
        self.o = Out(96, 1, 64, 32, 1)
        
    def forward(self, input: Tensor) -> Tensor:
        
        s1 = self.e1(input)
        s2 = self.e2(s1)
        s3 = self.e2(s2)
        s4 = self.e2(s3)
        s = self.e2(s4)
        s = self.b(s)
        s = self.d1(s, s4)
        s = self.d2(s, s3)
        s = self.d2(s, s2)
        s = self.d2(s, s1)
        s = self.o(s, input)
        
        return input - s


""" The UNet model """    
class UNet_v2(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        """ Encoder """
        self.e1 = Encoder(1, 48, depth=2)
        self.e2 = Encoder(48, 48)
        self.e3 = Encoder(48, 48)
        self.e4 = Encoder(48, 48)
        self.e5 = Encoder(48, 48)

        """ Bottleneck """
        self.b = Conv_LR(48, 48)
        
        """ Decoder """
        self.d1 = Decoder(48, 48, 96)
        self.d2 = Decoder(96, 48, 96)
        self.d3 = Decoder(96, 48, 96)
        self.d4 = Decoder(96, 48, 96)
        
        """ Out """
        self.o = Out(96, 1, 64, 32, 1)
        
    def forward(self, input: Tensor) -> Tensor:
        
        s1 = self.e1(input)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        s = self.e5(s4)
        s = self.b(s)
        s = self.d1(s, s4)
        s = self.d2(s, s3)
        s = self.d3(s, s2)
        s = self.d4(s, s1)
        s = self.o(s, input)
        
        return input - s
