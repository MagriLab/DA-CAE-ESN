import torch.nn as nn
import torch
from collections import OrderedDict
import random

random.seed(0)


class CAE(nn.Module):
    """
    Convolutional Autoencoder (CAE) for 1D PDEs.

    The encoder compresses the input sequence into a lower-dimensional latent space, and the decoder reconstructs the sequence from the latent representation.
    """

    def __init__(self, latent_size, weight_init_name="kaiming_uniform", random_seed=0):
        super(CAE, self).__init__()
        torch.manual_seed(random_seed)
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    ("enc_conv0", nn.Conv1d(in_channels=1,
                     out_channels=2, kernel_size=8, stride=1, padding=1)),
                    ("tanh0", nn.Tanh()),
                    ("enc_conv1", nn.Conv1d(in_channels=2,
                     out_channels=4, kernel_size=8, stride=2)),
                    ("tanh1", nn.Tanh()),
                    ("enc_conv2", nn.Conv1d(in_channels=4,
                     out_channels=8, kernel_size=3, stride=2)),
                    ("tanh2", nn.Tanh()),
                    ("enc_conv3", nn.Conv1d(in_channels=8,
                     out_channels=16, kernel_size=3, stride=2)),
                    ("tanh3", nn.Tanh()),
                    ("enc_flat", nn.Flatten()),
                    ("enc_linear_dense", nn.Linear(in_features=(
                        208), out_features=self.latent_size)),
                    ("tanh4", nn.Tanh()),
                ]
            )
        )

        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    ("dec_linear_dense", nn.Linear(
                        in_features=self.latent_size, out_features=(208))),
                    ("dec_tanh_dense", nn.Tanh()),
                    ("dec_unflat", nn.Unflatten(1, (16, 13))),
                    ("dec_tconv1", nn.ConvTranspose1d(in_channels=16,
                     out_channels=8, kernel_size=3, stride=2)),
                    ("dec_tconv1_2", nn.ConvTranspose1d(in_channels=8,
                     out_channels=8, kernel_size=3, stride=1, padding=0),),
                    ("dec_tanh1", nn.Tanh()),
                    ("dec_tconv2", nn.ConvTranspose1d(in_channels=8,
                     out_channels=4, kernel_size=5, stride=2)),
                    ("dec_tconv2_2", nn.ConvTranspose1d(in_channels=4,
                     out_channels=4, kernel_size=3, stride=1, padding=1),),
                    ("dec_tanh2", nn.Tanh()),
                    ("dec_tconv3", nn.ConvTranspose1d(in_channels=4,
                     out_channels=2, kernel_size=8, stride=2)),
                    ("dec_tconv3_2", nn.ConvTranspose1d(in_channels=2,
                     out_channels=1, kernel_size=3, stride=1, padding=1),),
                    ("dec_tanh3", nn.Tanh()),
                ]
            )
        )
        self.initialize_weights(weight_init_name)

    def initialize_weights(self, weight_init_name):
        weight_init_dict = {
            "xavier_uniform": nn.init.xavier_uniform_,
            "xavier_normal": nn.init.xavier_normal_,
            # default choice of torch, corresponds to He initialization
            "kaiming_uniform": nn.init.kaiming_uniform_,
            "kaiming_normal": nn.init.kaiming_uniform_,
        }

        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                weight_init_dict[weight_init_name](module.weight)
        for name, module in self.decoder.named_modules() or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv1d):
                weight_init_dict[weight_init_name](module.weight)

    def decode(self, input):
        return self.decoder(input)

    def encode(self, input):
        return self.encoder(input)

    def forward(self, input):
        """
        Forward pass through the CAE.

        Returns:
            encoded (torch.Tensor): Latent space representation
            decoded (torch.Tensor): Reconstruction of the input
        """
        encoded = self.encode(input)
        decoded = self.decode(encoded)
        return encoded, decoded
