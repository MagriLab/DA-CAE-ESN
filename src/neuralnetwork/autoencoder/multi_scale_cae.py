import torch
import torch.nn as nn
from collections import OrderedDict


class MultiScaleCAE(nn.Module):
    """
    Multi-Scale Convolutional Autoencoder (CAE) for 2D data.

    - Multiple encoder-decoder branches are built with different kernel sizes to capture multi-scale spatial features.
    - Outputs from the branches are summed up for a unified latent state and reconstruction.
    """

    def __init__(
        self,
        latent_size,
        kernel_size_list,
        weight_init_name="kaiming_uniform",
        random_seed=0,
        in_channels=1
    ):
        super(MultiScaleCAE, self).__init__()
        torch.manual_seed(random_seed)
        self.latent_size = latent_size
        self.kernel_size_list = kernel_size_list
        self.out_channels = 32
        self.in_channels = in_channels
        self.encoder = nn.ModuleList(
            [self.init_encoder(kernel_size) for kernel_size in kernel_size_list]
        )
        self.decoder = nn.ModuleList(
            [self.init_decoder(kernel_size) for kernel_size in kernel_size_list]
        )
        # Initialize weights
        self.initialize_weights(weight_init_name)
        

    def init_encoder(self, kernel_size):
        encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc_conv0",
                        nn.Conv2d(
                            in_channels=self.in_channels,
                            out_channels=4,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("tanh0", nn.Tanh()),
                    (
                        "enc_conv1",
                        nn.Conv2d(
                            in_channels=4,
                            out_channels=8,
                            kernel_size=kernel_size,
                            stride=2,
                        ),
                    ),
                    ("tanh1", nn.Tanh()),
                    (
                        "enc_conv2",
                        nn.Conv2d(
                            in_channels=8,
                            out_channels=16,
                            kernel_size=kernel_size,
                            stride=2,
                        ),
                    ),
                    ("tanh2", nn.Tanh()),
                    (
                        "enc_conv3",
                        nn.Conv2d(
                            in_channels=16,
                            out_channels=32,
                            kernel_size=kernel_size,
                            stride=2,
                        ),
                    ),
                    ("tanh3", nn.Tanh()),
                    ("enc_flat", nn.Flatten()),
                    (
                        "enc_linear_dense",
                        nn.Linear(
                            in_features=(
                                calculate_flattened_dim(kernel_size) ** 2
                                * self.out_channels
                            ),
                            out_features=self.latent_size,
                        ),
                    ),
                    ("tanh4", nn.Tanh()),
                ]
            )
        )
        return encoder

    def init_decoder(self, kernel_size):
        decoder = nn.Sequential(
            OrderedDict(
                [
                    # Fully connected layer to reshape the latent vector
                    (
                        "dec_linear_dense",
                        nn.Linear(
                            in_features=self.latent_size,
                            out_features=(
                                calculate_flattened_dim(kernel_size) ** 2 * 32
                            ),
                        ),
                    ),
                    ("tanh0", nn.Tanh()),
                    (
                        "dec_unflat",
                        nn.Unflatten(
                            dim=1,
                            unflattened_size=(
                                32,
                                calculate_flattened_dim(kernel_size),
                                calculate_flattened_dim(kernel_size),
                            ),
                        ),
                    ),
                    # Transposed convolutions for upsampling
                    (
                        "dec_deconv3",
                        nn.ConvTranspose2d(
                            in_channels=32,
                            out_channels=16,
                            kernel_size=kernel_size,
                            stride=2,
                        ),
                    ),
                    ("tanh1", nn.Tanh()),
                    (
                        "dec_deconv2",
                        nn.ConvTranspose2d(
                            in_channels=16,
                            out_channels=8,
                            kernel_size=kernel_size,
                            stride=2,
                        ),
                    ),
                    ("tanh2", nn.Tanh()),
                    (
                        "dec_deconv1",
                        nn.ConvTranspose2d(
                            in_channels=8,
                            out_channels=4,
                            kernel_size=kernel_size,
                            stride=2,
                        ),
                    ),
                    ("tanh3", nn.Tanh()),
                    # Final layer with output_padding to ensure correct dimensions
                    (
                        "dec_deconv0",
                        nn.ConvTranspose2d(
                            in_channels=4,
                            out_channels=self.in_channels,
                            kernel_size=kernel_size + 1,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("tanh4", nn.Tanh()),
                ]
            )
        )
        return decoder

    def initialize_weights(self, weight_init_name):
        weight_init_dict = {
            "xavier_uniform": nn.init.xavier_uniform_,
            "xavier_normal": nn.init.xavier_normal_,
            "kaiming_uniform": nn.init.kaiming_uniform_,
            "kaiming_normal": nn.init.kaiming_normal_,  # Corrected typo here
        }

        # Initialize encoder weights
        for name, module in self.encoder.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight_init_dict[weight_init_name](module.weight)

        # Initialize decoder weights
        for decoder in self.decoder:
            for name, module in decoder.named_modules():
                if isinstance(module, (nn.ConvTranspose2d, nn.Linear)):
                    weight_init_dict[weight_init_name](module.weight)

        print(f"Weights initialized with {weight_init_name}")

    def forward(self, input):
        """
        Forward pass through all branches, run in parallel.
        Returns:
            encoded: latent space state (sum over branches)
            decoded: reconstructed output (sum over branches)
        """
        encoded = self.encode(input)
        decoded = self.decode(encoded)
        return encoded, decoded

    def encode(self, input):
        """
        Parallel execution of encoder branches using torch.jit.fork.
        Sum outputs to form a latent state.
        """
        encoded_futures = [torch.jit.fork(encoder, input) for encoder in self.encoder]
        encoded = sum([torch.jit.wait(fut) for fut in encoded_futures])
        return encoded

    def decode(self, encoded):
        """
        Parallel execution of all decoder branches.
        Sum their outputs to form the physical state reconstruction.
        """
        decoded_futures = [torch.jit.fork(decoder, encoded) for decoder in self.decoder]
        decoded = sum([torch.jit.wait(fut) for fut in decoded_futures])
        return decoded


def calculate_flattened_dim(kernel_size):
    out_dim = output_shape(48, kernel_size, padding=1, stride=1)
    for _ in range(3):  # Number of layers is fixed to 3, as in the original code
        out_dim = output_shape(out_dim, kernel_size, padding=0, stride=2)
    return out_dim


def output_shape(input_dim, kernel_size, padding, stride):
    return (input_dim - kernel_size + 2 * padding) // stride + 1
