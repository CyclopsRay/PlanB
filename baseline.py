import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, cell_num, gene_num, latent_space=64, conv_channels=[32, 64]):
        super(AE, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, conv_channels[0], kernel_size=3, stride=2, padding=1),  # First convolutional layer
            nn.ReLU(True),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1),  # Second convolutional layer
            nn.ReLU(True),
            nn.Flatten(),  # Flatten the output for the linear layer
            nn.Linear(conv_channels[1] * (cell_num // 4) * (gene_num // 4), latent_space)  # Linear layer to latent space
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_space, conv_channels[1] * (cell_num // 4) * (gene_num // 4)),  # Linear layer from latent space
            nn.ReLU(True),
            nn.Unflatten(1, (conv_channels[1], cell_num // 4, gene_num // 4)),  # Unflatten to feed into conv transpose layers
            nn.ConvTranspose2d(conv_channels[1], conv_channels[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # First deconvolutional layer
            nn.ReLU(True),
            nn.ConvTranspose2d(conv_channels[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Second deconvolutional layer
            nn.Sigmoid()  # Sigmoid activation to get the final output
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x