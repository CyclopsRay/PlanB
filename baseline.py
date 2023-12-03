import torch
import torch.nn as nn

# class AE(nn.Module):
#     def __init__(self, cell_num, gene_num, latent_space=64, conv_channels=[32, 64]):
#         super(AE, self).__init__()

#         # Encoder layers
#         # self.encoder = nn.Sequential(
#         #     nn.Conv2d(1, conv_channels[0], kernel_size=3, stride=2, padding=1),  # First convolutional layer
#         #     nn.ReLU(True),
#         #     nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1),  # Second convolutional layer
#         #     nn.ReLU(True),
#         #     nn.Flatten(),  # Flatten the output for the linear layer
#         #     nn.Linear(conv_channels[1] * (cell_num // 4) * (gene_num // 4), latent_space)  # Linear layer to latent space
#         # )

#         # # Decoder layers
#         # self.decoder = nn.Sequential(
#         #     nn.Linear(latent_space, conv_channels[1] * (cell_num // 4) * (gene_num // 4)),  # Linear layer from latent space
#         #     nn.ReLU(True),
#         #     nn.Unflatten(1, (conv_channels[1], cell_num // 4, gene_num // 4)),  # Unflatten to feed into conv transpose layers
#         #     nn.ConvTranspose2d(conv_channels[1], conv_channels[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # First deconvolutional layer
#         #     nn.ReLU(True),
#         #     nn.ConvTranspose2d(conv_channels[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Second deconvolutional layer
#         #     nn.Sigmoid()  # Sigmoid activation to get the final output
#         # )
#                 # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output: [batch, 16, D, 750]
#             nn.ReLU(True),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: [batch, 32, D/2, 375]
#             nn.ReLU(True),
#             nn.Flatten()
#         )

#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [batch, 16, D, 750]
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: [batch, 1, D*2, 1500]
#             nn.Sigmoid()
#         )
class AE(nn.Module):
    def __init__(self, latent_dim, D_height, D_width):
        super(AE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output size: [batch, 16, D_height/2, D_width/2]
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output size: [batch, 32, D_height/4, D_width/4]
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(32 * (D_height//4) * (D_width//4), latent_dim),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * (D_height//4) * (D_width//4)),
            nn.ReLU(True),
            nn.Unflatten(1, (32, D_height//4, D_width//4)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def latent(self, x):
        x = self.encoder(x)
        return x