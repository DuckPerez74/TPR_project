import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=12, hidden_dim=30):
        super(AutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 24),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(24),
            nn.Dropout(0.1),
            nn.Linear(24, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, encoding_dim),
            # No activation on bottleneck to preserve full information
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 24),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(24),
            nn.Dropout(0.1),
            nn.Linear(24, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def reconstruction_error(self, x):
        reconstructed = self.forward(x)
        mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse
