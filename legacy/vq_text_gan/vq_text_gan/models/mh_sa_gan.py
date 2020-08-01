import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, model_dim):
        super().__init__()

        self.input_to_model = nn.Linear(input_dim, model_dim)
        self.activation = nn.LeakyReLU(0.2)

        self.encoder_layer = nn.TransformerEncoderLayer(model_dim, 1, 512, dropout=0)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 1)

        self.model_to_output = nn.Linear(model_dim, output_dim)

    def forward(self, input):
        out = self.activation(self.input_to_model(input))
        out = self.encoder(out)
        out = self.model_to_output(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_dim, model_dim):
        super().__init__()

        self.input_to_model = nn.Linear(input_dim, model_dim)
        self.activation = nn.LeakyReLU(0.2)

        self.encoder_layer = nn.TransformerEncoderLayer(model_dim, 1, 512, dropout=0.3)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 1)

        self.model_to_output = nn.Linear(model_dim, 1)

    def forward(self, input):
        out = self.activation(self.input_to_model(input))
        out = self.encoder(out)
        out = self.model_to_output(out)
        return out
