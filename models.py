import torch
import torch.nn as nn
import torch.fft as fft

def get_activation(name):
    activations = {
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'LeakyReLU': nn.LeakyReLU,
        'ELU': nn.ELU,
        'None': nn.Identity
    }
    return activations[name]()

class LatentFCProjector(nn.Module):
    def __init__(self, input_size, latent_size, layer_shapes, activations):
        super(LatentFCProjector, self).__init__()
        layers = []
        in_features = input_size
        for i, out_features in enumerate(layer_shapes):
            layers.append(nn.Linear(in_features, out_features))
            if activations[i] != 'None':
                layers.append(get_activation(activations[i]))
            in_features = out_features
        layers.append(nn.Linear(in_features, latent_size))
        self.fc = nn.Sequential(*layers)
        self.latent_size = latent_size

    def forward(self, x):
        return self.fc(x)

class LatentRNNProjector(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, rnn_num_layers, latent_size):
        super(LatentRNNProjector, self).__init__()
        self.rnn = nn.LSTM(input_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, latent_size)
        self.latent_size = latent_size

    def forward(self, x):
        batch_1, batch_2, timesteps = x.size()
        out, _ = self.rnn(x.view(batch_1 * batch_2, timesteps))
        latent = self.fc(out).view(batch_1, batch_2, self.latent_size)
        return latent

class FourierTransformLayer(nn.Module):
    def forward(self, x):
        x_fft = fft.rfft(x, dim=-1)
        return x_fft

class LatentFourierProjector(nn.Module):
    def __init__(self, input_size, latent_size, layer_shapes, activations, pass_raw_len=None):
        super(LatentFourierProjector, self).__init__()
        self.fourier_transform = FourierTransformLayer()
        layers = []
        if pass_raw_len is None:
            pass_raw_len = input_size
        else:
            assert pass_raw_len <= input_size
        in_features = pass_raw_len + (input_size // 2 + 1) * 2  # (input_size // 2 + 1) real + imaginary parts
        for i, out_features in enumerate(layer_shapes):
            layers.append(nn.Linear(in_features, out_features))
            if activations[i] != 'None':
                layers.append(get_activation(activations[i]))
            in_features = out_features
        layers.append(nn.Linear(in_features, latent_size))
        self.fc = nn.Sequential(*layers)
        self.latent_size = latent_size
        self.pass_raw_len = pass_raw_len

    def forward(self, x):
        # Apply Fourier Transform
        x_fft = self.fourier_transform(x)
        # Separate real and imaginary parts and combine them
        x_fft_real_imag = torch.cat((x_fft.real, x_fft.imag), dim=-1)
        # Combine part of the raw input with Fourier features
        combined_input = torch.cat([x[:, -self.pass_raw_len:], x_fft_real_imag], dim=-1)
        # Process through fully connected layers
        latent = self.fc(combined_input)
        return latent

class MiddleOut(nn.Module):
    def __init__(self, latent_size, region_latent_size, num_peers):
        super(MiddleOut, self).__init__()
        self.num_peers = num_peers
        self.fc = nn.Linear(latent_size * 2 + 1, region_latent_size)

    def forward(self, my_latent, peer_latents, peer_metrics):
        new_latents = []
        for p in range(peer_latents.shape[-2]):
            peer_latent, metric = peer_latents[:, p, :], peer_metrics[:, p]
            combined_input = torch.cat((my_latent, peer_latent, metric.unsqueeze(1)), dim=-1)
            new_latent = self.fc(combined_input)
            new_latents.append(new_latent * metric.unsqueeze(1))
        
        new_latents = torch.stack(new_latents)
        averaged_latent = torch.mean(new_latents, dim=0)
        return averaged_latent

class Predictor(nn.Module):
    def __init__(self, region_latent_size, layer_shapes, activations):
        super(Predictor, self).__init__()
        layers = []
        in_features = region_latent_size
        for i, out_features in enumerate(layer_shapes):
            layers.append(nn.Linear(in_features, out_features))
            if activations[i] != 'None':
                layers.append(get_activation(activations[i]))
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, latent):
        return self.fc(latent)