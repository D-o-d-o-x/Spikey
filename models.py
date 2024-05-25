import torch
import torch.nn as nn

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

class LatentProjector(nn.Module):
    def __init__(self, input_size, latent_size, layer_shapes, activations):
        super(LatentProjector, self).__init__()
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
        out, _ = self.rnn(x)
        latent = self.fc(out)
        return latent

class MiddleOut(nn.Module):
    def __init__(self, latent_size, output_size, num_peers):
        super(MiddleOut, self).__init__()
        self.num_peers = num_peers
        self.fc = nn.Linear(latent_size * 2 + 1, output_size)

    def forward(self, my_latent, peer_latents, peer_correlations):
        new_latents = []
        for p in range(peer_latents.shape[-2]):
            peer_latent, correlation = peer_latents[:, p, :], peer_correlations[:, p]
            combined_input = torch.cat((my_latent, peer_latent, correlation.unsqueeze(1)), dim=-1)
            new_latent = self.fc(combined_input)
            new_latents.append(new_latent * correlation.unsqueeze(1))
        
        new_latents = torch.stack(new_latents)
        averaged_latent = torch.mean(new_latents, dim=0)
        return my_latent - averaged_latent

class Predictor(nn.Module):
    def __init__(self, output_size, layer_shapes, activations):
        super(Predictor, self).__init__()
        layers = []
        in_features = output_size
        for i, out_features in enumerate(layer_shapes):
            layers.append(nn.Linear(in_features, out_features))
            if activations[i] != 'None':
                layers.append(get_activation(activations[i]))
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, latent):
        return self.fc(latent)
