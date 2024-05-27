import torch
import torch.nn as nn
import torch.fft as fft
import pywt

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

class FeatureExtractor(nn.Module):
    def __init__(self, input_size, transforms):
        super(FeatureExtractor, self).__init__()
        self.input_size = input_size
        self.transforms = self.build_transforms(transforms)

    def build_transforms(self, config):
        transforms = []
        for item in config:
            transform_type = item['type']
            length = item.get('length', self.input_size)
            if length in [None, -1]:
                length = self.input_size

            if transform_type == 'identity':
                transforms.append(('identity', length))
            elif transform_type == 'fourier':
                transforms.append(('fourier', length))
            elif transform_type == 'wavelet':
                wavelet_type = item['wavelet_type']
                transforms.append(('wavelet', wavelet_type, length))
        return transforms

    def forward(self, x):
        batch_1, batch_2, timesteps = x.size()
        x = x.view(batch_1 * batch_2, timesteps)  # Combine batch dimensions for processing
        outputs = []
        for transform in self.transforms:
            if transform[0] == 'identity':
                _, length = transform
                outputs.append(x[:, -length:])
            elif transform[0] == 'fourier':
                _, length = transform
                fourier_transform = fft.fft(x[:, -length:], dim=1)
                fourier_real = fourier_transform.real
                fourier_imag = fourier_transform.imag
                outputs.append(fourier_real)
                outputs.append(fourier_imag)
            elif transform[0] == 'wavelet':
                _, wavelet_type, length = transform
                coeffs = pywt.wavedec(x[:, -length:].cpu().numpy(), wavelet_type)
                wavelet_coeffs = [torch.tensor(coeff, dtype=torch.float32, device=x.device) for coeff in coeffs]
                wavelet_coeffs = torch.cat(wavelet_coeffs, dim=1)
                outputs.append(wavelet_coeffs)
        concatenated_outputs = torch.cat(outputs, dim=1)
        concatenated_outputs = concatenated_outputs.view(batch_1, batch_2, -1)  # Reshape back to original batch dimensions
        return concatenated_outputs

    def compute_output_size(self):
        size = 0
        for transform in self.transforms:
            if transform[0] == 'identity':
                _, length = transform
                size += length
            elif transform[0] == 'fourier':
                _, length = transform
                size += length * 2  # Fourier transform outputs both real and imaginary parts
            elif transform[0] == 'wavelet':
                _, wavelet_type, length = transform
                # Find the true size of the wavelet coefficients
                test_signal = torch.zeros(length)
                coeffs = pywt.wavedec(test_signal.numpy(), wavelet_type)
                wavelet_size = sum(len(c) for c in coeffs)
                size += wavelet_size
        return size

class LatentFCProjector(nn.Module):
    def __init__(self, feature_size, latent_size, layer_shapes, activations):
        super(LatentFCProjector, self).__init__()
        layers = []
        in_features = feature_size
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
    def __init__(self, feature_size, rnn_hidden_size, rnn_num_layers, latent_size):
        super(LatentRNNProjector, self).__init__()
        self.rnn = nn.LSTM(feature_size, rnn_hidden_size, rnn_num_layers, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, latent_size)
        self.latent_size = latent_size

    def forward(self, x):
        batch_1, batch_2, timesteps = x.size()
        out, _ = self.rnn(x.view(batch_1 * batch_2, timesteps))
        latent = self.fc(out).view(batch_1, batch_2, self.latent_size)
        return latent

class MiddleOut(nn.Module):
    def __init__(self, latent_size, region_latent_size, num_peers, residual=False):
        super(MiddleOut, self).__init__()
        if residual:
            assert latent_size == region_latent_size
        if num_peers == 0:
            assert latent_size == region_latent_size
        self.num_peers = num_peers
        self.fc = nn.Linear(latent_size * 2 + 1, region_latent_size)
        self.residual = residual

    def forward(self, my_latent, peer_latents, peer_metrics):
        if self.num_peers == 0:
            return my_latent
        new_latents = []
        for p in range(peer_latents.shape[-2]):
            peer_latent, metric = peer_latents[:, p, :], peer_metrics[:, p]
            combined_input = torch.cat((my_latent, peer_latent, metric.unsqueeze(1)), dim=-1)
            new_latent = self.fc(combined_input)
            if self.residual:
                new_latent = new_latent * metric.unsqueeze(1)
            new_latents.append(new_latent)
        
        new_latents = torch.stack(new_latents)
        averaged_latent = torch.mean(new_latents, dim=0)
        if self.residual:
            return my_latent - averaged_latent
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