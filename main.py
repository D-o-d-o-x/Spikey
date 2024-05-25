import os
import torch
import torch.nn as nn
import numpy as np
import random
from utils import download_and_extract_data, load_all_wavs, split_data_by_time, compute_correlation_matrix, visualize_prediction, plot_delta_distribution
from models import LatentProjector, LatentRNNProjector, MiddleOut, Predictor
from bitstream import IdentityEncoder, ArithmeticEncoder, Bzip2Encoder
import wandb
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import slate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpikeRunner:
    def __init__(self, config):
        self.config = config
        self.name = slate.consume(config, 'name', default='Test')

        training_config = slate.consume(config, 'training', expand=True)
        data_config = slate.consume(config, 'data', expand=True)

        data_url = slate.consume(data_config, 'url')
        data_dir = slate.consume(data_config, 'directory')
        cut_length = slate.consume(data_config, 'cut_length', None)
        download_and_extract_data(data_url, data_dir)
        all_data = load_all_wavs(data_dir, cut_length)

        split_ratio = slate.consume(data_config, 'split_ratio', 0.5)
        self.train_data, self.test_data = split_data_by_time(all_data, split_ratio)

        # Compute correlation matrix
        self.correlation_matrix = compute_correlation_matrix(self.train_data)

        # Model setup
        latent_projector_type = slate.consume(config, 'latent_projector.type', default='fc')

        if latent_projector_type == 'fc':
            self.projector = LatentProjector(**slate.consume(config, 'latent_projector', expand=True)).to(device)
        elif latent_projector_type == 'rnn':
            self.projector = LatentRNNProjector(**slate.consume(config, 'latent_projector', expand=True)).to(device)

        self.middle_out = MiddleOut(**slate.consume(config, 'middle_out', expand=True)).to(device)
        self.predictor = Predictor(**slate.consume(config, 'predictor', expand=True)).to(device)

        # Training parameters
        self.epochs = slate.consume(training_config, 'epochs')
        self.batch_size = slate.consume(training_config, 'batch_size')
        self.learning_rate = slate.consume(training_config, 'learning_rate')
        self.eval_freq = slate.consume(training_config, 'eval_freq')
        self.save_path = slate.consume(training_config, 'save_path')

        # Evaluation parameter
        self.full_compression = slate.consume(config, 'evaluation.full_compression', default=False)

        # Bitstream encoding
        bitstream_type = slate.consume(config, 'bitstream_encoding.type', default='identity')
        if bitstream_type == 'identity':
            self.encoder = IdentityEncoder()
        elif bitstream_type == 'arithmetic':
            self.encoder = ArithmeticEncoder()
        elif bitstream_type == 'bzip2':
            self.encoder = Bzip2Encoder()

        # Optimizer
        self.optimizer = torch.optim.Adam(list(self.projector.parameters()) + list(self.middle_out.parameters()) + list(self.predictor.parameters()), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()

    def run(self, run, forceNoProfile=False):
        if self.slate.consume(self.config, 'profiler.enable', False) and not forceNoProfile:
            print('{PROFILER RUNNING}')
            with PyCallGraph(output=GraphvizOutput(output_file=f'./profiler/{self.name}.png')):
                self.run(run, forceNoProfile=True)
            print('{PROFILER DONE}')
            return
        
        self.train_model()

    def train_model(self):
        max_length = max([len(seq) for seq in self.train_data])
        print(f"Max sequence length: {max_length}")
        
        best_test_score = float('inf')

        for epoch in range(self.epochs):
            total_loss = 0
            random.shuffle(self.train_data)
            for i in range(0, len(self.train_data[0]) - self.input_size, self.input_size):
                batch_data = np.array([lead[i:i+self.input_size] for lead in self.train_data])
                inputs = torch.tensor(batch_data, dtype=torch.float32).unsqueeze(2).to(device)
                
                batch_loss = 0
                for lead_idx in range(len(inputs)):
                    lead_data = inputs[lead_idx]
                    latents = self.projector(lead_data)
                    
                    for t in range(latents.shape[0]):
                        my_latent = latents[t]
                        
                        peer_latents = []
                        peer_correlations = []
                        for peer_idx in np.argsort(self.correlation_matrix[lead_idx])[-self.num_peers:]:
                            peer_latent = latents[t]
                            peer_correlation = torch.tensor([self.correlation_matrix[lead_idx, peer_idx]], dtype=torch.float32).to(device)
                            peer_latents.append(peer_latent)
                            peer_correlations.append(peer_correlation)
                        
                        peer_latents = torch.stack(peer_latents).to(device)
                        peer_correlations = torch.stack(peer_correlations).to(device)
                        new_latent = self.middle_out(my_latent, peer_latents, peer_correlations)
                        prediction = self.predictor(new_latent)
                        target = lead_data[t+1] if t < latents.shape[0] - 1 else lead_data[t]
                        
                        loss = self.criterion(prediction, target)
                        batch_loss += loss.item()
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                
                total_loss += batch_loss
            
            wandb.log({"epoch": epoch, "loss": total_loss}, step=epoch)
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {total_loss}')
            
            if (epoch + 1) % self.eval_freq == 0:
                test_loss = self.evaluate_model(epoch)
                if test_loss < best_test_score:
                    best_test_score = test_loss
                    self.save_models(epoch)

    def evaluate_model(self, epoch):
        self.projector.eval()
        self.middle_out.eval()
        self.predictor.eval()

        total_loss = 0
        all_true = []
        all_predicted = []
        all_deltas = []
        compression_ratios = []
        exact_matches = 0
        total_sequences = 0

        with torch.no_grad():
            for lead_idx in range(len(self.test_data)):
                lead_data = torch.tensor(self.test_data[lead_idx], dtype=torch.float32).unsqueeze(1).to(device)
                latents = self.projector(lead_data)

                true_data = []
                predicted_data = []
                delta_data = []

                for t in range(latents.shape[0]):
                    my_latent = latents[t]

                    peer_latents = []
                    peer_correlations = []
                    for peer_idx in np.argsort(self.correlation_matrix[lead_idx])[-self.num_peers:]:
                        peer_latent = latents[t]
                        peer_correlation = torch.tensor([self.correlation_matrix[lead_idx, peer_idx]], dtype=torch.float32).to(device)
                        peer_latents.append(peer_latent)
                        peer_correlations.append(peer_correlation)

                    peer_latents = torch.stack(peer_latents).to(device)
                    peer_correlations = torch.stack(peer_correlations).to(device)
                    new_latent = self.middle_out(my_latent, peer_latents, peer_correlations)
                    prediction = self.predictor(new_latent)
                    target = lead_data[t+1] if t < latents.shape[0] - 1 else lead_data[t]

                    loss = self.criterion(prediction, target)
                    total_loss += loss.item()

                    true_data.append(target.cpu().numpy())
                    predicted_data.append(prediction.cpu().numpy())
                    delta_data.append((target - prediction).cpu().numpy())

                all_true.append(true_data)
                all_predicted.append(predicted_data)
                all_deltas.append(delta_data)

                if self.full_compression:
                    self.encoder.build_model(latents.cpu().numpy())
                    compressed_data = self.encoder.encode(latents.cpu().numpy())
                    decompressed_data = self.encoder.decode(compressed_data, len(latents))
                    compression_ratio = len(latents) / len(compressed_data)
                    compression_ratios.append(compression_ratio)

                    # Check if decompressed data matches the original data
                    if np.allclose(latents.cpu().numpy(), decompressed_data, atol=1e-5):
                        exact_matches += 1
                    total_sequences += 1

                visualize_prediction(np.array(true_data), np.array(predicted_data), np.array(delta_data), sample_rate=1, epoch=epoch)

        avg_loss = total_loss / len(self.test_data)
        print(f'Epoch {epoch+1}, Evaluation Loss: {avg_loss}')
        wandb.log({"evaluation_loss": avg_loss}, step=epoch)

        delta_plot_path = plot_delta_distribution(np.concatenate(all_deltas), epoch)
        wandb.log({"delta_distribution": wandb.Image(delta_plot_path)}, step=epoch)

        if self.full_compression:
            avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
            exact_match_percentage = (exact_matches / total_sequences) * 100
            print(f'Epoch {epoch+1}, Average Compression Ratio: {avg_compression_ratio}')
            print(f'Epoch {epoch+1}, Exact Match Percentage: {exact_match_percentage}%')
            wandb.log({"average_compression_ratio": avg_compression_ratio}, step=epoch)
            wandb.log({"exact_match_percentage": exact_match_percentage}, step=epoch)

        return avg_loss

    def save_models(self, epoch):
        torch.save(self.projector.state_dict(), os.path.join(self.save_path, f"best_projector_epoch_{epoch+1}.pt"))
        torch.save(self.middle_out.state_dict(), os.path.join(self.save_path, f"best_middle_out_epoch_{epoch+1}.pt"))
        torch.save(self.predictor.state_dict(), os.path.join(self.save_path, f"best_predictor_epoch_{epoch+1}.pt"))
        print(f"New high score! Models saved at epoch {epoch+1}.")

if __name__ == '__main__':
    slate = Slate({'spikey': SpikeRunner})
    slate.from_args()
