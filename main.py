import os
import torch
import torch.nn as nn
import numpy as np
import random, math
from utils import visualize_prediction, plot_delta_distribution
from data_processing import download_and_extract_data, load_all_wavs, split_data_by_time, compute_topology_metrics, unfuckify_all, refuckify
from models import LatentFCProjector, LatentRNNProjector, MiddleOut, Predictor, FeatureExtractor
from bitstream import IdentityEncoder, ArithmeticEncoder, Bzip2Encoder, BinomialHuffmanEncoder, RiceEncoder
import wandb
from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
from slate import Slate, Slate_Runner


class SpikeRunner(Slate_Runner):
    def setup(self, name):
        print("Setup SpikeRunner")

        self.name = name
        slate, config = self.slate, self.config

        training_config = slate.consume(config, 'training', expand=True)
        data_config = slate.consume(config, 'data', expand=True)

        data_url = slate.consume(data_config, 'url')
        cut_length = slate.consume(data_config, 'cut_length', None)
        download_and_extract_data(data_url)
        self.all_data = load_all_wavs('data', cut_length)

        split_ratio = slate.consume(data_config, 'split_ratio', 0.5)
        self.train_data, self.test_data = split_data_by_time(unfuckify_all(self.all_data), split_ratio)

        print("Reconstructing thread topology")
        self.topology_matrix = compute_topology_metrics(self.train_data)

        # Number of peers for message passing
        self.num_peers = slate.consume(config, 'middle_out.num_peers')

        # Precompute sorted indices for the top num_peers correlated leads
        print("Precomputing sorted peer indices")
        self.sorted_peer_indices = np.argsort(-self.topology_matrix, axis=1)[:, :self.num_peers]

        # Model setup
        print("Setting up models")
        latent_projector_type = slate.consume(config, 'latent_projector.type', default='fc')
        latent_size = slate.consume(config, 'latent_projector.latent_size')
        input_size = slate.consume(config, 'feature_extractor.input_size')
        region_latent_size = slate.consume(config, 'middle_out.region_latent_size')
        self.delta_shift = slate.consume(config, 'predictor.delta_shift', True)
        device = slate.consume(training_config, 'device', 'auto')
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.feat = FeatureExtractor(input_size=input_size, **slate.consume(config, 'feature_extractor', expand=True)).to(device)
        feature_size = self.feat.compute_output_size()

        if latent_projector_type == 'fc':
            self.projector = LatentFCProjector(latent_size=latent_size, feature_size=feature_size, **slate.consume(config, 'latent_projector', expand=True)).to(device)
        elif latent_projector_type == 'rnn':
            self.projector = LatentRNNProjector(latent_size=latent_size, feature_size=feature_size, **slate.consume(config, 'latent_projector', expand=True)).to(device)
        else:
            raise Exception('No such Latent Projector')

        self.middle_out = MiddleOut(latent_size=latent_size, region_latent_size=region_latent_size, num_peers=self.num_peers, **slate.consume(config, 'middle_out', expand=True)).to(device)
        self.predictor = Predictor(region_latent_size=region_latent_size, **slate.consume(config, 'predictor', expand=True)).to(device)

        # Training parameters
        self.input_size = input_size
        self.epochs = slate.consume(training_config, 'epochs')
        self.batch_size = slate.consume(training_config, 'batch_size')
        self.num_batches = slate.consume(training_config, 'num_batches')
        self.learning_rate = slate.consume(training_config, 'learning_rate')
        self.eval_freq = slate.consume(training_config, 'eval_freq', -1)
        self.save_path = slate.consume(training_config, 'save_path')
        self.peer_gradients_factor = float(slate.consume(training_config, 'peer_gradients_factor', 1.0))
        self.value_scale = slate.consume(training_config, 'value_scale', 1.0)

        # Evaluation parameter
        self.full_compression = slate.consume(config, 'evaluation.full_compression', default=False)

        # Bitstream encoding
        bitstream_type = slate.consume(config, 'bitstream_encoding.type', default='rice')
        if bitstream_type == 'identity':
            self.encoder = IdentityEncoder()
        elif bitstream_type == 'arithmetic':
            self.encoder = ArithmeticEncoder()
        elif bitstream_type == 'bzip2':
            self.encoder = Bzip2Encoder()
        elif bitstream_type == 'binomHuffman':
            self.encoder = BinomialHuffmanEncoder()
        elif bitstream_type == 'rice':
            self.encoder = RiceEncoder()
        else:
            raise Exception('No such Encoder')
        self.bitstream_encoder_config = slate.consume(config, 'bitstream_encoding')

        # Optimizer
        self.optimizer = torch.optim.Adam(list(self.projector.parameters()) + list(self.middle_out.parameters()) + list(self.predictor.parameters()), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()
        print("SpikeRunner initialization complete")

    def run(self, run, forceNoProfile=False):
        if self.slate.consume(self.config, 'profiler.enable', False) and not forceNoProfile:
            print('{PROFILER RUNNING}')
            with PyCallGraph(output=GraphvizOutput(output_file=f'./profiler/{self.name}.png')):
                self.run(run, forceNoProfile=True)
            print('{PROFILER DONE}')
            return
        
        self.train_model()

    def train_model(self):
        device = self.device
        min_length = min([len(seq) for seq in self.train_data])

        best_test_score = float('inf')

        for epoch in range(self.epochs):
            total_loss = 0
            errs = []
            rels = []
            derrs = []
            for batch_num in range(self.num_batches):

                # Create indices for training data and shuffle them
                indices = list(range(len(self.train_data)))
                random.shuffle(indices)

                stacked_segments = []
                peer_metrics = []
                targets = []
                lasts = []

                for idx in indices[:self.batch_size]:
                    lead_data = self.train_data[idx][:min_length]

                    # Slide a window over the data with overlap
                    stride = max(1, self.input_size // 3)  # Ensuring stride is at least 1
                    offset = np.random.randint(0, stride)
                    for i in range(offset, len(lead_data) - self.input_size-1-offset, stride):
                        lead_segment = lead_data[i:i + self.input_size]
                        inputs = torch.tensor(lead_segment, dtype=torch.float32).to(device)

                        # Collect the segments for the current lead and its peers
                        peer_segments = []
                        for peer_idx in self.sorted_peer_indices[idx]:
                            peer_segment = self.train_data[peer_idx][i:i + self.input_size]
                            peer_segments.append(torch.tensor(peer_segment, dtype=torch.float32).to(device))
                        peer_metric = torch.tensor([self.topology_matrix[idx, peer_idx] for peer_idx in self.sorted_peer_indices[idx]], dtype=torch.float32).to(device)
                        peer_metrics.append(peer_metric)

                        # Stack the segments to form the batch
                        stacked_segment = torch.stack([inputs] + peer_segments).to(device)
                        stacked_segments.append(stacked_segment)
                        target = lead_data[i + self.input_size]
                        targets.append(target)
                        last = lead_data[i + self.input_size - 1]
                        lasts.append(last)

                las = torch.tensor(lasts, dtype=torch.float32).unsqueeze(-1).to(device)

                inp = torch.stack(stacked_segments) / self.value_scale
                feat = self.feat(inp)
                latents = self.projector(feat)

                my_latent = latents[:, 0, :]
                peer_latents = latents[:, 1:, :]

                # Scale gradients during backwards pass as configured
                if self.peer_gradients_factor == 1.0:
                    pass
                elif self.peer_gradients_factor == 0.0:
                    peer_latents = peer_latents.detach()
                else:
                    peer_latents.register_hook(lambda grad: grad*self.peer_gradients_factor)

                # Pass through MiddleOut
                region_latent = self.middle_out(my_latent, peer_latents, torch.stack(peer_metrics))
                prediction = self.predictor(region_latent)*self.value_scale

                if self.delta_shift:
                    prediction = prediction + las

                # Calculate loss and backpropagate
                tar = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1).to(device)
                loss = self.criterion(prediction, tar)
                err = np.sum(np.abs(prediction.cpu().detach().numpy() - tar.cpu().detach().numpy()))
                derr = np.sum(np.abs(las.cpu().detach().numpy() - tar.cpu().detach().numpy()))
                rel = err / np.sum(tar.cpu().detach().numpy())
                total_loss += loss.item()
                derrs.append(derr/np.prod(tar.size()).item())
                errs.append(err/np.prod(tar.size()).item())
                rels.append(rel.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            tot_err = sum(errs)/len(errs)
            tot_derr = sum(derrs)/len(derrs)
            adv_delta = tot_derr / tot_err
            approx_ratio = 1/(sum(rels)/len(rels))
            wandb.log({"train/epoch": epoch, "train/loss": total_loss, "train/err": tot_err, "train/approx_ratio": approx_ratio, "train/adv_delta": adv_delta}, step=epoch)
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss}')

            if self.eval_freq != -1 and (epoch + 1) % self.eval_freq == 0:
                print(f'Starting evaluation for epoch {epoch + 1}')
                test_loss = self.evaluate_model(epoch)
                if test_loss < best_test_score:
                    best_test_score = test_loss
                    self.save_models(epoch)
                print(f'Evaluation complete for epoch {epoch + 1}')

    def evaluate_model(self, epoch):
        print('Evaluating model...')
        device = self.device

        # Save the current mode of the models
        projector_mode = self.projector.training
        middle_out_mode = self.middle_out.training
        predictor_mode = self.predictor.training

        # Set models to evaluation mode
        self.projector.eval()
        self.middle_out.eval()
        self.predictor.eval()

        total_loss = 0
        all_true = []
        all_predicted = []
        all_deltas = []
        all_steps = []

        with torch.no_grad():
            min_length = min([len(seq) for seq in self.test_data])

            errs = []
            rels = []
            derrs = []

            indices = list(range(len(self.test_data)))
            random.shuffle(indices)

            for lead_idx in indices[:16]:
                lead_data = self.test_data[lead_idx][:min_length]

                stacked_segments = []
                peer_metrics = []
                targets = []
                lasts = []

                for i in range(0, len(lead_data) - self.input_size - 1, self.input_size // 8):
                    lead_segment = lead_data[i:i + self.input_size]
                    inputs = torch.tensor(lead_segment, dtype=torch.float32).to(device)

                    peer_segments = []
                    for peer_idx in self.sorted_peer_indices[lead_idx]:
                        peer_segment = self.test_data[peer_idx][:min_length][i:i + self.input_size]
                        peer_segments.append(torch.tensor(peer_segment, dtype=torch.float32).to(device))
                    peer_metric = torch.tensor([self.topology_matrix[lead_idx, peer_idx] for peer_idx in self.sorted_peer_indices[lead_idx]], dtype=torch.float32).to(device)
                    peer_metrics.append(peer_metric)

                    stacked_segment = torch.stack([inputs] + peer_segments).to(device)
                    stacked_segments.append(stacked_segment)
                    target = lead_data[i + self.input_size]
                    targets.append(target)
                    last = lead_data[i + self.input_size - 1]
                    lasts.append(last)

                las = torch.tensor(lasts, dtype=torch.float32).unsqueeze(-1).to(device)

                inp = torch.stack(stacked_segments) / self.value_scale
                feat = self.feat(inp)
                latents = self.projector(feat)

                my_latent = latents[:, 0, :]
                peer_latents = latents[:, 1:, :]

                region_latent = self.middle_out(my_latent, peer_latents, torch.stack(peer_metrics))
                prediction = self.predictor(region_latent) * self.value_scale

                if self.delta_shift:
                    prediction = prediction + las

                tar = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1).to(device)
                loss = self.criterion(prediction, tar)
                delta = prediction.cpu().detach().numpy() - tar.cpu().detach().numpy()
                err = np.sum(np.abs(delta))
                derr = np.sum(np.abs(las.cpu().detach().numpy() - tar.cpu().detach().numpy()))
                step = las.cpu().detach().numpy() - tar.cpu().detach().numpy()
                rel = err / np.sum(tar.cpu().detach().numpy())
                total_loss += loss.item()
                derrs.append(derr / np.prod(tar.size()).item())
                errs.append(err / np.prod(tar.size()).item())
                rels.append(rel.item())

                all_true.extend(tar.cpu().numpy())
                all_predicted.extend(prediction.cpu().numpy())
                all_deltas.extend(delta.tolist())
                all_steps.extend(step.tolist())

        if self.full_compression:
            self.encoder.build_model(delta_samples=delta, **self.bitstream_encoder_config)
            raw_l = len(refuckify(np.concatenate(all_true)).astype(np.int16))*16
            comp_l = len(self.encoder.encode(np.concatenate(all_deltas)))
            ratio = raw_l / comp_l
            wandb.log({"eval/ratio": ratio}, step=epoch)

        avg_loss = total_loss / len(self.test_data)
        tot_err = sum(errs) / len(errs)
        tot_derr = sum(derrs) / len(derrs)
        adv_delta = tot_derr / tot_err
        approx_ratio = 1 / (sum(rels) / len(rels))

        print(f'Epoch {epoch+1}, Evaluation Loss: {avg_loss}')
        wandb.log({"eval/loss": avg_loss, "eval/err": tot_err, "eval/approx_ratio": approx_ratio, "eval/adv_delta": adv_delta}, step=epoch)

        # Visualize predictions
        #visualize_prediction(all_true, all_predicted, all_deltas, epoch=epoch, num_points=1953, name='0.1s')
        img = visualize_prediction(all_true, all_predicted, all_deltas, all_steps, epoch=epoch, num_points=195)
        try:
            wandb.log({f"Prediction vs True Data 0.01s": wandb.Image(img)}, step=epoch)
        except:
            pass
        #visualize_prediction(all_true, all_predicted, all_deltas, epoch=epoch, num_points=20, name='0.001s')

        # Plot delta distribution
        delta_plot_path = plot_delta_distribution(np.array(all_deltas), epoch)
        try:
            wandb.log({"delta_distribution": wandb.Image(delta_plot_path)}, step=epoch)
        except:
            pass

        #if self.full_compression:
        #    avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
        #    exact_match_percentage = (exact_matches / total_sequences) * 100
        #    print(f'Epoch {epoch+1}, Average Compression Ratio: {avg_compression_ratio}')
        #    print(f'Epoch {epoch+1}, Exact Match Percentage: {exact_match_percentage}%')
        #    wandb.log({"average_compression_ratio": avg_compression_ratio}, step=epoch)
        #    wandb.log({"exact_match_percentage": exact_match_percentage}, step=epoch)

        # Restore the original mode of the models
        if projector_mode:
            self.projector.train()
        else:
            self.projector.eval()

        if middle_out_mode:
            self.middle_out.train()
        else:
            self.middle_out.eval()

        if predictor_mode:
            self.predictor.train()
        else:
            self.predictor.eval()

        print('Evaluation done for this epoch.')
        return avg_loss

    def save_models(self, epoch):
        return
        print('Saving models...')
        torch.save(self.projector.state_dict(), os.path.join(self.save_path, f"best_projector_epoch_{epoch+1}.pt"))
        torch.save(self.middle_out.state_dict(), os.path.join(self.save_path, f"best_middle_out_epoch_{epoch+1}.pt"))
        torch.save(self.predictor.state_dict(), os.path.join(self.save_path, f"best_predictor_epoch_{epoch+1}.pt"))
        print(f"New high score! Models saved at epoch {epoch+1}.")

    def compress(raw):
        threads = unfuckify_all(raw)
        for thread in threads:
            pass
            # 1. featExtr
            # 2. latentProj
            # 3. middleOut
            # 4. predictor
            # 5. calc delta
            # 6. encode
        # 7. return

if __name__ == '__main__':
    print('Initializing...')
    slate = Slate({'spikey': SpikeRunner})
    slate.from_args()
    print('Done.')