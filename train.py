import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import random
import os
import pickle
from data_processing import delta_encode, delta_decode, save_wav
from utils import visualize_prediction, plot_delta_distribution
from bitstream import ArithmeticEncoder

def pad_sequence(sequence, max_length):
    padded_seq = np.zeros((max_length, *sequence.shape[1:]))
    padded_seq[:sequence.shape[0], ...] = sequence
    return padded_seq

def evaluate_model(model, data, use_delta_encoding, encoder, sample_rate=19531, epoch=0):
    compression_ratios = []
    identical_count = 0
    all_deltas = []

    for i, file_data in enumerate(data):
        file_data = torch.tensor(file_data, dtype=torch.float32).unsqueeze(1).to(model.device)
        encoded_data = model.encode(file_data.squeeze(1).cpu().numpy())
        encoder.build_model(encoded_data)
        compressed_data = encoder.encode(encoded_data)
        decompressed_data = encoder.decode(compressed_data, len(encoded_data))
        
        if use_delta_encoding:
            decompressed_data = delta_decode(decompressed_data)
        
        # Ensure the lengths match
        min_length = min(len(file_data), len(decompressed_data))
        file_data = file_data[:min_length]
        decompressed_data = decompressed_data[:min_length]

        identical = np.allclose(file_data.cpu().numpy(), decompressed_data, atol=1e-5)
        if identical:
            identical_count += 1

        compression_ratio = len(file_data) / len(compressed_data)
        compression_ratios.append(compression_ratio)

        predicted_data = model(torch.tensor(encoded_data, dtype=torch.float32).unsqueeze(1).to(model.device)).squeeze(1).detach().cpu().numpy()
        if use_delta_encoding:
            predicted_data = delta_decode(predicted_data)
        
        # Ensure predicted_data is a flat list of floats
        predicted_data = predicted_data[:min_length]

        delta_data = [file_data[i].item() - predicted_data[i] for i in range(min_length)]
        all_deltas.extend(delta_data)
        
        if i == (epoch % len(data)):
            visualize_prediction(file_data.cpu().numpy(), predicted_data, delta_data, sample_rate, epoch=epoch)

    identical_percentage = (identical_count / len(data)) * 100
    delta_plot_path = plot_delta_distribution(all_deltas, epoch)
    wandb.log({"delta_distribution": wandb.Image(delta_plot_path)}, step=epoch)
    
    return compression_ratios, identical_percentage

def train_model(model, train_data, test_data, epochs, batch_size, learning_rate, use_delta_encoding, encoder, eval_freq, save_path):
    wandb.init(project="wav-compression")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_test_score = float('inf')

    model.to(model.device)

    max_length = max([len(seq) for seq in train_data])
    print(f"Max sequence length: {max_length}")

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(train_data)
        for i in range(0, len(train_data) - batch_size, batch_size):
            batch_data = [pad_sequence(np.array(train_data[j]), max_length) for j in range(i, i+batch_size)]
            batch_data = np.array(batch_data)
            inputs = torch.tensor(batch_data, dtype=torch.float32).unsqueeze(2).to(model.device)
            targets = torch.tensor(batch_data, dtype=torch.float32).unsqueeze(2).to(model.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        wandb.log({"epoch": epoch, "loss": total_loss}, step=epoch)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss}')
        
        if (epoch + 1) % eval_freq == 0:
            train_compression_ratios, train_identical_percentage = evaluate_model(model, train_data, use_delta_encoding, encoder, epoch=epoch)
            test_compression_ratios, test_identical_percentage = evaluate_model(model, test_data, use_delta_encoding, encoder, epoch=epoch)
            
            wandb.log({
                "train_compression_ratio_mean": np.mean(train_compression_ratios),
                "train_compression_ratio_std": np.std(train_compression_ratios),
                "train_compression_ratio_min": np.min(train_compression_ratios),
                "train_compression_ratio_max": np.max(train_compression_ratios),
                "test_compression_ratio_mean": np.mean(test_compression_ratios),
                "test_compression_ratio_std": np.std(test_compression_ratios),
                "test_compression_ratio_min": np.min(test_compression_ratios),
                "test_compression_ratio_max": np.max(test_compression_ratios),
                "train_identical_percentage": train_identical_percentage,
                "test_identical_percentage": test_identical_percentage,
            }, step=epoch)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Compression Ratio: Mean={np.mean(train_compression_ratios)}, Std={np.std(train_compression_ratios)}, Min={np.min(train_compression_ratios)}, Max={np.max(train_compression_ratios)}, Identical={train_identical_percentage}%')
            print(f'Epoch {epoch+1}/{epochs}, Test Compression Ratio: Mean={np.mean(test_compression_ratios)}, Std={np.std(test_compression_ratios)}, Min={np.min(test_compression_ratios)}, Max={np.max(test_compression_ratios)}, Identical={test_identical_percentage}%')
            
            test_score = np.mean(test_compression_ratios)
            if test_score < best_test_score:
                best_test_score = test_score
                model_path = os.path.join(save_path, f"best_model_epoch_{epoch+1}.pt")
                encoder_path = os.path.join(save_path, f"best_encoder_epoch_{epoch+1}.pkl")
                torch.save(model.state_dict(), model_path)
                with open(encoder_path, 'wb') as f:
                    pickle.dump(encoder, f)
                print(f'New highscore on test data! Model and encoder saved to {model_path} and {encoder_path}')
