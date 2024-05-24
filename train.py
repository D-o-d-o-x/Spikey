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

def evaluate_model(model, data, use_delta_encoding, encoder, sample_rate=19531, epoch=0, num_points=None):
    compression_ratios = []
    identical_count = 0
    all_deltas = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for file_data in data:
        file_data = torch.tensor(file_data, dtype=torch.float32).unsqueeze(1).to(device)
        encoded_data = model(file_data).squeeze(1).cpu().detach().numpy().tolist()
        encoder.build_model(encoded_data)
        compressed_data = encoder.encode(encoded_data)
        decompressed_data = encoder.decode(compressed_data, len(encoded_data))
        
        # Check equivalence
        if use_delta_encoding:
            decompressed_data = delta_decode(decompressed_data)
        identical = np.allclose(file_data.cpu().numpy(), decompressed_data, atol=1e-5)
        if identical:
            identical_count += 1

        compression_ratio = len(file_data) / len(compressed_data)
        compression_ratios.append(compression_ratio)
        
        # Compute and collect deltas
        predicted_data = model(torch.tensor(encoded_data, dtype=torch.float32).unsqueeze(1).to(device)).squeeze(1).cpu().detach().numpy().tolist()
        if use_delta_encoding:
            predicted_data = delta_decode(predicted_data)
        delta_data = [file_data[i].item() - predicted_data[i] for i in range(len(file_data))]
        all_deltas.extend(delta_data)
        
        # Visualize prediction vs data vs error
        visualize_prediction(file_data.cpu().numpy(), predicted_data, delta_data, sample_rate, num_points)

    identical_percentage = (identical_count / len(data)) * 100
    
    # Plot delta distribution
    delta_plot_path = plot_delta_distribution(all_deltas, epoch)
    wandb.log({"delta_distribution": wandb.Image(delta_plot_path)})
    
    return compression_ratios, identical_percentage

def train_model(model, train_data, test_data, epochs, batch_size, learning_rate, use_delta_encoding, encoder, eval_freq, save_path, num_points=None):
    """Train the model."""
    wandb.init(project="wav-compression")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_test_score = float('inf')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(train_data)  # Shuffle data for varied batches
        for i in range(0, len(train_data) - batch_size, batch_size):
            inputs = torch.tensor(train_data[i:i+batch_size], dtype=torch.float32).unsqueeze(2).to(device)
            targets = torch.tensor(train_data[i+1:i+batch_size+1], dtype=torch.float32).unsqueeze(2).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        wandb.log({"epoch": epoch, "loss": total_loss})
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss}')
        
        if (epoch + 1) % eval_freq == 0:
            # Evaluate on train and test data
            train_compression_ratios, train_identical_percentage = evaluate_model(model, train_data, use_delta_encoding, encoder, epoch=epoch, num_points=num_points)
            test_compression_ratios, test_identical_percentage = evaluate_model(model, test_data, use_delta_encoding, encoder, epoch=epoch, num_points=num_points)
            
            # Log statistics
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
            })
            
            print(f'Epoch {epoch+1}/{epochs}, Train Compression Ratio: Mean={np.mean(train_compression_ratios)}, Std={np.std(train_compression_ratios)}, Min={np.min(train_compression_ratios)}, Max={np.max(train_compression_ratios)}, Identical={train_identical_percentage}%')
            print(f'Epoch {epoch+1}/{epochs}, Test Compression Ratio: Mean={np.mean(test_compression_ratios)}, Std={np.std(test_compression_ratios)}, Min={np.min(test_compression_ratios)}, Max={np.max(test_compression_ratios)}, Identical={test_identical_percentage}%')
            
            # Save model and encoder if new highscore on test data
            test_score = np.mean(test_compression_ratios)
            if test_score < best_test_score:
                best_test_score = test_score
                model_path = os.path.join(save_path, f"best_model_epoch_{epoch+1}.pt")
                encoder_path = os.path.join(save_path, f"best_encoder_epoch_{epoch+1}.pkl")
                torch.save(model.state_dict(), model_path)
                with open(encoder_path, 'wb') as f:
                    pickle.dump(encoder, f)
                print(f'New highscore on test data! Model and encoder saved to {model_path} and {encoder_path}')
