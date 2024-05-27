import matplotlib.pyplot as plt
import numpy as np
import wandb
import os

def visualize_wav_data(sample_rate, data, title="WAV Data", num_points=None):
    """Visualize WAV data using matplotlib."""
    if num_points:
        data = data[:num_points]
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(data) / sample_rate, num=len(data)), data)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()

def visualize_prediction(true_data, predicted_data, delta_data, num_points=None, epoch=None, name=''):
    """Visualize the true data, predicted data, deltas, and combined plot."""
    if num_points:
        true_data = true_data[:num_points]
        predicted_data = predicted_data[:num_points]
        delta_data = delta_data[:num_points]

    plt.figure(figsize=(20, 5))

    plt.subplot(2, 2, 1)
    plt.plot(true_data, label='True Data')
    plt.title('True Data')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 3)
    plt.plot(predicted_data, label='Predicted Data', color='orange')
    plt.title('Predicted Data')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 4)
    plt.plot(delta_data, label='Delta', color='red')
    plt.title('Delta')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 2)
    plt.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Predicted Data', color='orange')
    plt.plot(delta_data, label='Delta', color='red')
    plt.title('Combined Data')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    tmp_dir = os.getenv('TMPDIR', '/tmp')
    file_path = os.path.join(tmp_dir, f'prediction_plot_{np.random.randint(1e6)}.png')
    plt.savefig(file_path)
    plt.close()
    wandb.log({f"Prediction vs True Data {name}": wandb.Image(file_path)}, step=epoch)


def plot_delta_distribution(deltas, epoch):
    """Plot the distribution of deltas."""
    plt.figure(figsize=(10, 6))
    plt.hist(deltas, bins=100, density=True, alpha=0.6, color='g')
    plt.title(f'Delta Distribution at Epoch {epoch}')
    plt.xlabel('Delta')
    plt.ylabel('Density')
    plt.grid(True)
    tmp_dir = os.getenv('TMPDIR', '/tmp')
    file_path = os.path.join(tmp_dir, f'delta_distribution_epoch_{epoch}_{np.random.randint(1e6)}.png')
    plt.savefig(file_path)
    plt.close()
    return file_path
