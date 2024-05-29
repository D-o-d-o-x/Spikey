import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
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

def visualize_prediction_grid(true_data, predicted_data, delta_data, num_points=None, epoch=None):
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
    return file_path

def visualize_prediction(true_data, predicted_data, delta_data, steps_data, num_points=None, epoch=None):
    """Visualize the combined plot of true data, predicted data, and deltas."""
    if num_points:
        true_data = true_data[:num_points]
        predicted_data = predicted_data[:num_points]
        delta_data = delta_data[:num_points]
        steps_data = steps_data[:num_points]

    plt.figure(figsize=(20, 10))
    
    plt.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Predicted Data', color='orange')
    plt.plot(delta_data, label='Delta', color='red')
    plt.plot(steps_data, label='Naive', color='darkred', linestyle='--')

    # Add horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    plt.title('Combined Data')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    tmp_dir = os.getenv('TMPDIR', '/tmp')
    file_path = os.path.join(tmp_dir, f'prediction_plot_{np.random.randint(1e6)}.png')
    plt.savefig(file_path)
    plt.close()
    return file_path


def plot_delta_distribution(deltas, epoch):
    mu, std = 0, np.std(deltas)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    count, bins, ignored = plt.hist(deltas, bins=min(100, np.max(deltas) - np.min(deltas)), density=True, alpha=0.6, color='g')
    
    # Plot Gaussian curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
        
    # Add title and labels
    plt.title(f'Delta Distribution at Epoch {epoch}')
    plt.xlabel('Delta')
    plt.ylabel('Density')
    plt.grid(True)
    
    # Save the plot
    tmp_dir = os.getenv('TMPDIR', '/tmp')
    file_path = os.path.join(tmp_dir, f'delta_distribution_epoch_{epoch}_{np.random.randint(1e6)}.png')
    plt.savefig(file_path)
    plt.close()
    
    return file_path