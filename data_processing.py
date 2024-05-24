import os
import numpy as np
from scipy.io import wavfile
import urllib.request
import zipfile

def download_and_extract_data(url, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        zip_path = os.path.join(data_dir, 'data.zip')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

def load_wav(file_path):
    """Load WAV file and return sample rate and data."""
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

def load_all_wavs(data_dir):
    """Load all WAV files in the given directory."""
    wav_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
    all_data = []
    for file_path in wav_files:
        _, data = load_wav(file_path)
        all_data.append(data)
    return all_data

def save_wav(file_path, sample_rate, data):
    """Save data to a WAV file."""
    wavfile.write(file_path, sample_rate, np.asarray(data, dtype=np.float32))

def delta_encode(data):
    """Apply delta encoding to the data."""
    deltas = [data[0]]
    for i in range(1, len(data)):
        deltas.append(data[i] - data[i - 1])
    return deltas

def delta_decode(deltas):
    """Decode delta encoded data."""
    data = [deltas[0]]
    for i in range(1, len(deltas)):
        data.append(data[-1] + deltas[i])
    return data
