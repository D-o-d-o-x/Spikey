import numpy as np
from scipy.io import wavfile
import urllib.request
import zipfile
import os

def download_and_extract_data(url):
    if not os.path.exists('data'):
        zip_path = os.path.join('.', 'data.zip')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(zip_path)

def load_wav(file_path):
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

def load_all_wavs(data_dir, cut_length=None):
    wav_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
    all_data = []
    for file_path in wav_files:
        _, data = load_wav(file_path)
        if cut_length is not None:
            print(cut_length)
            data = data[:cut_length]
        all_data.append(data)
    return all_data

def compute_topology_metrics(data):
    num_leads = len(data)
    min_length = min(len(d) for d in data)
    
    # Trim all leads to the minimum length
    trimmed_data = [d[:min_length] for d in data]

    metric_matrix = np.corrcoef(trimmed_data)
    np.fill_diagonal(metric_matrix, 0)
    return np.abs(metric_matrix)

def split_data_by_time(data, split_ratio=0.5):
    train_data = []
    test_data = []
    for lead in data:
        split_idx = int(len(lead) * split_ratio)
        train_data.append(lead[:split_idx])
        test_data.append(lead[split_idx:])
    return train_data, test_data
