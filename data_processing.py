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

def save_wav(file_path, data, sample_rate=19531):
    wavfile.write(file_path, sample_rate, data)

def save_all_wavs(output_dir, all_data, input_filenames):
    for data, filename in zip(all_data, input_filenames):
        output_file_path = os.path.join(output_dir, filename)
        save_wav(output_file_path, data)

def compute_topology_metrics(data):
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

def unfuckify(nums):
    return np.round((nums + 33) / 64).astype(int)

def unfuckify_all(wavs):
    return [unfuckify(wav) for wav in wavs]

# The released dataset is 10bit resolution encoded in a 16bit range with a completely fucked up mapping, which we have to replicate for lossless fml
# This func works for all samples contained in the provided dataset, but I don't guarentee it works for all possible data
# The solution would be to just never fuck up the data (operate on the true 10bit values)
def refuckify(nums):
    n = np.round((nums * 64) - 32).astype(int)
    n[n >= 32] -= 1
    n[n >= 160] -= 1
    n[n >= 222] -= -1

    for i in [543, 1568, 2657, 3682, 4707, 5732, 6821, 7846, 8871, 9896, 10921, 12010, 13035, 14060, 15085, 16174, 17199, 18224, 19249, 20338, 21363, 22388, 23413, 24502, 25527, 26552, 27577, 28666, 29691, 30716, 31741]:
        n[n >= i] -= -1
        n[n <= -(i+1)] -= 1

    n[n <= -32742] -= 3
    n[n <= -32770] -= -2
    n[n <= -32832] -= -65599
    
    return n

def refuckify_all(wavs):
    return [refuckify(wav) for wav in wavs]