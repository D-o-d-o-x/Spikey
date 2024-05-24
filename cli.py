import argparse
import yaml
import os
import torch
from data_processing import download_and_extract_data, load_all_wavs, save_wav, delta_encode, delta_decode
from main import SpikeRunner

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description="WAV Compression with Neural Networks")
    parser.add_argument('action', choices=['compress', 'decompress'], help="Action to perform")
    parser.add_argument('--config', default='config.yaml', help="Path to the configuration file")
    parser.add_argument('--input_file', help="Path to the input WAV file")
    parser.add_argument('--output_file', help="Path to the output file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    spike_runner = SpikeRunner(None, config)
    spike_runner.setup('CLI')

    if args.action == 'compress':
        data = load_all_wavs(args.input_file)
        if spike_runner.slate.consume(config['preprocessing'], 'use_delta_encoding'):
            data = [delta_encode(d) for d in data]
        
        spike_runner.encoder.build_model(data)
        encoded_data = [spike_runner.model(torch.tensor(d, dtype=torch.float32).unsqueeze(0)).squeeze(0).detach().numpy().tolist() for d in data]
        compressed_data = [spike_runner.encoder.encode(ed) for ed in encoded_data]
        
        with open(args.output_file, 'wb') as f:
            for cd in compressed_data:
                f.write(bytearray(cd))
    
    elif args.action == 'decompress':
        with open(args.input_file, 'rb') as f:
            compressed_data = list(f.read())
        
        decoded_data = [spike_runner.encoder.decode(cd, len(cd)) for cd in compressed_data]
        if spike_runner.slate.consume(config['preprocessing'], 'use_delta_encoding'):
            decoded_data = [delta_decode(dd) for dd in decoded_data]
        
        save_wav(args.output_file, 19531, decoded_data)  # Assuming 19531 Hz sample rate

if __name__ == "__main__":
    main()
