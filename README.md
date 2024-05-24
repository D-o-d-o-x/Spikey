# Spikey

This repository contains a solution for the [Neuralink Compression Challenge](https://content.neuralink.com/compression-challenge/README.html). The challenge involves compressing raw electrode recordings from a Neuralink implant. These recordings are taken from the motor cortex of a non-human primate while playing a video game.

## Challenge Overview

The Neuralink N1 implant generates approximately 200Mbps of electrode data (1024 electrodes @ 20kHz, 10-bit resolution) and can transmit data wirelessly at about 1Mbps. This means a compression ratio of over 200x is required. The compression must run in real-time (< 1ms) and consume low power (< 10mW, including radio).

## Installation

To install the necessary dependencies, create a virtual environment and install the requirements:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

### Configuration

The configuration for training and evaluation is specified in a YAML file. Below is an example configuration:

```yaml
name: Test

preprocessing:
  use_delta_encoding: true # Whether to use delta encoding.

predictor:
  type: lstm # Options: 'lstm', 'fixed_input_nn'
  input_size: 1 # Input size for the LSTM predictor.
  hidden_size: 128 # Hidden size for the LSTM or Fixed Input NN predictor.
  num_layers: 2 # Number of layers for the LSTM predictor.
  fixed_input_size: 10 # Input size for the Fixed Input NN predictor. Only used if type is 'fixed_input_nn'.

training:
  epochs: 10 # Number of training epochs.
  batch_size: 32 # Batch size for training.
  learning_rate: 0.001 # Learning rate for the optimizer.
  eval_freq: 2 # Frequency of evaluation during training (in epochs).
  save_path: models # Directory to save the best model and encoder.
  num_points: 1000 # Number of data points to visualize.

bitstream_encoding:
  type: arithmetic # Use arithmetic encoding.

data:
  url: https://content.neuralink.com/compression-challenge/data.zip # URL to download the dataset.
  directory: data # Directory to extract and store the dataset.
  split_ratio: 0.8 # Ratio to split the data into train and test sets.
```

### Running the Code

To train the model and compress/decompress WAV files, use the CLI provided:

```bash
python cli.py compress --config config.yaml --input_file path/to/input.wav --output_file path/to/output.bin
python cli.py decompress --config config.yaml --input_file path/to/output.bin --output_file path/to/output.wav
```

### Training

Requires Slate, which is not currently publicaly avaible. Install via (requires repo access)

```bash
pip install -e git+ssh://git@dominik-roth.eu/dodox/Slate.git#egg=slate
```

To train the model, run:

```bash
python main.py config.yaml Test
```
