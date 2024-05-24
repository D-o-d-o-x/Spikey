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
