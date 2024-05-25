# Spikey

This repository contains a solution for the [Neuralink Compression Challenge](https://content.neuralink.com/compression-challenge/README.html). The challenge involves compressing raw electrode recordings from a Neuralink implant. These recordings are taken from the motor cortex of a non-human primate while playing a video game.

## Challenge Overview

The Neuralink N1 implant generates approximately 200Mbps of electrode data (1024 electrodes @ 20kHz, 10-bit resolution) and can transmit data wirelessly at about 1Mbps. This means a compression ratio of over 200x is required. The compression must run in real-time (< 1ms) and consume low power (< 10mW, including radio).

## Data Analysis

The `analysis.ipynb` notebook contains a detailed analysis of the data. We found that there is sometimes significant cross-correlation between the different leads, so we find it vital to use this information for better compression. This cross-correlation allows us to improve the accuracy of our predictions and reduce the overall amount of data that needs to be transmitted. As part of the analysis, we also note that achieving a 200x compression ratio is highly unlikely to be possible and is also nonsensical, a very close reproduction is sufficient.

## Compression Details

The solution leverages three neural network models to achieve effective compression:

1. **Latent Projector**: This module takes in a segment of a lead and projects it into a latent space. The latent projector can be configured as a fully connected network or an RNN (LSTM) based on the configuration.

2. **MiddleOut (Message Passer)**: For each lead, this module looks up the `n` most correlated leads and uses their latent representations along with their correlation values to generate a new latent representation. This is done by training a fully connected layer to map from (our_latent, their_latent, correlation) -> new_latent and then averaging over all new_latent values to get the final representation.

3. **Predictor**: This module takes the new latent representation from the MiddleOut module and predicts the next timestep. The goal is to minimize the prediction error during training.

By accurately predicting the next timestep, the delta (difference) between the actual value and the predicted value is minimized. Small deltas mean that fewer bits are needed to store these values, which are then efficiently encoded using the bitstream encoder.

The neural networks used in this solution are tiny, making it possible to meet the latency and power requirements if implemented more efficiently.

## Installation

To install the necessary dependencies, create a virtual environment and install the requirements:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

### Training

Requires Slate, which is not currently publicly available. Install via (requires repo access):

```bash
pip install -e git+ssh://git@dominik-roth.eu/dodox/Slate.git#egg=slate
```

To train the model, run:

```bash
python main.py <config_file.yaml> <exp_name>
```

