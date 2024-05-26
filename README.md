<p align='center'>
  <img src='./spikey.svg'>
</p>

# Spikey

This repository contains a solution for the [Neuralink Compression Challenge](https://content.neuralink.com/compression-challenge/README.html). The challenge involves compressing raw electrode recordings from a Neuralink implant. These recordings are taken from the motor cortex of a non-human primate while playing a video game.

## Challenge Overview

The Neuralink N1 implant generates approximately 200 Mbps of electrode data (1024 electrodes @ 20 kHz, 10-bit resolution) and can transmit data wirelessly at about 1 Mbps. This means a compression ratio of over 200x is required. The compression must run in real-time (< 1 ms) and consume low power (< 10 mW, including radio).

## Data Analysis

The `analysis.ipynb` notebook contains a detailed analysis of the data. We found that there is sometimes significant cross-correlation between the different leads, so we find it vital to use this information for better compression. This cross-correlation allows us to improve the accuracy of our predictions and reduce the overall amount of data that needs to be transmitted. As part of the analysis, we also note that achieving a 200x compression ratio is highly unlikely to be possible and is also nonsensical; a very close reproduction is sufficient.

## Algorithm Overview

### 1 - Thread Topology Reconstruction

As the first step, we analyze readings from the leads to construct an approximate topology of the threads in the brain. The distance metric we generate only approximately represents true Euclidean distances, but rather the 'distance' in common activity. This topology must only be computed once for a given implant and may be updated for thread movements but is not part of the regular compression/decompression process.

### 2 - Predictive Architecture

The main workhorse of our compression approach is a predictive model running both in the compressor and decompressor. With good predictions of the data, only the error between the prediction and actual data must be transmitted. We make use of the previously constructed topology to allow the predictive model's latent to represent the activity of brain regions based on the reading of the threads instead of just for threads themselves.

We seperate the predictive model into three parts:

1. **Latent Projector**: This module takes in a segment of a lead and projects it into a latent space. The latent projector can be configured as a fully connected network or an RNN (LSTM) with an arbitrary shape.

2. **MiddleOut (Message Passer)**: For each lead, this module performs message passing according to the thread topology. Their latent representations along with their distance metrics are used to generate region latent representation. This is done by training a fully connected layer to map from (our_latent, their_latent, metric) -> region_latent and then averaging over all region_latent values to get the final representation.

3. **Predictor**: This module takes the new latent representation from the MiddleOut module and predicts the next timestep. The goal is to minimize the prediction error during training. It can be configured to be an FCNN of arbitrary shape.

The neural networks used are rather small, making it possible to meet the latency and power requirements if implemented more efficiently.

If we were to give up on lossless compression, one could expand MiddleOut to form a joint latent over all threads and transmit that.

### 3 - Efficient Bitstream Encoding

Based on an expected distribution of deltas that have to be transmitted, an efficient Huffman-like binary format is used for encoding the data.

## TODO

- Our flagship bitstream encoder builds an optimal huffman tree assuming the deltas are binomially distributed. Should be updated when we know a more precise approx of the delta dist.
- All trained models stick mostly suck. Im not beating a compression ratio of ~2x (not counting bitstream encoder). Probably a bug somewhere in our code?

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

```calibash
python main.py <config_file.yaml> <exp_name>
```
