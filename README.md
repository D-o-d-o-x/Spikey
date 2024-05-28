<p align='center'>
  <img src='./spikey.svg'>
</p>

# Spikey

This repository contains a solution for the [Neuralink Compression Challenge](https://content.neuralink.com/compression-challenge/README.html). The challenge involves compressing raw electrode recordings from a Neuralink implant. These recordings are taken from the motor cortex of a non-human primate while playing a video game.

**TL;DR;** We achieve a lossless compression ratio of **4.445** using a predictive model that employs discrete Meyer wavelet convolution for signal decomposition, inter-thread message passing to account for underlying brain region activity, and Rice coding for efficient entropy encoding. We believe this to be close to the optimum achievable with lossless compression and argue against pursuing lossless compression as a further goal.

## Challenge Overview

The Neuralink N1 implant generates approximately 200 Mbps of electrode data (1024 electrodes @ 20 kHz, 10-bit resolution) and can transmit data wirelessly at about 1 Mbps. This means a compression ratio of over 200x is required. The compression must run in real-time (< 1 ms) and consume low power (< 10 mW, including radio).

## Data Analysis

The `analysis.ipynb` notebook contains a detailed analysis of the data. We found that there is sometimes significant cross-correlation between the different threads, so we find it vital to use this information for better compression. This cross-correlation allows us to improve the accuracy of our predictions and reduce the overall amount of data that needs to be transmitted. We sometimes even observe certain 'structures' on multiple threads, but shifted a couple steps in time (might that not be handy for compression?).

## Algorithm Overview

### 1 - Thread Topology Reconstruction

As the first step, we analyze readings from the threads to construct an approximate topology of the threads in the brain. The distance metric we generate only approximately represents true Euclidean distances, but rather the 'distance' in common activity. This topology must only be computed once for a given implant and may be updated for thread movements but is not part of the regular compression/decompression process.

### 2 - Predictive Model

The main workhorse of our compression approach is a predictive model running both in the compressor and decompressor. With good predictions of the data, only the error between the prediction and actual data must be transmitted. We make use of the previously constructed topology to allow the predictive model's latent to represent the activity of brain regions based on the reading of the threads instead of just for threads themselves.

We separate the predictive model into four parts:

1. **Feature Extraction**: This module processes a given history of readings for a single thread and extracts relevant features (using mostly wavelet and Fourier transforms). Highly configurable, this module performs the heavy lifting of signal analysis, allowing shallow neural networks to handle the rest. (Full disclosure: I have no idea what half of the implemented wavelet transforms actually do. We just throw anything sensible at the problem and will narrow down later; making effective use of the fact that 'fuck around' and 'find out' are positively correlated.)

2. **Latent Projector**: This takes the feature vectors and projects them into a latent space. The latent projector can be configured as a fully connected network or an RNN (LSTM) with an arbitrary shape.

3. **[MiddleOut](https://www.youtube.com/watch?v=l49MHwooaVQ)**: For each thread, this module performs message passing according to the thread topology. Their latent representations along with their distance metrics are used to generate region latent representations. This is done by training a fully connected layer to map from (our_latent, their_latent, metric) -> region_latent and then averaging over all region_latent values to get the final representation.

4. **Predictor**: This module takes the new latent representation from the MiddleOut module and predicts the next timestep. The goal is to minimize the prediction error during training. It can be configured to be an FCNN of arbitrary shape.

The neural networks used are rather small, making it possible to meet the latency and power requirements if implemented more efficiently. (Some of the available feature extractors are somewhat expensive thought).

If we were to give up on lossless compression, one could expand MiddleOut to form a joint latent over all threads and transmit that.

### 3 - Efficient Bitstream Encoding

The best performing available bitstream encoder is Rice, but we also provide a prebuild Hoffman based on a binomial prior and some others.

Check the `config.yaml` for a bit more info on these.

## Discussion

### On Lossless 200x Compression

Expecting a 200x compression ratio is ludicrous, as it would mean transmitting only 1 bit per 20 data points. Given the high entropy of the readings, this is an absurd goal. Anyone who thinks lossless 200x compression is remotely feasible has a woefully inadequate grasp of information theory. Please, do yourself a favor and read Shannon’s paper.

Furthermore, there's no need for lossless compression. These readings feed into an ML model to extract intent, and any such encoder inherently reduces information content with each layer ('intelligence is the ability to disregard irrelevant information'). Instead, compression should be regarded as an integral part of the ML pipeline for intent extraction. It should be allowed to be lossy, with the key being to define the loss metric not by information loss in the input space, but rather in the latent space of the pipeline.

Let's see how far we can get with the approach presented here...

### On fucked up wav files

Why is the dataset provided not 10-bit if the readings are? They are all 16-bit. And the last 6 bits are not all zeros. We know they can't encode sensible information when the readings are only 10-bit, but we also can't just throw them away since they do contain something. We also observe that all possible values the data points can take on are separated by 64 or 63 (64 would make sense; 63 very much does not). (See `fucked_up_wavs.py`)

### On Evaluation

The provided eval.sh script is also somewhat flawed (as in: not aligned with what should be optimized for), since it counts the size of the compressor and decompressor as part of the transmitted data. Especially the decompressor part makes no sense. It also makes it impossible to compress data from multiple threads together, which is required for the free lunch we can get from topological reconstruction.

## Preliminary Results
Current best: **4.445** (not counting encoder / decoder size, just data)

Theoretical max via Shannon: [3.439](https://x.com/usrbinishan/status/1794948522112151841), best found online: [3.35](https://github.com/phoboslab/neuralink_brainwire). (Shannon assumptions don't hold for this algo, so max does not apply)
Config Outline: Meyer Wavelets for feature extraction (are great at recognizing spikes). Rice as bitstream encoder with k=2. 8D Latents. Residual skip-con in MiddleOut. (See `Proto_2_k2` in `config.yaml`)

That result is actually impressive as fuck (maybe not if one expects 200x). Our predictor works amazingly well. The decompressor is not yet fully implemented so I'm unable to ensure there are no bugs eating information. I'm also currently ignoring compression ratios for the first 0.1s of data, since the NNs window is not yet filled then. Need to either train the NNs to be able to handle that or use a more naive compression method for that timeframe. Also, this is only on 20% of dataset (test set to prevent overfitting on training set).

## TODO

- Tune the BinomialHuffman, maybe we can beat rice as the bistream encoder?

- Tune HPs / Benchmark networks

- cli for compress / decompress

- implement full compression / decompression

- add CNN based feature extractor

## Installation

To install the necessary dependencies, create a virtual environment and install the requirements:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

### CLI

TODO

### Training

Requires Slate, which is not currently publicly available. Install via (requires repo access):

```bash
pip install -e git+ssh://git@dominik-roth.eu/dodox/Slate.git#egg=slate
```

To train the model, run:

```bash
python main.py <config_file.yaml> <exp_name>
```

## Icon Attribution
The icon used in this repository is a combination of the Pied Piper logo from the HBO show _Silicon Valley_ and the Neuralink logo. I do not hold any trademarks on either logo; they are owned by their respective entities.

## License

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0). For commercial use, including commercial usage of derived works, please contact me at [mail@dominik-roth.eu](mailto:mail@dominik-roth.eu).

You can view the full text of the license [here](LICENSE).