<p align='center'>
  <img src='./spikey.svg'>
</p>

# Spikey

This repository contains a solution for the [Neuralink Compression Challenge](https://content.neuralink.com/compression-challenge/README.html). The challenge involves compressing raw electrode recordings from a Neuralink implant. These recordings are taken from the motor cortex of a non-human primate while playing a video game.

**TL;DR;** We achieve a lossless compression ratio of **3.513** using a predictive model that employs discrete Meyer wavelet convolution for signal decomposition, inter-thread message passing to account for underlying brain region activity, and Rice coding for efficient entropy encoding. We believe this to be close to the optimum achievable with lossless compression and argue against pursuing lossless compression as a further goal.

## Challenge Overview

The Neuralink N1 implant generates approximately 200 Mbps of electrode data from 1024 electrodes, each sampling at 20 kHz with a 10-bit resolution. This data is recorded from the motor cortex of a non-human primate while playing video games. Given the implant's wireless transmission capability of about 1 Mbps, achieving a compression ratio of over 200x is essential.

#### Key Requirements:

- **Real-time Compression**: The compression algorithm must operate in less than 1 millisecond to ensure real-time performance.
- **Low Power Consumption**: The total power consumption, including the radio, must be below 10 milliwatts.
- **Lossless Compression**: The compression must be lossless to maintain data integrity.

## Data Analysis

The `analysis.ipynb` notebook contains a detailed analysis of the data. We found that there is sometimes significant cross-correlation between the different threads, so we find it vital to use this information for better compression. This cross-correlation allows us to improve the accuracy of our predictions and reduce the overall amount of data that needs to be transmitted. We sometimes even observe certain 'structures' on multiple threads, but shifted a couple steps in time (might that not be handy for compression?).

## Algorithm Overview

### 1 - Thread Topology Reconstruction

As the first step, we analyze readings from the threads to construct an approximate topology of the threads in the brain. The distance metric we generate only approximately represents true Euclidean distances, but rather the 'distance' in common activity. This topology must only be computed once for a given implant and may be updated for thread movements but is not part of the regular compression/decompression process.

### 2 - Predictive Model

The main workhorse of our compression approach is a predictive model running both in the compressor and decompressor. With good predictions of the data, only the error between the prediction and actual data must be transmitted. We make use of the previously constructed topology to allow the predictive model's latent to represent the activity of brain regions based on the reading of the threads instead of just for threads themselves.

We separate the predictive model into four parts:

1. **Feature Extraction**: This module processes a given history of readings for a single thread and extracts relevant features (using mostly wavelet and Fourier transforms). Highly configurable, this module performs the heavy lifting of signal analysis, allowing shallow neural networks to handle the rest.

2. **Latent Projector**: This takes the feature vectors and projects them into a latent space. The latent projector can be configured as a fully connected network or an RNN (LSTM) with an arbitrary shape.

3. **[MiddleOut](https://www.youtube.com/watch?v=l49MHwooaVQ)**: For each thread, this module performs message passing according to the thread topology. Their latent representations along with their distance metrics are used to generate region latent representations.

4. **Predictor**: This module takes the region latent representation from the MiddleOut module and predicts the next timestep. The goal is to minimize the prediction error during training. It can be configured to be an FCNN of arbitrary shape.

The neural networks used are rather small, making it possible to meet the latency and power requirements if implemented more efficiently. (Some of the available feature extractors are somewhat expensive thought).

If we were to give up on lossless compression, one could expand MiddleOut to form a joint latent over all threads and transmit that.

### 3 - Efficient Bitstream Encoding

The best performing available bitstream encoder is a Huffman code based on a binomial prior fitted to the delta distribution, but we also provide others such as Rice.

Check the `config.yaml` for a bit more info on the architecture.

## Discussion

### On Lossless 200x Compression

Expecting a 200x compression ratio is ludicrous, as it would mean transmitting only 1 bit per 20 data points. Given the high entropy of the readings, this is an absurd goal. Anyone who thinks lossless 200x compression is remotely feasible has a woefully inadequate grasp of information theory. Please, do yourself a favor and read Shannon’s paper.

Let's see how far we can get with the approach presented here...

### On fucked up wav files

Why is the dataset provided not 10-bit if the readings are? They are all 16-bit. And the last 6 bits are not all zeros. We know they can't encode sensible information when the readings are only 10-bit, but we also can't just throw them away since they do contain something. We also observe that all possible values the data points can take on are separated by 64 or 63 (64 would make sense; 63 very much does not). (See `fucked_up_wavs.py`)

### Speculation on the Challenge Background

Neuralink designed the N1 implant with on-chip spike detection and analysis capabilities, assuming these spike descriptions would suffice for intent recognition and could be transmitted via a low bandwidth 1 Mbps connection, with the rest being noise. However, during the PRIME study with 'Noland Arbaugh', the implant's threads moved out of the brain more than expected, leading to a degradation of intent recognition capabilities.

In response, Neuralink tried a hail mary: They ignored the electrodes no longer in the brain, skipped on-device spike analysis, and transmitted the remaining data as losslessly as possible. Remarkably, using advanced ML algorithms, they improved intent detection significantly, outperforming the old pipeline with all electrodes intact.

This led to a new strategy: Discard spike analysis and use the new algorithm on all electrode data in future trials. However, the vast amount of data generated couldn't be transmitted using the existing bandwidth. So Neuralink turned to the internet for solutions, essentially crowd-sourcing their problem-solving because they couldn't figure it out themselves.

The new ML algorithm's effectiveness at extracting valuable information from what was previously considered noise makes the goal of 200x compression even less sensible. If the new ML algorithms can extract more information, this 'true' information contained in the readings is rather incompressible.

Neuralink should regard compression as part of their ML model for intent extraction. "Intelligence is the ability to disregard irrelevant information." The focus should be on lossy compression that minimizes information loss in the latent space of the ML pipeline rather than the input space. There should be no decompression step (except for entropy coding), just stay in the 'compressed' latent space. Future implants should also have increased bandwidth to support this approach.

## Preliminary Results

Current best: **3.513** (not counting encoder / decoder size, just data)

Theoretical max via Shannon: [3.439](https://x.com/usrbinishan/status/1794948522112151841), best found online: [3.35](https://github.com/phoboslab/neuralink_brainwire). (Shannon assumptions don't hold for this algo, so max does not apply)  
Config Outline: Meyer Wavelets for feature extraction (are great at recognizing spikes). Rice as bitstream encoder with k=2. 8D Latents. Residual skip-con in MiddleOut.

The presented python implementation should be regarded as a POC; the used networks are rather small, making them trivially usable on-chip if implemented more efficiently. Only the discrete Meyer wavelet convolution could be somewhat difficult to pull off, but the chips contain hardware for spike detection and analysis (according to information released by Neuralink), so these could be used instead. There is no lookahead of any kind, so we can send each new reading off once it went though the math. Compression and decompression has to be performed jointly over all threads, since we pass messages between threads during MiddleOut.

## TODO

- Tune HPs / Benchmark networks

- cli for compress / decompress

- implement full compression / decompression

- add CNN based feature extractor

- make usable with eval.sh


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

---

And always remember: Fuel on!
