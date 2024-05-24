from slate import Slate, Slate_Runner

from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput

from data_processing import download_and_extract_data, load_all_wavs, delta_encode
from model import LSTMPredictor, FixedInputNNPredictor
from train import train_model
from bitstream import ArithmeticEncoder, IdentityEncoder, Bzip2Encoder

class SpikeRunner(Slate_Runner):
    def setup(self, name):
        self.name = name
        slate, config = self.slate, self.config

        # Consume config sections
        preprocessing_config = slate.consume(config, 'preprocessing', expand=True)
        predictor_config = slate.consume(config, 'predictor', expand=True)
        training_config = slate.consume(config, 'training', expand=True)
        bitstream_config = slate.consume(config, 'bitstream_encoding', expand=True)
        data_config = slate.consume(config, 'data', expand=True)

        # Data setup
        data_url = slate.consume(data_config, 'url')
        data_dir = slate.consume(data_config, 'directory')
        download_and_extract_data(data_url, data_dir)
        all_data = load_all_wavs(data_dir)
        
        self.epochs = slate.consume(training_config, 'epochs')
        self.batch_size = slate.consume(training_config, 'batch_size')
        self.learning_rate = slate.consume(training_config, 'learning_rate')
        self.use_delta_encoding = slate.consume(preprocessing_config, 'use_delta_encoding')
        self.eval_freq = slate.consume(training_config, 'eval_freq')
        self.save_path = slate.consume(training_config, 'save_path', 'models')

        if self.use_delta_encoding:
            all_data = [delta_encode(d) for d in all_data]

        # Split data into train and test sets
        split_ratio = slate.consume(data_config, 'split_ratio', 0.8)
        split_idx = int(len(all_data) * split_ratio)
        self.train_data = all_data[:split_idx]
        self.test_data = all_data[split_idx:]
        
        # Model setup
        self.model = self.get_model(predictor_config)
        self.encoder = self.get_encoder(bitstream_config)

    def get_model(self, config):
        model_type = slate.consume(config, 'type')
        if model_type == 'lstm':
            return LSTMPredictor(
                input_size=slate.consume(config, 'input_size'), 
                hidden_size=slate.consume(config, 'hidden_size'), 
                num_layers=slate.consume(config, 'num_layers')
            )
        elif model_type == 'fixed_input_nn':
            return FixedInputNNPredictor(
                input_size=slate.consume(config, 'fixed_input_size'), 
                hidden_size=slate.consume(config, 'hidden_size')
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_encoder(self, config):
        encoder_type = slate.consume(config, 'type')
        if encoder_type == 'arithmetic':
            return ArithmeticEncoder()
        elif encoder_type == 'identity':
            return IdentityEncoder()
        elif encoder_type == 'bzip2':
            return Bzip2Encoder()
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def run(self, run, forceNoProfile=False):
        if self.slate.consume(self.config, 'profiler.enable', False) and not forceNoProfile:
            print('{PROFILER RUNNING}')
            with PyCallGraph(output=GraphvizOutput(output_file=f'./profiler/{self.name}.png')):
                self.run(run, forceNoProfile=True)
            print('{PROFILER DONE}')
            return

        train_model(
            self.model, self.train_data, self.test_data, 
            self.epochs, self.batch_size, self.learning_rate, 
            self.use_delta_encoding, self.encoder, self.eval_freq, self.save_path
        )

if __name__ == '__main__':
    slate = Slate({'spikey': SpikeRunner})
    slate.from_args()
