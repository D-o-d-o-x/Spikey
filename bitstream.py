import bz2
from abc import ABC, abstractmethod
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel

class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, data):
        pass

    @abstractmethod
    def decode(self, encoded_data, num_symbols):
        pass

    @abstractmethod
    def build_model(self, data):
        pass

class IdentityEncoder(BaseEncoder):
    def encode(self, data):
        return data

    def decode(self, encoded_data, num_symbols):
        return encoded_data

    def build_model(self, data):
        pass

class ArithmeticEncoder(BaseEncoder):
    def encode(self, data):
        if not hasattr(self, 'model'):
            raise ValueError("Model not built. Call build_model(data) before encoding.")
        coder = AECompressor(self.model)
        compressed_data = coder.compress(data)
        return compressed_data

    def decode(self, encoded_data, num_symbols):
        coder = AECompressor(self.model)
        decoded_data = coder.decompress(encoded_data, num_symbols)
        return decoded_data

    def build_model(self, data):
        # Convert data to list of tuples
        data = [tuple(d) for d in data]
        symbol_counts = {symbol: data.count(symbol) for symbol in set(data)}
        total_symbols = sum(symbol_counts.values())
        probabilities = {symbol: count / total_symbols for symbol, count in symbol_counts.items()}
        self.model = StaticModel(probabilities)

class Bzip2Encoder(BaseEncoder):
    def encode(self, data):
        return bz2.compress(bytearray(data))

    def decode(self, encoded_data, num_symbols):
        return list(bz2.decompress(encoded_data))

    def build_model(self, data):
        pass
