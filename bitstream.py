import bz2, math
import heapq
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

class BinomialHuffmanEncoder(BaseEncoder):
    def encode(self, data):
        return ''.join(self.codebook[int(value)+512] for value in data)

    def decode(self, encoded_data):
        decoded_output = []
        current_node = self.root
        for bit in encoded_data:
            if bit == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right
            
            if current_node.left is None and current_node.right is None:
                decoded_output.append(current_node.value)
                current_node = self.root
        
        return decoded_output

    def _generate_codes(self, node, prefix="", codebook={}):
        if node is not None:
            if node.value is not None:
                codebook[node.value] = prefix
            self._generate_codes(node.left, prefix + "0", codebook)
            self._generate_codes(node.right, prefix + "1", codebook)
        return codebook

    def build_model(self, data):
        # Number of possible 10-bit integers (0 to 1023)
        num_symbols = 1024

        # Mean and standard deviation for a binomial distribution centered around 0
        mean = (num_symbols - 1) / 2
        std_dev = math.sqrt(num_symbols / 4)

        class Node:
            def __init__(self, value, freq):
                self.value = value
                self.freq = freq
                self.left = None
                self.right = None

            def __lt__(self, other):
                return self.freq < other.freq

        # Build a min-heap
        heap = [Node(x, (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / std_dev) ** 2)) for x in range(num_symbols)]
        heapq.heapify(heap)

        # Merge nodes to build the Huffman tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = Node(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            heapq.heappush(heap, merged)

        # The root of the Huffman tree
        self.root = heapq.heappop(heap)
        self.codebook = self._generate_codes(self.root)
