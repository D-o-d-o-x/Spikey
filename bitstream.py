import bz2, math
import heapq
from abc import ABC, abstractmethod
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel
import numpy as np

class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, data):
        pass

    @abstractmethod
    def decode(self, encoded_data):
        pass

    @abstractmethod
    def build_model(self, delta_samples):
        pass

class IdentityEncoder(BaseEncoder):
    def encode(self, data):
        return data

    def decode(self, encoded_data):
        return encoded_data

    def build_model(self, delta_samples):
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

    def build_model(self, delta_samples):
        # Convert data to list of tuples
        data = [tuple(d) for d in delta_samples]
        symbol_counts = {symbol: data.count(symbol) for symbol in set(data)}
        total_symbols = sum(symbol_counts.values())
        probabilities = {symbol: count / total_symbols for symbol, count in symbol_counts.items()}
        self.model = StaticModel(probabilities)

class Bzip2Encoder(BaseEncoder):
    def encode(self, data):
        return bz2.compress(bytearray(data))

    def decode(self, encoded_data):
        return list(bz2.decompress(encoded_data))

    def build_model(self, data):
        pass

class BinomialHuffmanEncoder(BaseEncoder):
    def encode(self, data):
        return ''.join(self.codebook[int(value)+1024] for value in data)

    def decode(self, encoded_data):
        decoded_output = []
        current_node = self.root
        for bit in encoded_data:
            if bit == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right
            
            if current_node.left is None and current_node.right is None:
                decoded_output.append(current_node.value-1024)
                current_node = self.root
        
        return decoded_output

    def _generate_codes(self, root):
        if root is None:
            return {}
        codebook = {}
        stack = [(root, "")]
        while stack:
            node, prefix = stack.pop()
            if node.value is not None:
                codebook[node.value] = prefix
            if node.right is not None:
                stack.append((node.right, prefix + "1"))
            if node.left is not None:
                stack.append((node.left, prefix + "0"))
        return codebook

    def build_model(self, delta_samples, adaptive=True):
        num_symbols = 2**11

        mean = (num_symbols - 1) / 2
        std_dev = math.sqrt(num_symbols / 4)
        if adaptive:
            std_dev = np.std(delta_samples)

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

class RiceEncoder(BaseEncoder):
    def encode(self, data):
        data = np.array(data).astype(int)
        encoded_data = []

        for num in data:
            num = self.zigzag_encode(num)
            q = num // self.m
            r = num % self.m
            encoded_data.append('1' * q + '0' + format(r, f'0{self.k}b'))
        
        return ''.join(encoded_data)

    def decode(self, encoded_data):
        decoded_output = []
        i = 0

        while i < len(encoded_data):
            q = 0
            while encoded_data[i] == '1':
                q += 1
                i += 1
            i += 1  # skip the '0'
            r = int(encoded_data[i:i + self.k], 2)
            i += self.k
            num = q * self.m + r
            decoded_output.append(self.zigzag_decode(num))
        
        return np.array(decoded_output)

    def build_model(self, data, k=3):
        self.k = k
        self.m = 1 << k

    def zigzag_encode(self, value):
        return (value << 1) ^ (value >> 31)

    def zigzag_decode(self, value):
        return (value >> 1) ^ -(value & 1)