import wave
import numpy as np
import matplotlib.pyplot as plt

def load_wav(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        num_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        raw_data = wav_file.readframes(num_frames)
        
    return sample_rate, num_channels, sampwidth, raw_data

def inspect_wav(file_path):
    sample_rate, num_channels, sampwidth, raw_data = load_wav(file_path)
    
    fmt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth)
    
    data = np.frombuffer(raw_data, dtype=fmt)
    
    print(f"Sample Rate: {sample_rate}")
    print(f"Channels: {num_channels}")
    print(f"Sample Width: {sampwidth} bytes")
    
    # Calculate and print max/min values and required bits
    max_value = np.max(data)
    min_value = np.min(data)
    max_bits = np.ceil(np.log2(max_value + 1))
    min_bits = np.ceil(np.log2(abs(min_value) + 1))
    
    # Ensure to include the sign bit
    bits_required = max(max_bits, min_bits) + 1
    
    print(f"Maximum Value: {max_value}")
    print(f"Minimum Value: {min_value}")
    print(f"Bits Required to Represent Maximum Value: {max_bits}")
    print(f"Bits Required to Represent Minimum Value: {min_bits}")
    print(f"Total Bits Required (including sign bit): {bits_required}")

file_path = 'data/d657634f-4d93-410c-8a95-52e2da100a72.wav'
inspect_wav(file_path)


# Sample Rate: 19531
# Channels: 1
# Sample Width: 2 bytes
# Maximum Value: 18929
# Minimum Value: -9513
# Bits Required to Represent Maximum Value: 15.0
# Bits Required to Represent Minimum Value: 14.0
# Total Bits Required (including sign bit): 16.0
