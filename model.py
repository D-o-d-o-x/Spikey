import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def encode(self, data):
        pass

    @abstractmethod
    def decode(self, encoded_data):
        pass

class LSTMPredictor(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMPredictor, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out)
        return out

    def encode(self, data):
        self.eval()
        encoded_data = []

        with torch.no_grad():
            for i in range(len(data) - 1):
                context = torch.tensor(data[max(0, i - self.hidden_size):i]).view(1, -1, 1).float()
                prediction = self.forward(context).item()
                delta = data[i] - prediction
                encoded_data.append(delta)
        
        return encoded_data

    def decode(self, encoded_data):
        self.eval()
        decoded_data = []

        with torch.no_grad():
            for i in range(len(encoded_data)):
                context = torch.tensor(decoded_data[max(0, i - self.hidden_size):i]).view(1, -1, 1).float()
                prediction = self.forward(context).item()
                decoded_data.append(prediction + encoded_data[i])
        
        return decoded_data

class FixedInputNNPredictor(BaseModel):
    def __init__(self, input_size, hidden_size):
        super(FixedInputNNPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.input_size = input_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def encode(self, data):
        self.eval()
        encoded_data = []

        with torch.no_grad():
            for i in range(len(data) - self.input_size):
                context = torch.tensor(data[i:i + self.input_size]).view(1, -1).float()
                prediction = self.forward(context).item()
                delta = data[i + self.input_size] - prediction
                encoded_data.append(delta)
        
        return encoded_data

    def decode(self, encoded_data):
        self.eval()
        decoded_data = []

        with torch.no_grad():
            for i in range(len(encoded_data)):
                context = torch.tensor(decoded_data[max(0, i - self.input_size):i]).view(1, -1).float()
                prediction = self.forward(context).item()
                decoded_data.append(prediction + encoded_data[i])
        
        return decoded_data
