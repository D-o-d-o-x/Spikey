import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

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

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        c0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out)
        return out

    def encode(self, data):
        self.eval()
        encoded_data = []

        with torch.no_grad():
            for i in range(len(data) - 1):
                context = torch.tensor(data[max(0, i - self.rnn.hidden_size):i], dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(next(self.parameters()).device)
                if context.shape[1] == 0:
                    context = torch.zeros((1, 1, 1)).to(next(self.parameters()).device)
                prediction = self.forward(context).cpu().numpy()[0][0]
                delta = data[i] - prediction
                encoded_data.append(delta)
        
        return encoded_data

    def decode(self, encoded_data):
        self.eval()
        decoded_data = []

        with torch.no_grad():
            for i in range(len(encoded_data)):
                context = torch.tensor(decoded_data[max(0, i - self.rnn.hidden_size):i], dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(next(self.parameters()).device)
                if context.shape[1] == 0:
                    context = torch.zeros((1, 1, 1)).to(next(self.parameters()).device)
                prediction = self.forward(context).cpu().numpy()[0][0]
                decoded_data.append(prediction + encoded_data[i])
        
        return decoded_data

class FixedInputNNPredictor(BaseModel):
    def __init__(self, input_size, hidden_size):
        super(FixedInputNNPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def encode(self, data):
        self.eval()
        encoded_data = []

        with torch.no_grad():
            for i in range(len(data) - self.fc1.in_features):
                context = torch.tensor(data[i:i + self.fc1.in_features], dtype=torch.float32).unsqueeze(0).to(next(self.parameters()).device)
                prediction = self.forward(context).cpu().numpy()[0][0]
                delta = data[i + self.fc1.in_features] - prediction
                encoded_data.append(delta)
        
        return encoded_data

    def decode(self, encoded_data):
        self.eval()
        decoded_data = []

        with torch.no_grad():
            for i in range(len(encoded_data)):
                context = torch.tensor(decoded_data[max(0, i - self.fc1.in_features):i], dtype=torch.float32).unsqueeze(0).to(next(self.parameters()).device)
                prediction = self.forward(context).cpu().numpy()[0][0]
                decoded_data.append(prediction + encoded_data[i])
        
        return decoded_data
