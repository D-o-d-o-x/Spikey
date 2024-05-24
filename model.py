import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module):
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
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(self.device)
        c0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(self.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out)
        return out

    def encode(self, data):
        self.eval()
        encoded_data = []

        context_size = self.hidden_size  # Define an appropriate context size
        with torch.no_grad():
            for i in range(len(data) - 1):
                context = torch.tensor(data[max(0, i - context_size):i]).reshape(1, -1, 1).to(self.device)
                if context.size(1) == 0:  # Handle empty context
                    continue
                prediction = self.forward(context).squeeze(0).cpu().numpy()[0]
                delta = data[i] - prediction
                encoded_data.append(delta)
        
        return encoded_data

    def decode(self, encoded_data):
        self.eval()
        decoded_data = []

        context_size = self.hidden_size  # Define an appropriate context size
        with torch.no_grad():
            for i in range(len(encoded_data)):
                context = torch.tensor(decoded_data[max(0, i - context_size):i]).reshape(1, -1, 1).to(self.device)
                if context.size(1) == 0:  # Handle empty context
                    continue
                prediction = self.forward(context).squeeze(0).cpu().numpy()[0]
                decoded_data.append(prediction + encoded_data[i])
        
        return decoded_data

class FixedInputNNPredictor(BaseModel):
    def __init__(self, input_size, hidden_size):
        super(FixedInputNNPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def encode(self, data):
        self.eval()
        encoded_data = []

        context_size = self.fc1.in_features  # Define an appropriate context size
        with torch.no_grad():
            for i in range(len(data) - context_size):
                context = torch.tensor(data[i:i + context_size]).reshape(1, -1).to(self.device)
                if context.size(1) == 0:  # Handle empty context
                    continue
                prediction = self.forward(context).squeeze(0).cpu().numpy()[0]
                delta = data[i + context_size] - prediction
                encoded_data.append(delta)
        
        return encoded_data

    def decode(self, encoded_data):
        self.eval()
        decoded_data = []

        context_size = self.fc1.in_features  # Define an appropriate context size
        with torch.no_grad():
            for i in range(len(encoded_data)):
                context = torch.tensor(decoded_data[max(0, i - context_size):i]).reshape(1, -1).to(self.device)
                if context.size(1) == 0:  # Handle empty context
                    continue
                prediction = self.forward(context).squeeze(0).cpu().numpy()[0]
                decoded_data.append(prediction + encoded_data[i])
        
        return decoded_data
