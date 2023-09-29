import torch.nn as nn

from rnn_models import RegBase, DeepNeuralNetwork

class Simple1DCNN(RegBase):
    def __init__(self,architecture, input_channels, output_size, kernel_size=3, stride=2):
        super(Simple1DCNN, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, 15, kernel_size, stride)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = DeepNeuralNetwork(15, output_size, architecture)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x