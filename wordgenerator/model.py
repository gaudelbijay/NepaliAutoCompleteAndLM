import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout_rate=0.0, rnn_type='GRU'):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.rnn_type = rnn_type

        self.encoder = nn.Embedding(input_size, hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout_rate)
        else:
            self.rnn - nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout_rate)

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        inputs = self.encoder(inputs.view(1, -1))
        output, hidden = self.rnn(inputs.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
