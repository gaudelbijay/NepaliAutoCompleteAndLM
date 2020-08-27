import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_size, output_size,
                 n_layers=1,batch_first=True, dropout_rate=0.0, rnn_type='GRU'):
        super(RNN, self).__init__()
        self.batch_first = batch_first
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.rnn_type = rnn_type

        self.encoder = nn.Embedding(input_size, embedding_dim=embed_dim)
        
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_size, n_layers, dropout=self.dropout_rate, batch_first=batch_first)
        else:
            self.rnn = nn.LSTM(embed_dim, hidden_size, n_layers, dropout=self.dropout_rate, batch_first=batch_first)
        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, inputs):
        
        props = self.encoder(inputs)
        props, hidden = self.rnn(props)
        props = self.lin1(props)
        props = self.relu(props)
        props = self.decoder(props)        
        output = self.softmax(props)
        return output,hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
