import torch
import torch.nn as nn
from model import RNN
import utils
from train import train
import numpy as np

if __name__ == '__main__':
    sentence, vocab = utils.read_from_text()
    vocab_len = len(vocab)
    idx2vocab = np.array(vocab)
    X, Y = utils.tensor_from_sentences(sentence, vocab)
    # batches_X, batches_Y = utils.batchify(X, Y)
    embed_size = 50

    network = RNN(input_size=vocab_len, embed_dim=embed_size, hidden_size=1000, output_size=vocab_len,
                  n_layers=2)
    # optimizer = torch.optim.Adam(network.parameters())
    loss_fuction = nn.KLDivLoss()
    train(network, loss_fuction, 1, X, Y)
    t = 250
    seq = utils.tensor_from_single_batch(sentence[t], vocab)
    print(sentence[t])
    utils.predict(network, seq[:-1], idx2vocab)
