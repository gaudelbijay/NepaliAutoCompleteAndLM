import torch
import torch.nn as nn
from model import RNN
import utils
from train import train

if __name__ == '__main__':
    sentence, vocab = utils.read_from_text()
    vocab_len = len(vocab)
    X, Y = utils.tensor_from_sentences(sentence, vocab)
    batches_X, batches_Y = utils.batchify(X, Y)

    network = RNN(vocab_len, 1000, vocab_len, 2)
    optimizer = torch.optim.Adam(network.parameters())
    loss_fuction = nn.KLDivLoss()
    train(network, optimizer, loss_fuction, 1, batches_X, batches_Y)
