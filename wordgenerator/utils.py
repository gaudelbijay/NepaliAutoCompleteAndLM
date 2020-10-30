import torch
# import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt


def read_from_text(filepath="../preprocess/cleaned.txt", encoding='utf8', threshold=3):
    f = open(filepath, "r", encoding=encoding)
    text = f.read()
    f.close()
    sentences = text.split("।")
    words = []
    sens_len = []
    for s in sentences:
        w_s = s.split()
        sens_len.append(len(w_s))

        for w in s.split():
            words.append(w)
    # plt.hist(sens_len, 20)
    vocab_counter = Counter(words)
    vocab = ["<UNK>"] + [w for (w, c) in vocab_counter.most_common() if c >= threshold]
    return sentences, vocab


def tensor_from_sentences(sentences, vocab, max_len=30):
    X = []
    Y = []
    vocab_len = len(vocab)
    for s in sentences:
        w_s = s.split()
        idx_ip = [vocab.index(w) if w in vocab else 0 for w in w_s]
        len_words = len(idx_ip)
        if len_words < max_len:
            pad = [0] * (max_len - len_words)
            idx_ip += pad
        else:
            idx_ip = idx_ip[: max_len]
        idx_op = idx_ip[1:] + [0]

        one_hot = []
        for idx in idx_op:
            hot = [0] * vocab_len
            hot[idx] = 1
            one_hot.append(hot)
        X.append(idx_ip)
        Y.append(one_hot)
    return X, Y


def tensor_from_single_batch(sentence, vocab):
    words = sentence.split(' ')
    idx_ip = [vocab.index(w) if w in vocab else 0 for w in words]
    idx_ip = torch.tensor(idx_ip, dtype=torch.long)
    return idx_ip


def batchify(X, Y, batch_size=100):
    batches_x = []
    batches_y = []
    len_data = len(X)
    for start in range(0, len_data, batch_size):
        end = None
        if start + batch_size < len_data:
            end = start + batch_size
        else:
            end = len_data
        x = torch.tensor(X[start:end], dtype=torch.long)
        y = torch.tensor(Y[start:end], dtype=torch.float)

        batches_x.append(x)
        batches_y.append(y)
    return batches_x, batches_y


def predict(model, sequence, idx2vocab, len_pred=10, device='cpu'):
    sequence = sequence.to(device)
    sequence = torch.unsqueeze(sequence, 0)
    prediction, _ = model(sequence)

    size = prediction.size()
    output = prediction.view(-1, size[0] * size[1], size[-1])
    output = torch.squeeze(output)
    output = torch.argmax(output, dim=0)
    # print(output)

    out = ' '
    for w in output[:len_pred]:
        out += '\t' + idx2vocab[w]

    print('output: ', out[1:], '\n')
