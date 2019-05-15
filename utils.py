import random
from itertools import chain

import torch
from torch.autograd import Variable

from vocab import Vocab


def chat(input_text, tokenizer, model, vocab, max_output_len=25):
    input_text = vocab.label_encode(tokenizer(input_text))
    input_text = Variable(torch.LongTensor([input_text]), volatile=True).cuda()
    input_text = model.embedding(input_text)
    output = [vocab.SOS_LABEL]

    _, thought = model.encoder(input_text)

    prediction = vocab.SOS_LABEL
    while len(output) < max_output_len and prediction != vocab.EOS_LABEL:
        prediction = Variable(torch.LongTensor([[prediction]]), volatile=True).cuda()
        prediction = model.embedding(prediction)
        prediction, thought = model.decoder(prediction, thought)
        prediction = prediction.data.topk(1)[1][0, 0, 0]
        output.append(prediction)

    return ' '.join(vocab.label_decode(output))


def pad_seqs(seqs):
    max_seq_len = max(len(s) for s in seqs)
    result = []
    for s in seqs:
        padding = [Vocab.PAD_LABEL] * (max_seq_len - len(s))
        result.append(s + padding)
    return result


def split_data(data, train_percent=.6, val_percent=.2):
    random.shuffle(data)

    num_examples = len(data)
    num_train = int(train_percent * num_examples)
    num_val = int(val_percent * num_examples)

    return data[:num_train], data[num_train:num_train + num_val], data[num_train + num_val:]


def save_data(data, encoding, file_path):
    data = [' '.join(line) for line in chain(*data)]
    with open(file_path, 'w') as f:
        f.write('\n'.join(data))


def load_data(encoding, file_path):
    with open(file_path, 'r', encoding = 'ISO-8859-2') as f:
        file_data = [line.split() for line in f]

    output_data = []
    for i in range(0, len(file_data), 2):
        qa_pair = (file_data[i], file_data[i + 1])
        output_data.append(qa_pair)
    return output_data
