import ast
import os

from nltk import word_tokenize

from utils import save_data
from utils import split_data

CORPUS_PATH = 'data/cornell movie-dialogs corpus'
PROCESSED_PATH = 'data/processed/'
TRAIN_PATH = os.path.join(PROCESSED_PATH, 'cornell_train.txt')
VAL_PATH = os.path.join(PROCESSED_PATH, 'cornell_val.txt')
TEST_PATH = os.path.join(PROCESSED_PATH, 'cornell_test.txt')

ENCODING = 'ISO-8859-2'

MAX_SEQ_LEN = 50
MIN_SEQ_LEN = 10


def get_cornell_data():
    data = []
    for c in get_conversations():
        num_movie_lines = len(c)
        for i in range(num_movie_lines - 1):
            qa_pair = (c[i], c[i + 1])
            data.append(qa_pair)
    return data


def get_conversations():
    conversations_path = os.path.join(CORPUS_PATH, 'movie_conversations.txt')
    conversations = load_raw_data(conversations_path)
    id_to_line = get_id_to_line()
    for c in conversations:
        line_id_list = c[-1]  # represented as a string
        yield [id_to_line[line_id] for line_id in ast.literal_eval(line_id_list)]


def get_id_to_line():
    lines_path = os.path.join(CORPUS_PATH, 'movie_lines.txt')
    lines = load_raw_data(lines_path)
    id_to_line = {}
    for l in lines:
        line_id = l[0]
        line_text = l[-1]
        id_to_line[line_id] = line_text
    return id_to_line


def load_raw_data(file_path):
    seperator = ' +++$+++ '
    with open(file_path, 'r', encoding = 'ISO-8859-2') as f:
        for line in f:
            yield line.strip().split(seperator)


def filter_seq(seq):
    return len(seq) < MIN_SEQ_LEN or len(seq) > MAX_SEQ_LEN


def tokenize(text):
    return [word for word in word_tokenize(text.lower()) if not is_number(word)]


# https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
def is_number(word):
    try:
        float(word)
    except ValueError:
        return False
    return True


if __name__ == '__main__':
    print('Getting data...')
    data = get_cornell_data()

    print('Tokenizing...')
    data = [(tokenize(q), tokenize(a)) for q, a in data]

    print('Filtering...')
    data = [(q, a) for q, a in data if not filter_seq(q) and not filter_seq(a)]

    print('Saving...')
    train, val, test = split_data(data)
    save_data(train, ENCODING, TRAIN_PATH)
    save_data(val, ENCODING, VAL_PATH)
    save_data(test, ENCODING, TEST_PATH)
