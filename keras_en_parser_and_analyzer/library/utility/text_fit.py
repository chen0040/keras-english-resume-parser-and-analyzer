import collections

from keras_en_parser_and_analyzer.library.utility.tokenizer_utils import word_tokenize
import os


def fit_text(data_dir_path, max_vocab_size=None, label_type=None):
    if label_type is None:
        label_type = 'line_type'
    if max_vocab_size is None:
        max_vocab_size = 5000
    counter = collections.Counter()
    max_len = 0
    labels = dict()
    for f in os.listdir(data_dir_path):
        data_file_path = os.path.join(data_dir_path, f)
        if os.path.isfile(data_file_path) and f.lower().endswith('.txt'):
            file = open(data_file_path, mode='rt', encoding='utf8')

            for line in file:
                line_type, line_label, sentence = line.strip().split('\t')
                tokens = [x.lower() for x in word_tokenize(sentence)]
                for token in tokens:
                    counter[token] += 1
                max_len = max(max_len, len(tokens))
                label = line_label
                if label_type != 'line_label':
                    label = line_type
                if label not in labels:
                    labels[label] = len(labels)
            file.close()

    word2idx = collections.defaultdict(int)
    for idx, word in enumerate(counter.most_common(max_vocab_size)):
        word2idx[word[0]] = idx
    idx2word = {v: k for k, v in word2idx.items()}
    vocab_size = len(word2idx) + 1

    model = dict()

    model['word2idx'] = word2idx
    model['idx2word'] = idx2word
    model['vocab_size'] = vocab_size
    model['max_len'] = max_len
    model['labels'] = labels

    return model
