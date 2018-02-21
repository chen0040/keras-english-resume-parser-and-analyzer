import urllib.request
import os
import zipfile
import sys
import numpy as np

from keras_en_parser_and_analyzer.library.utility.tokenizer_utils import word_tokenize


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_glove(data_dir_path, glove_file_path):
    if not os.path.exists(glove_file_path):
        if not os.path.exists(data_dir_path):
            os.makedirs(data_dir_path)

        glove_zip = data_dir_path + '/glove.6B.zip'

        if not os.path.exists(glove_zip):
            print('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        print('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall('very_large_data')
        zip_ref.close()


def load_glove(data_dir_path=None, embedding_dim=None):
    """
    Load the glove models (and download the glove model if they don't exist in the data_dir_path
    :param data_dir_path: the directory path on which the glove model files will be downloaded and store
    :param embedding_dim: the dimension of the word embedding, available dimensions are 50, 100, 200, 300, default is 100
    :return: the glove word embeddings
    """
    if embedding_dim is None:
        embedding_dim = 100

    glove_file_path = data_dir_path + "/glove.6B." + str(embedding_dim) + "d.txt"
    download_glove(data_dir_path, glove_file_path)
    _word2em = {}
    file = open(glove_file_path, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
    file.close()
    return _word2em


class GloveModel(object):
    """
    Class the provides the glove embedding and document encoding functions
    """
    model_name = 'glove-model'

    def __init__(self):
        self.word2em = None
        self.embedding_dim = None

    def load(self, data_dir_path, embedding_dim=None):
        if embedding_dim is None:
            embedding_dim = 100
        self.embedding_dim = embedding_dim
        self.word2em = load_glove(data_dir_path, embedding_dim)

    def encode_word(self, word):
        w = word.lower()
        if w in self.word2em:
            return self.word2em[w]
        else:
            return np.zeros(shape=(self.embedding_dim, ))

    def encode_docs(self, docs, max_allowed_doc_length=None):
        if max_allowed_doc_length is None:
            max_allowed_doc_length = 500
        doc_count = len(docs)
        X = np.zeros(shape=(doc_count, self.embedding_dim))
        max_len = 0
        for doc in docs:
            max_len = max(max_len, len([word_tokenize(doc)]))
        max_len = min(max_len, max_allowed_doc_length)
        for i in range(0, doc_count):
            doc = docs[i]
            words = [w.lower() for w in word_tokenize(doc)]
            E = np.zeros(shape=(self.embedding_dim, max_len))
            for j in range(max_len):
                word = words[j]
                try:
                    E[:, j] = self.word2em[word]
                except KeyError:
                    pass
            X[i, :] = np.sum(E, axis=1)

        return X

    def encode_doc(self, doc, max_allowed_doc_length=None):
        if max_allowed_doc_length is None:
            max_allowed_doc_length = 500

        words = [w.lower() for w in word_tokenize(doc)]
        max_len = min(len(words), max_allowed_doc_length)
        E = np.zeros(shape=(self.embedding_dim, max_len))
        X = np.zeros(shape=(self.embedding_dim, ))
        for j in range(max_len):
            word = words[j]
            try:
                E[:, j] = self.word2em[word]
            except KeyError:
                pass
        X[:] = np.sum(E, axis=1)
        return X
