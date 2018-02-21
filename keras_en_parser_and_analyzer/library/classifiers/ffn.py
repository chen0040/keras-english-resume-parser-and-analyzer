from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import model_from_json, Sequential
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from keras_en_parser_and_analyzer.library.utility.glove_loader import GloveModel
from keras_en_parser_and_analyzer.library.utility.tokenizer_utils import word_tokenize


class WordVecGloveFFN(object):

    model_name = 'glove_ffn'

    def __init__(self):
        self.model = None
        self.glove_model = GloveModel()
        self.config = None
        self.word2idx = None
        self.idx2word = None
        self.max_len = None
        self.config = None
        self.vocab_size = None
        self.labels = None

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecGloveFFN.model_name + '_weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecGloveFFN.model_name + '_config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecGloveFFN.model_name + '_architecture.json'

    def load_model(self, model_dir_path):
        json = open(self.get_architecture_file_path(model_dir_path), 'r').read()
        self.model = model_from_json(json)
        self.model.load_weights(self.get_weight_file_path(model_dir_path))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        config_file_path = self.get_config_file_path(model_dir_path)

        self.config = np.load(config_file_path).item()

        self.idx2word = self.config['idx2word']
        self.word2idx = self.config['word2idx']
        self.max_len = self.config['max_len']
        self.vocab_size = self.config['vocab_size']
        self.labels = self.config['labels']

    def load_glove_model(self, data_dir_path, embedding_dim=None):
        self.glove_model.load(data_dir_path, embedding_dim=embedding_dim)

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', input_dim=self.glove_model.embedding_dim))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=2, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, text_data_model, text_label_pairs, model_dir_path, batch_size=None, epochs=None,
            test_size=None, random_state=None):
        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 20
        if test_size is None:
            test_size = 0.3
        if random_state is None:
            random_state = 42

        self.config = text_data_model
        self.idx2word = self.config['idx2word']
        self.word2idx = self.config['word2idx']
        self.max_len = self.config['max_len']
        self.vocab_size = self.config['vocab_size']
        self.labels = self.config['labels']

        np.save(self.get_config_file_path(model_dir_path), self.config)

        self.create_model()
        json = self.model.to_json()
        open(self.get_architecture_file_path(model_dir_path), 'w').write(json)

        ys = []
        X = np.zeros(shape=(len(text_label_pairs), self.glove_model.embedding_dim))
        for i, (text, label) in enumerate(text_label_pairs):
            words = [w.lower() for w in word_tokenize(text)]
            E = np.zeros(shape=(self.glove_model.embedding_dim, self.max_len))
            for j in range(len(words)):
                word = words[j]
                try:
                    E[:, j] = self.glove_model.encode_word(word)
                except KeyError:
                    pass
            X[i, :] = np.sum(E, axis=1)
            ys.append(self.labels[label])
        Y = np_utils.to_categorical(ys, len(self.labels))

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        weight_file_path = self.get_weight_file_path(model_dir_path)

        checkpoint = ModelCheckpoint(weight_file_path)

        history = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data=[x_test, y_test], callbacks=[checkpoint],
                                 verbose=1)

        self.model.save_weights(weight_file_path)

        np.save(model_dir_path + '/' + WordVecGloveFFN.model_name + '-history.npy', history.history)

        score = self.model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
        print('score: ', score[0])
        print('accuracy: ', score[1])

        return history

    def predict(self, sentence):

        tokens = [w.lower() for w in word_tokenize(sentence)]

        X = np.zeros(shape=(1, self.glove_model.embedding_dim))
        E = np.zeros(shape=(self.glove_model.embedding_dim, self.max_len))
        for j in range(0, len(tokens)):
            word = tokens[j]
            try:
                E[:, j] = self.glove_model.encode_word(word)
            except KeyError:
                pass
        X[0, :] = np.sum(E, axis=1)
        output = self.model.predict(X)
        return output[0]

    def predict_class(self, sentence):
        predicted = self.predict(sentence)
        idx2label = dict([(idx, label) for label, idx in self.labels.items()])
        return idx2label[np.argmax(predicted)]

    def test_run(self, sentence):
        print(self.predict(sentence))


def main():
    app = WordVecGloveFFN()
    app.test_run('i liked the Da Vinci Code a lot.')
    app.test_run('i hated the Da Vinci Code a lot.')


if __name__ == '__main__':
    main()
