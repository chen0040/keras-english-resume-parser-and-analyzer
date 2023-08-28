from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Bidirectional
from keras.models import model_from_json, Sequential
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras_en_parser_and_analyzer.library.utility.tokenizer_utils import word_tokenize


class WordVecLstmSigmoid(object):
    model_name = 'lstm_sigmoid'

    def __init__(self):
        self.model = None
        self.word2idx = None
        self.idx2word = None
        self.max_len = None
        self.config = None
        self.vocab_size = None
        self.labels = None

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecLstmSigmoid.model_name + '_architecture.json'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecLstmSigmoid.model_name + '_weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecLstmSigmoid.model_name + '_config.npy'

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

    def create_model(self):
        embedding_size = 100
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocab_size, output_dim=embedding_size, input_length=self.max_len))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

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

        xs = []
        ys = []
        for text, label in text_label_pairs:
            tokens = [x.lower() for x in word_tokenize(text)]
            wid_list = list()
            for w in tokens:
                wid = 0
                if w in self.word2idx:
                    wid = self.word2idx[w]
                wid_list.append(wid)
            xs.append(wid_list)
            ys.append(self.labels[label])

        X = pad_sequences(xs, maxlen=self.max_len)
        # Y = np_utils.to_categorical(ys, len(self.labels))
        Y = np.array(ys, dtype=np.float32)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        weight_file_path = self.get_weight_file_path(model_dir_path)

        checkpoint = ModelCheckpoint(weight_file_path)

        history = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data=[x_test, y_test], callbacks=[checkpoint],
                                 verbose=1)

        self.model.save_weights(weight_file_path)

        np.save(model_dir_path + '/' + WordVecLstmSigmoid.model_name + '-history.npy', history.history)

        score = self.model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
        print('score: ', score[0])
        print('accuracy: ', score[1])

        return history

    def predict(self, sentence):
        xs = []
        tokens = [w.lower() for w in word_tokenize(sentence)]
        wid = [self.word2idx[token] if token in self.word2idx else 1 for token in tokens]
        xs.append(wid)
        x = pad_sequences(xs, self.max_len)
        output = self.model.predict(x)[0]
        return [1-output[0], output[0]]

    def predict_class(self, sentence):
        predicted = self.predict(sentence)
        idx2label = dict([(idx, label) for label, idx in self.labels.items()])
        return idx2label[np.argmax(predicted)]

    def test_run(self, sentence):
        print(self.predict(sentence))
        
        
class WordVecLstmSoftmax(object):
    model_name = 'lstm_softmax'

    def __init__(self):
        self.model = None
        self.word2idx = None
        self.idx2word = None
        self.max_len = None
        self.config = None
        self.vocab_size = None
        self.labels = None

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecLstmSoftmax.model_name + '_architecture.json'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecLstmSoftmax.model_name + '_weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecLstmSoftmax.model_name + '_config.npy'

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

    def create_model(self):
        embedding_size = 100
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocab_size, output_dim=embedding_size, input_length=self.max_len))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(len(self.labels), activation='softmax'))

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

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

        xs = []
        ys = []
        for text, label in text_label_pairs:
            tokens = [x.lower() for x in word_tokenize(text)]
            wid_list = list()
            for w in tokens:
                wid = 0
                if w in self.word2idx:
                    wid = self.word2idx[w]
                wid_list.append(wid)
            xs.append(wid_list)
            ys.append(self.labels[label])

        X = pad_sequences(xs, maxlen=self.max_len)
        Y = to_categorical(ys, len(self.labels))

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        weight_file_path = self.get_weight_file_path(model_dir_path)

        checkpoint = ModelCheckpoint(weight_file_path)

        history = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data=[x_test, y_test], callbacks=[checkpoint],
                                 verbose=1)

        self.model.save_weights(weight_file_path)

        np.save(model_dir_path + '/' + WordVecLstmSoftmax.model_name + '-history.npy', history.history)

        score = self.model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
        print('score: ', score[0])
        print('accuracy: ', score[1])

        return history

    def predict(self, sentence):
        xs = []
        tokens = [w.lower() for w in word_tokenize(sentence)]
        wid = [self.word2idx[token] if token in self.word2idx else 1 for token in tokens]
        xs.append(wid)
        x = pad_sequences(xs, self.max_len)
        output = self.model.predict(x)
        return output[0]

    def predict_class(self, sentence):
        predicted = self.predict(sentence)
        idx2label = dict([(idx, label) for label, idx in self.labels.items()])
        return idx2label[np.argmax(predicted)]

    def test_run(self, sentence):
        print(self.predict(sentence))
        
        
class WordVecBidirectionalLstmSoftmax(object):
    model_name = 'bidirectional_lstm_softmax'

    def __init__(self):
        self.model = None
        self.word2idx = None
        self.idx2word = None
        self.max_len = None
        self.config = None
        self.vocab_size = None
        self.labels = None

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecBidirectionalLstmSoftmax.model_name + '_architecture.json'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecBidirectionalLstmSoftmax.model_name + '_weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecBidirectionalLstmSoftmax.model_name + '_config.npy'

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

    def create_model(self):
        embedding_size = 100
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.vocab_size, output_dim=embedding_size, input_length=self.max_len))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(
            Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, input_shape=(self.max_len, embedding_size))))
        self.model.add(Dense(len(self.labels), activation='softmax'))

        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

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

        xs = []
        ys = []
        for text, label in text_label_pairs:
            tokens = [x.lower() for x in word_tokenize(text)]
            wid_list = list()
            for w in tokens:
                wid = 0
                if w in self.word2idx:
                    wid = self.word2idx[w]
                wid_list.append(wid)
            xs.append(wid_list)
            ys.append(self.labels[label])

        X = pad_sequences(xs, maxlen=self.max_len)
        Y = to_categorical(ys, len(self.labels))

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        weight_file_path = self.get_weight_file_path(model_dir_path)

        checkpoint = ModelCheckpoint(weight_file_path)

        history = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data=[x_test, y_test], callbacks=[checkpoint],
                                 verbose=1)

        self.model.save_weights(weight_file_path)

        np.save(model_dir_path + '/' + WordVecBidirectionalLstmSoftmax.model_name + '-history.npy', history.history)

        score = self.model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
        print('score: ', score[0])
        print('accuracy: ', score[1])

        return history

    def predict(self, sentence):
        xs = []
        tokens = [w.lower() for w in word_tokenize(sentence)]
        wid = [self.word2idx[token] if token in self.word2idx else 1 for token in tokens]
        xs.append(wid)
        x = pad_sequences(xs, self.max_len)
        output = self.model.predict(x)
        return output[0]

    def predict_class(self, sentence):
        predicted = self.predict(sentence)
        idx2label = dict([(idx, label) for label, idx in self.labels.items()])
        return idx2label[np.argmax(predicted)]

    def test_run(self, sentence):
        print(self.predict(sentence))


def main():
    app = WordVecLstmSigmoid()
    app.test_run('i liked the Da Vinci Code a lot.')


if __name__ == '__main__':
    main()
