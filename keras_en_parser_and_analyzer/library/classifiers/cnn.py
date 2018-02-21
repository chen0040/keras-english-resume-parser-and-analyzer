from keras.models import Model, model_from_json, Sequential
from keras.layers import Input, SpatialDropout1D, GlobalMaxPool1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from keras_en_parser_and_analyzer.library.utility.tokenizer_utils import word_tokenize


class WordVecCnn(object):
    model_name = 'wordvec_cnn'

    def __init__(self):
        self.model = None
        self.word2idx = None
        self.idx2word = None
        self.max_len = None
        self.config = None
        self.vocab_size = None
        self.labels = None

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecCnn.model_name + '_weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecCnn.model_name + '_config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + WordVecCnn.model_name + '_architecture.json'

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
        self.model.add(Embedding(input_dim=self.vocab_size, input_length=self.max_len, output_dim=embedding_size))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Dense(units=len(self.labels), activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self, sentence):
        xs = []
        tokens = [w.lower() for w in word_tokenize(sentence)]
        wid = [self.word2idx[token] if token in self.word2idx else len(self.word2idx) for token in tokens]
        xs.append(wid)
        x = pad_sequences(xs, self.max_len)
        output = self.model.predict(x)
        return output[0]

    def predict_class(self, sentence):
        predicted = self.predict(sentence)
        idx2label = dict([(idx, label) for label, idx in self.labels.items()])
        return idx2label[np.argmax(predicted)]

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
        Y = np_utils.to_categorical(ys, len(self.labels))

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        weight_file_path = self.get_weight_file_path(model_dir_path)

        checkpoint = ModelCheckpoint(weight_file_path)

        history = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data=[x_test, y_test], callbacks=[checkpoint],
                                 verbose=1)

        self.model.save_weights(weight_file_path)

        np.save(model_dir_path + '/' + WordVecCnn.model_name + '-history.npy', history.history)

        score = self.model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
        print('score: ', score[0])
        print('accuracy: ', score[1])

        return history

    def test_run(self, sentence):
        print(self.predict(sentence))


class WordVecMultiChannelCnn(object):
    model_name = 'wordvec_multi_channel_cnn'

    def __init__(self):
        self.model = None
        self.config = None
        self.word2idx = None
        self.idx2word = None
        self.max_len = None
        self.config = None
        self.vocab_size = None
        self.labels = None

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + os.path.sep + WordVecMultiChannelCnn.model_name + '_weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + os.path.sep + WordVecMultiChannelCnn.model_name + '_config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + os.path.sep + WordVecMultiChannelCnn.model_name + '_architecture.npy'

    def load_model(self, model_dir_path):

        config_file_path = self.get_config_file_path(model_dir_path)

        self.config = np.load(config_file_path).item()

        self.idx2word = self.config['idx2word']
        self.word2idx = self.config['word2idx']
        self.max_len = self.config['max_len']
        self.vocab_size = self.config['vocab_size']
        self.labels = self.config['labels']

        max_input_tokens = len(self.word2idx)
        self.model = self.define_model(self.max_len, max_input_tokens)
        self.model.load_weights(self.get_weight_file_path(model_dir_path))

    def define_model(self, length, vocab_size):

        embedding_size = 100
        cnn_filter_size = 32

        inputs1 = Input(shape=(length,))
        embedding1 = Embedding(vocab_size, embedding_size)(inputs1)
        conv1 = Conv1D(filters=cnn_filter_size, kernel_size=4, activation='relu')(
            embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)

        inputs2 = Input(shape=(length,))
        embedding2 = Embedding(vocab_size, embedding_size)(inputs2)
        conv2 = Conv1D(filters=cnn_filter_size, kernel_size=6, activation='relu')(
            embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)

        inputs3 = Input(shape=(length,))
        embedding3 = Embedding(vocab_size, embedding_size)(inputs3)
        conv3 = Conv1D(filters=cnn_filter_size, kernel_size=8, activation='relu')(
            embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)

        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(10, activation='relu')(merged)

        outputs = Dense(units=len(self.labels), activation='softmax')(dense1)

        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        # compile
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # summarize
        print(model.summary())
        return model

    def fit(self, text_data_model, text_label_pairs, model_dir_path,
            test_size=None, random_state=None,
            epochs=None, batch_size=None):
        if epochs is None:
            epochs = 10
        if batch_size is None:
            batch_size = 16
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

        verbose = 1

        config_file_path = WordVecMultiChannelCnn.get_config_file_path(model_dir_path)
        np.save(config_file_path, text_data_model)

        max_input_tokens = len(self.word2idx)
        self.model = self.define_model(self.max_len, max_input_tokens)
        open(self.get_architecture_file_path(model_dir_path), 'wt').write(self.model.to_json())

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
        Y = np_utils.to_categorical(ys, len(self.labels))

        weight_file_path = WordVecMultiChannelCnn.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)

        history = self.model.fit([X, X, X], Y, epochs=epochs, batch_size=batch_size,
                                 validation_split=test_size,
                                 verbose=verbose, callbacks=[checkpoint])
        # save the model
        self.model.save(weight_file_path)

        np.save(model_dir_path + '/' + WordVecMultiChannelCnn.model_name + '-history.npy', history.history)

        return history

    def predict(self, sentence):
        xs = []
        tokens = [w.lower() for w in word_tokenize(sentence)]
        wid = [self.word2idx[token] if token in self.word2idx else len(self.word2idx) for token in tokens]
        xs.append(wid)
        x = pad_sequences(xs, self.max_len)
        output = self.model.predict([x, x, x])
        return output[0]

    def predict_class(self, sentence):
        predicted = self.predict(sentence)
        idx2label = dict([(idx, label) for label, idx in self.labels.items()])
        return idx2label[np.argmax(predicted)]

    def test_run(self, sentence):
        print(self.predict(sentence))