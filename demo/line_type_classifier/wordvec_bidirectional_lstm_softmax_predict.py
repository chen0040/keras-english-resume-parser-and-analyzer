from random import shuffle

import numpy as np

from keras_en_parser_and_analyzer.library.classifiers.lstm import WordVecBidirectionalLstmSoftmax
from keras_en_parser_and_analyzer.library.utility.simple_data_loader import load_text_label_pairs


def main():
    random_state = 42
    np.random.seed(random_state)

    model_dir_path = './models'
    data_file_path = '../data/training_data'
    text_label_pairs = load_text_label_pairs(data_file_path)

    classifier = WordVecBidirectionalLstmSoftmax()
    classifier.load_model(model_dir_path=model_dir_path)

    shuffle(text_label_pairs)

    for i in range(20):
        text, label = text_label_pairs[i]
        print('Output: ', classifier.predict(sentence=text))
        predicted_label = classifier.predict_class(text)
        print('Sentence: ', text)
        print('Predicted: ', predicted_label, 'Actual: ', label)


if __name__ == '__main__':
    main()
