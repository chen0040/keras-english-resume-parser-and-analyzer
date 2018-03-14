import numpy as np
import sys
import os


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from keras_en_parser_and_analyzer.library.dl_based_parser import ResumeParser

    random_state = 42
    np.random.seed(random_state)

    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir is not '' else '.'
    output_dir_path = current_dir + '/models'
    training_data_dir_path = current_dir + '/data/training_data'

    classifier = ResumeParser()
    batch_size = 64
    epochs = 20
    history = classifier.fit(training_data_dir_path=training_data_dir_path,
                             model_dir_path=output_dir_path,
                             batch_size=batch_size, epochs=epochs,
                             test_size=0.3,
                             random_state=random_state)


if __name__ == '__main__':
    main()
