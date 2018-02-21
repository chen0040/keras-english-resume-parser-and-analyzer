import os
import numpy as np
from matplotlib import pyplot as plt


def main():
    model_dir_path = './models'
    models = ['lstm_softmax', 'bidirectional_lstm_softmax', 'wordvec_cnn', 'wordvec_multi_channel_cnn', 'wordvec_cnn_lstm']
    acc_cmp = dict()
    val_acc_cmp = dict()
    labels = list()
    for model_name in models:
        acc_cmp[model_name] = list()
        val_acc_cmp[model_name] = list()
        history_file_name = model_name + '-history.npy'
        history_file_path = os.path.join(model_dir_path, history_file_name)
        history = np.load(history_file_path).item()
        labels_not_set = len(labels) == 0
        for index in range(0, 20, 2):
            acc_data = history['acc']
            val_acc_data = history['val_acc']
            epoch = min(len(acc_data)-1, index)
            acc = acc_data[epoch]
            val_acc = val_acc_data[epoch]
            acc_cmp[model_name].append(acc)
            val_acc_cmp[model_name].append(val_acc)
            if labels_not_set:
                labels.append(epoch)

    plt.subplot(211)
    plt.title('Training Accuracy')
    for model_name, acc_data in acc_cmp.items():
        plt.plot(labels, acc_data, label=model_name)
    plt.legend(loc='best')

    plt.subplot(212)
    plt.title('Validation Accuracy')
    for model_name, acc_data in val_acc_cmp.items():
        plt.plot(labels, acc_data, label=model_name)
    plt.legend(loc='best')

    plt.xlabel('training epoch')

    plt.tight_layout()

    file_path = os.path.join(model_dir_path, 'training-history-comparison.png')
    plt.savefig(file_path)

    plt.show()


if __name__ == '__main__':
    main()
