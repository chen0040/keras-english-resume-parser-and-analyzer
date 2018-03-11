from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])


def word_tokenize(text):
    return nltk.word_tokenize(text)


# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


def main():
    trainLines = []
    # create tokenizer
    tokenizer = create_tokenizer(trainLines)
    # calculate max document length
    length = max_length(trainLines)
    # calculate vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Max document length: %d' % length)
    print('Vocabulary size: %d' % vocab_size)
    # encode data
    trainX = encode_text(tokenizer, trainLines, length)
    print(trainX.shape)


if __name__ == '__main__':
    main()
