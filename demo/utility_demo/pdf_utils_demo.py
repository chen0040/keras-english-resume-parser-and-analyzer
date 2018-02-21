from keras_en_parser_and_analyzer.library.utility.pdf_utils import pdf_to_text


def main():
    text = pdf_to_text('./data/sample.pdf')
    print(text)


if __name__ == '__main__':
    main()
