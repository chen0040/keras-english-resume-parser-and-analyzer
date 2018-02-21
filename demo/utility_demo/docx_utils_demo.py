from keras_en_parser_and_analyzer.library.utility.docx_utils import docx_to_text


def main():
    text = docx_to_text('./data/sample.docx')
    print(text)


if __name__ == '__main__':
    main()
