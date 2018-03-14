import sys
import os


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from keras_en_parser_and_analyzer.library.rule_based_parser import ResumeParser
    from keras_en_parser_and_analyzer.library.utility.io_utils import read_pdf_and_docx

    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir is not '' else '.'
    data_dir_path = current_dir + '/data' # directory to scan for any pdf and docx files

    collected = read_pdf_and_docx(data_dir_path, command_logging=True)
    for file_path, file_content in collected.items():

        print('parsing file: ', file_path)

        parser = ResumeParser()
        parser.parse(file_content)
        print(parser.raw) # print out the raw contents extracted from pdf or docx files

        if parser.unknown is False:
            print(parser.summary())

        print('++++++++++++++++++++++++++++++++++++++++++')

    print('count: ', len(collected))


if __name__ == '__main__':
    main()
