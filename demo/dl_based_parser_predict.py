from keras_en_parser_and_analyzer.library.dl_based_parser import ResumeParser
from keras_en_parser_and_analyzer.library.utility.io_utils import read_pdf_and_docx


def main():
    data_dir_path = './data/resume_samples' # directory to scan for any pdf and docx files

    def parse_resume(file_path, file_content):
        print('parsing file: ', file_path)

        parser = ResumeParser()
        parser.load_model('./models')
        parser.parse(file_content)
        print(parser.raw)  # print out the raw contents extracted from pdf or docx files

        if parser.unknown is False:
            print(parser.summary())

        print('++++++++++++++++++++++++++++++++++++++++++')

    collected = read_pdf_and_docx(data_dir_path, command_logging=True, callback=lambda index, file_path, file_content: {
        parse_resume(file_path, file_content)
    })

    print('count: ', len(collected))


if __name__ == '__main__':
    main()
