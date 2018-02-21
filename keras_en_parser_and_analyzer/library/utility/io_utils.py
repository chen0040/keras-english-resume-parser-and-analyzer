import os
from keras_en_parser_and_analyzer.library.utility.pdf_utils import pdf_to_text
from keras_en_parser_and_analyzer.library.utility.docx_utils import docx_to_text


def read_pdf_and_docx(dir_path, collected=None, command_logging=False, callback=None):
    if collected is None:
        collected = dict()
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        if os.path.isfile(file_path):
            txt = None
            if f.lower().endswith('.docx'):
                if command_logging:
                    print('extracting text from docx: ', file_path)
                txt = docx_to_text(file_path)
            elif f.lower().endswith('.pdf'):
                if command_logging:
                    print('extracting text from pdf: ', file_path)
                txt = pdf_to_text(file_path)
            if txt is not None and len(txt) > 0:
                if callback is not None:
                    callback(len(collected), file_path, txt)
                collected[file_path] = txt
        elif os.path.isdir(file_path):
            read_pdf_and_docx(file_path, collected, command_logging, callback)

    return collected


def read_pdf(dir_path, collected=None):
    if collected is None:
        collected = dict()
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        if os.path.isfile(file_path):
            txt = None
            if f.lower().endswith('.pdf'):
                txt = pdf_to_text(file_path)
            if txt is not None and len(txt) > 0:
                collected[file_path] = txt
        elif os.path.isdir(file_path):
            read_pdf(file_path, collected)

    return collected
