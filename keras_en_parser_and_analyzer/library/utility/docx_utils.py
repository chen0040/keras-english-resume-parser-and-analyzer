import docx

from keras_en_parser_and_analyzer.library.utility.text_utils import preprocess_text


def docx_to_text(file_path):
    doc = docx.Document(file_path)
    result = []
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt != '':
            txt = preprocess_text(txt)
            result.append(txt)
    return result
