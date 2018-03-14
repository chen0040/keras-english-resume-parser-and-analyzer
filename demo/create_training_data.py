import os
import sys
from tkinter import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from keras_en_parser_and_analyzer.library.dl_based_parser import line_types, line_labels
from keras_en_parser_and_analyzer.library.utility.io_utils import read_pdf_and_docx


class AnnotatorGui(Frame):
    def __init__(self, master, table_content):
        Frame.__init__(self, master=master)

        self.master.title("Annotate Resume Lines")

        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.grid(sticky=W + E + N + S)

        self.line_index_label_list = []
        self.line_content_text_list = []
        self.line_type_button_list = []
        self.line_label_button_list = []

        for line_index, line in enumerate(table_content):
            self.build_line(table_content, line_index, line)

        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)

    def build_line(self, table_content, line_index, line):
        line_content = line[0]

        line_index_label = Label(self, width=10, height=1, text=str(line_index))
        line_index_label.grid(row=line_index, column=0, sticky=W + E + N + S)
        line_content_text = Text(self, width=100, height=1)
        line_content_text.insert(INSERT, line_content)
        line_content_text.grid(row=line_index, column=1, sticky=W + E + N + S)

        def line_type_button_click(_line_index):
            line_type = table_content[_line_index][1]
            line_type = (line_type + 1) % len(line_types)
            table_content[_line_index][1] = line_type
            line_type_button["text"] = "Type: " + line_types[line_type]

        def line_label_button_click(_line_index):
            line_label = table_content[_line_index][2]
            line_label = (line_label + 1) % len(line_labels)
            table_content[_line_index][2] = line_label
            line_label_button["text"] = "Type: " + line_labels[line_label]

        line_type_button = Button(self, text="Type: Unknown", width=25,
                                  command=lambda: line_type_button_click(line_index))
        line_type_button.grid(row=line_index, column=2, sticky=W + E + N + S)
        line_label_button = Button(self, text='Label: Unknown', width=25,
                                   command=lambda: line_label_button_click(line_index))
        line_label_button.grid(row=line_index, column=3, sticky=W + E + N + S)

        if line[1] != -1:
            line_type_button["text"] = "Type: " + line_types[line[1]]
        if line[2] != -1:
            line_label_button["text"] = "Type: " + line_labels[line[2]]


def command_line_annotate(training_data_dir_path, index, file_path, file_content):
    with open(os.path.join(training_data_dir_path, str(index) + '.txt'), 'wt', encoding='utf8') as f:
        for line_index, line in enumerate(file_content):
            print('Line #' + str(line_index) + ': ', line)
            data_type = input('Type for line #' + str(line_index) + ' (options: 0=header 1=meta 2=content):')
            label = input('Label for line #' + str(line_index) +
                          ' (options: 0=experience 1=knowledge 2=education 3=project 4=others')
            data_type = int(data_type)
            label = int(label)
            f.write(line_types[data_type] + '\t' + line_labels[label] + '\t' + line)
            f.write('\n')


def guess_line_type(line):
    return -1


def guess_line_label(line):
    return -1


def gui_annotate(training_data_dir_path, index, file_path, file_content):
    root = Tk()
    table_content = [[line, guess_line_type(line), guess_line_label(line)] for line in file_content]
    gui = AnnotatorGui(root, table_content)

    def callback():
        root.destroy()
        output_file_path = os.path.join(training_data_dir_path, str(index) + '.txt')
        if os.path.exists(output_file_path):
            return
        with open(output_file_path, 'wt', encoding='utf8') as f:
            for line in table_content:
                line_content = line[0]
                data_type = line[1]
                label = line[2]

                if data_type == -1 or label == -1:
                    continue

                print('write line: ', line)
                f.write(line_types[data_type] + '\t' + line_labels[label] + '\t' + line_content)
                f.write('\n')

    root.protocol("WM_DELETE_WINDOW", callback)
    gui.mainloop()


def main():
    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir is not '' else '.'

    data_dir_path = current_dir + '/data'  # directory to scan for any pdf files
    training_data_dir_path = current_dir + '/data/training_data'
    collected = read_pdf_and_docx(data_dir_path, command_logging=True, callback=lambda index, file_path, file_content: {
        gui_annotate(training_data_dir_path, index, file_path, file_content)
    })

    print('count: ', len(collected))


if __name__ == '__main__':
    main()
