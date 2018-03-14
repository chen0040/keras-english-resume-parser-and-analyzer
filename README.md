# keras-english-resume-parser-and-analyzer

Deep learning project that parses and analyze english resumes.

The objective of this project is to use Keras and Deep Learning such as CNN and recurrent neural network to automate the
task of parsing a english resume. 


# Overview

### Parser Features 

* English NLP using NLTK
* Extract english texts using pdfminer.six and python-docx from PDF nad DOCX
* Rule-based resume parser has been implemented.

### Deep Learning Features

* Tkinter-based GUI tool to generate and annotate deep learning training data from pdf and docx files
* Deep learning multi-class classification using recurrent and cnn networks for
    * line type: classify each line of text extracted from pdf and docx file on whether it is a header, meta-data, or content
    * line label classify each line of text extracted from pdf and docx file on whether it implies experience, education, etc.
    
The included deep learning models that classify each line in the resume files include:

* [cnn.py](keras_en_parser_and_analyzer/library/classifiers/cnn.py)
    * 1-D CNN with Word Embedding 
    * Multi-Channel CNN with categorical cross-entropy loss function

* [cnn_lstm.py](keras_en_parser_and_analyzer/library/classifiers/cnn_lstm.py)
    * 1-D CNN + LSTM with Word Embedding

* [lstm.py](keras_en_parser_and_analyzer/library/classifiers/lstm.py)
    * LSTM with category cross-entropy loss function
    * Bi-directional LSTM/GRU with categorical cross-entropy loss function
    
# Usage 1: Rule-based English Resume Parser

The [sample code](demo/rule_base_parser.py) below shows how to scan all the resumes (in PDF and DOCX formats) from a 
[demo/data/resume_samples] folder and print out a summary from the resume parser if information extracted are available:

```python
from keras_en_parser_and_analyzer.library.rule_based_parser import ResumeParser
from keras_en_parser_and_analyzer.library.utility.io_utils import read_pdf_and_docx


def main():
    data_dir_path = './data/resume_samples' # directory to scan for any pdf and docx files
    collected = read_pdf_and_docx(data_dir_path)
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

```

IMPORTANT: the parser rules are implemented in the [parser_rules.py](keras_en_parser_and_analyzer/library/utility/parser_rules.py).
Each of these rules will be applied to every line of text in the resume file and return the target accordingly (or
return None if not found in a line). As these rules are very naive implementation, you may want to customize them further based on the resumes that you
are working with.

# Usage 2: Deep Learning Resume Parser

### Step 1: training data generation and annotation

A training data generation and annotation tool is created in the [demo](demo) folder which allows 
resume deep learning training data to be generated from any pdf and docx files stored in the 
[demo/data/resume_samples](demo/data/resume_samples) folder, To launch this tool, run the following 
command from the root directory of the project:

```batch
cd demo
python create_training_data.py
``` 

This will parse the pdf and docx files in [demo/data/resume_samples](demo/data/resume_samples) folder
and for each of these file launch a Tkinter-based GUI form to user to annotate individual text line
in the pdf or docx file (clicking the "Type: ..." and "Label: ..." buttons multiple time to select the 
correct annotation for each line). On each form closing, the generated and annotated data will be saved
to a text file in the [demo/data/training_data](demo/data/training_data) folder.  each line in the
text file will have the following format

```text
line_type   line_label  line_content
```

line_type and line_label has the following mapping to the actual class labels

```python
line_labels = {0: 'experience', 1: 'knowledge', 2: 'education', 3: 'project', 4: 'others'}
line_types = {0: 'header', 1: 'meta', 2: 'content'}
```

### Step 2: train the resume parser

After the training data is generated and annotated, one can train the resume parser by running the following
command:

```bash
cd demo
python dl_based_parser_train.py
```

Below is the code for [dl_based_parser_train.py](demo/dl_based_parser_train.py):

```python
import numpy as np
import os
import sys 


def main():
    random_state = 42
    np.random.seed(random_state)

    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir is not '' else '.'
    output_dir_path = current_dir + '/models'
    training_data_dir_path = current_dir + '/data/training_data'
    
    # add keras_en_parser_and_analyzer module to the system path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from keras_en_parser_and_analyzer.library.dl_based_parser import ResumeParser

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

```

Upon completion of training, the trained models will be saved in the [demo/models/line_label](demo/models/line_label)
and [demo/models/line_type](demo/models/line_type) folders

The default line label and line type classifier used in the deep learning ResumeParser is 
[WordVecBidirectionalLstmSoftmax](keras_en_parser_and_analyzer/library/classifiers/lstm.py).
But other classifiers can be used by adding the following line, for example:

```python
from keras_en_parser_and_analyzer.library.dl_based_parser import ResumeParser
from keras_en_parser_and_analyzer.library.classifiers.cnn_lstm import WordVecCnnLstm

classifier = ResumeParser()
classifier.line_label_classifier = WordVecCnnLstm()
classifier.line_type_classifier = WordVecCnnLstm()
...
```

(Do make sure that the requirements.txt are satisfied in your python env)

### Step 3: parse resumes using trained parser

After the trained models are saved in the [demo/models](demo/models) folder,
one can use the resume parser to parse the resumes in the [demo/data/resume_samples](demo/data/resume_samples)
by running the following command:

```bash
cd demo
python dl_based_parser_predict.py
```

Below is the code for [dl_based_parser_predict.py](demo/dl_based_parser_predict.py):

```python
import os
import sys 


def main():
    current_dir = os.path.dirname(__file__)
    current_dir = current_dir if current_dir is not '' else '.'
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from keras_en_parser_and_analyzer.library.dl_based_parser import ResumeParser
    from keras_en_parser_and_analyzer.library.utility.io_utils import read_pdf_and_docx
    
    data_dir_path = current_dir + '/data/resume_samples' # directory to scan for any pdf and docx files

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

```

# Configure to run on GPU on Windows

* Step 1: Change tensorflow to tensorflow-gpu in requirements.txt and install tensorflow-gpu
* Step 2: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Please note that
currently CUDA® Toolkit 9.1 is not yet supported by tensorflow, therefore you should download CUDA® Toolkit 9.0)
* Step 3: Download and unzip the [cuDNN 7.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 
