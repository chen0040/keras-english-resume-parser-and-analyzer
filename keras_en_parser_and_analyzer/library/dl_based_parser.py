
from keras_en_parser_and_analyzer.library.classifiers.lstm import WordVecBidirectionalLstmSoftmax
from keras_en_parser_and_analyzer.library.utility.parser_rules import *
from keras_en_parser_and_analyzer.library.utility.simple_data_loader import load_text_label_pairs
from keras_en_parser_and_analyzer.library.utility.text_fit import fit_text
from keras_en_parser_and_analyzer.library.utility.tokenizer_utils import word_tokenize
import os

line_labels = {0: 'experience', 1: 'knowledge', 2: 'education', 3: 'project', 4: 'others'}
line_types = {0: 'header', 1: 'meta', 2: 'content'}


class ResumeParser(object):

    def __init__(self):
        self.line_label_classifier = WordVecBidirectionalLstmSoftmax()
        self.line_type_classifier = WordVecBidirectionalLstmSoftmax()
        self.email = None
        self.name = None
        self.sex = None
        self.ethnicity = None
        self.education = []
        self.objective = None
        self.mobile = None
        self.experience = []
        self.knowledge = []
        self.project = []
        self.meta = list()
        self.header = list()
        self.unknown = True
        self.raw = None

    def load_model(self, model_dir_path):
        self.line_label_classifier.load_model(model_dir_path=os.path.join(model_dir_path, 'line_label'))
        self.line_type_classifier.load_model(model_dir_path=os.path.join(model_dir_path, 'line_type'))

    def fit(self, training_data_dir_path, model_dir_path, batch_size=None, epochs=None,
            test_size=None,
            random_state=None):
        line_label_history = self.fit_line_label(training_data_dir_path, model_dir_path=model_dir_path,
                                                 batch_size=batch_size, epochs=epochs, test_size=test_size,
                                                 random_state=random_state)

        line_type_history = self.fit_line_type(training_data_dir_path, model_dir_path=model_dir_path,
                                               batch_size=batch_size, epochs=epochs, test_size=test_size,
                                               random_state=random_state)

        history = [line_label_history, line_type_history]
        return history

    def fit_line_label(self, training_data_dir_path, model_dir_path, batch_size=None, epochs=None,
                       test_size=None,
                       random_state=None):
        text_data_model = fit_text(training_data_dir_path, label_type='line_label')
        text_label_pairs = load_text_label_pairs(training_data_dir_path, label_type='line_label')

        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 20
        history = self.line_label_classifier.fit(text_data_model=text_data_model,
                                                 model_dir_path=os.path.join(model_dir_path, 'line_label'),
                                                 text_label_pairs=text_label_pairs,
                                                 batch_size=batch_size, epochs=epochs,
                                                 test_size=test_size,
                                                 random_state=random_state)
        return history

    def fit_line_type(self, training_data_dir_path, model_dir_path, batch_size=None, epochs=None,
                      test_size=None,
                      random_state=None):
        text_data_model = fit_text(training_data_dir_path, label_type='line_type')
        text_label_pairs = load_text_label_pairs(training_data_dir_path, label_type='line_type')

        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 20
        history = self.line_label_classifier.fit(text_data_model=text_data_model,
                                                 model_dir_path=os.path.join(model_dir_path, 'line_type'),
                                                 text_label_pairs=text_label_pairs,
                                                 batch_size=batch_size, epochs=epochs,
                                                 test_size=test_size,
                                                 random_state=random_state)
        return history

    @staticmethod
    def extract_education(label, text):
        if label == 'education':
            return text
        return None

    @staticmethod
    def extract_project(label, text):
        if label == 'project':
            return text
        return None

    @staticmethod
    def extract_knowledge(label, text):
        if label == 'knowledge':
            return text
        return None

    @staticmethod
    def extract_experience(label, text):
        if label == 'experience':
            return text
        return None

    def parse(self, texts, print_line=False):
        self.raw = texts
        for p in texts:
            if len(p) > 10:
                s = word_tokenize(p.lower())
                line_label = self.line_label_classifier.predict_class(sentence=p)
                line_type = self.line_type_classifier.predict_class(sentence=p)
                unknown = True
                name = extract_name(s, p)
                email = extract_email(s, p)
                sex = extract_sex(s, p)
                race = extract_ethnicity(s, p)
                education = self.extract_education(line_label, p)
                project = self.extract_project(line_label, p)
                experience = self.extract_experience(line_label, p)
                objective = extract_objective(s, p)
                knowledge = self.extract_knowledge(line_label, p)
                mobile = extract_mobile(s, p)
                if name is not None:
                    self.name = name
                    unknown = False
                if email is not None:
                    self.email = email
                    unknown = False
                if sex is not None:
                    self.sex = sex
                    unknown = False
                if race is not None:
                    self.ethnicity = race
                    unknown = False
                if education is not None:
                    self.education.append(education)
                    unknown = False
                if knowledge is not None:
                    self.knowledge.append(knowledge)
                    unknown = False
                if project is not None:
                    self.project.append(project)
                    unknown = False
                if objective is not None:
                    self.objective = objective
                    unknown = False
                if experience is not None:
                    self.experience.append(experience)
                    unknown = False
                if mobile is not None:
                    self.mobile = mobile
                    unknown = False

                if line_type == 'meta':
                    self.meta.append(p)
                    unknown = False
                if line_type == 'header':
                    self.header.append(p)

                if unknown is False:
                    self.unknown = unknown

                if print_line:
                    print('parsed: ', p)

    def summary(self):
        text = ''
        if self.name is not None:
            text += 'name: {}\n'.format(self.name)
        if self.email is not None:
            text += 'email: {}\n'.format(self.email)
        if self.mobile is not None:
            text += 'mobile: {}\n'.format(self.mobile)
        if self.ethnicity is not None:
            text += 'ethnicity: {}\n'.format(self.ethnicity)
        if self.sex is not None:
            text += 'sex: {}\n'.format(self.sex)
        if self.objective is not None:
            text += 'objective: {}\n'.format(self.objective)

        for ex in self.experience:
            text += 'experience: {}\n'.format(ex)

        for edu in self.education:
            text += 'education: {}\n'.format(edu)

        for knowledge in self.knowledge:
            text += 'knowledge: {}\n'.format(knowledge)
        for project in self.project:
            text += 'project: {}\n'.format(project)

        for meta_data in self.meta:
            text += 'meta: {}\n'.format(meta_data)

        return text.strip()
