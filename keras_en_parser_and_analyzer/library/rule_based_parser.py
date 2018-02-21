from keras_en_parser_and_analyzer.library.utility.parser_rules import *
from keras_en_parser_and_analyzer.library.utility.tokenizer_utils import word_tokenize


class ResumeParser(object):

    def __init__(self):
        self.email = None
        self.name = None
        self.sex = None
        self.ethnicity = None
        self.education = None
        self.experience = None
        self.objective = None
        self.mobile = None
        self.expertise = []
        self.unknown = True
        self.raw = None

    def parse(self, texts, print_line=False):
        self.raw = texts
        for p in texts:
            if len(p) > 10:
                s = word_tokenize(p.lower())
                unknown = True
                name = extract_name(s, p)
                email = extract_email(s, p)
                sex = extract_sex(s, p)
                race = extract_ethnicity(s, p)
                education = extract_education(s, p)
                experience = extract_experience(s, p)
                objective = extract_objective(s, p)
                expertise = extract_expertise(s, p)
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
                    self.education = education
                    unknown = False
                if experience is not None:
                    self.experience = experience
                    unknown = False
                if objective is not None:
                    self.objective = objective
                    unknown = False
                if expertise is not None:
                    self.expertise.append(expertise)
                    unknown = False
                if mobile is not None:
                    self.mobile = mobile
                    unknown = False

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
        if self.education is not None:
            text += 'education: {}\n'.format(self.education)
        if self.experience is not None:
            text += 'experience: {}\n'.format(self.experience)
        if self.objective is not None:
            text += 'objective: {}\n'.format(self.objective)
        if len(self.expertise) > 0:
            for ex in self.expertise:
                text += 'expertise: {}\n'.format(ex)

        return text.strip()
