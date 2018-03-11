import re


def extract_email(s, line):
    email = None
    match = re.search(r'[\w\.-]+@[\w\.-]+', line)
    if match is not None:
        email = match.group(0)
    return email


def extract_sex(parts, line):
    sex_found = False
    sex = None
    for w in parts:
        if 'sex' in w:
            sex_found = True
            continue
        if sex_found and ':' not in w:
            if w == 'male':
                sex = 'male'
            else:
                sex = 'female'
            break
    return sex


def extract_education(parts, line):
    found = False
    education = None
    for w in parts:
        if 'education' in w:
            found = True
            continue
        if found and ':' not in w:
            education = w
            break
    return education


def extract_mobile(parts, line):
    found = False
    education = None
    for w in parts:
        if 'mobile' in w:
            found = True
            continue
        if found and ':' not in w:
            education = w
            break
    return education


def extract_experience(parts, line):
    found = False
    result = None
    for w in parts:
        if w.find('experience') != -1:
            found = True
            continue
        if found and ':' not in w:
            result = w
            break
    return result


def extract_expertise(parts, line):
    length = 4
    line = line.lower()
    index = line.find('know')
    if index == -1:
        length = 2
        index = line.find('familiar')
    if index == -1:
        length = 2
        index = line.find('use')
    if index == -1:
        length = 2
        index = line.find('master')
    if index == -1:
        length = 4
        index = line.find('understand')
    if index == -1:
        length = 4
        index = line.find('develop')

    result = None
    if index == -1:
        return None
    else:
        result = line[index + length:].replace(':', '').strip()
        if result == '':
            return None
    return result


def extract_ethnicity(parts, line):
    race_found = False
    race = None
    for w in parts:
        if w.find('race') != -1:
            race_found = True
            continue
        if race_found and w.find(':') == -1:
            race = w
            break
    return race


def extract_name(parts, line):
    found = False
    result = None
    for w in parts:
        if w.find('name') != -1:
            found = True
            continue
        if found and w.find(':') == -1:
            result = w
            break
    return result


def extract_objective(parts, line):
    found = False
    result = None
    for w in parts:
        if w.find('objective') != -1:
            found = True
            continue
        if found and ':' not in w:
            result = w
            break
    return result
