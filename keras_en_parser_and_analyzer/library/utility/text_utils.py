def preprocess_text(text):
    text = ' '.join(text.split())
    text = join_name_tag(text)
    return text


def join_name_tag(text):
    text = text.replace('\u2003', '')
    return text


def main():
    print(preprocess_text('name: Xianshun Chen'))


if __name__ == '__main__':
    main()
