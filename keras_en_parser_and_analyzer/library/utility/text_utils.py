def preprocess_text(text):
    text = ' '.join(text.split())
    text = join_name_tag(text)
    return text


def join_name_tag(text):
    text = text.replace('\u2003', '')\
        .replace('姓 名', '姓名').replace('专 业', '专业').replace('手 机', '手机')\
        .replace('学 历', '学历').replace('邮 箱', '邮箱').replace('性 别', '性别').replace('民 族', '民族')
    return text


def main():
    print(preprocess_text('姓    名： 牛冠群'))


if __name__ == '__main__':
    main()
