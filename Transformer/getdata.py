import re
from sklearn.model_selection import train_test_split

def get_raw_file(txt_dir):
    lines = [line.rstrip() for line in open(txt_dir,encoding='utf-8')]
    en = []
    zh = []
    for i in range(len(lines)):
        value = re.split(r'\t', lines[i])
        if len(value)<2:
            continue
        en.append(value[0])
        zh.append(value[1])
    return en, zh

def split(savefile, srcdata, trgdata):
    with open(savefile, "w", encoding='utf-8') as f:
        for i in range(len(srcdata)):
            if len(srcdata[i])<2 or len(trgdata[i])<2:
                continue
            f.write(srcdata[i]+'\t'+trgdata[i]+'\n')

     
def main():
    raw_en, raw_zh = get_raw_file('news-commentary-v14.en-zh.tsv')
    train_src, test_src, train_trg, test_trg = train_test_split(raw_en, raw_zh, test_size=0.05, random_state=0)
    train_src, valid_src, train_trg, valid_trg = train_test_split(train_src, train_trg, test_size=0.05, random_state=0)
    print(len(train_src),len(valid_src),len(test_src))
    split('data/train.txt', train_src, train_trg)
    split('data/valid.txt', valid_src, valid_trg)
    split('data/test.txt', test_src, test_trg)


if __name__ == '__main__':
    main()