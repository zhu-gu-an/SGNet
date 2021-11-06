#-*- coding:utf-8 -*-
#author:zhuxuechao

'''
此函数是用作于加载字典里面的符号，用于声学模型的训练以及语言模型的训练；
'''

def get_vocabs(datapath='/root/data/asr/dict.txt'):
    vocabs = []
    with open(datapath, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            vocab = line.split()
            vocabs.append(vocab[0])
        vocabs.append('_')
    return vocabs
def get_vocabs_size(datapath):
    vocabs_size = len(get_vocabs(datapath))
    return vocabs_size

if __name__ == '__main__':
    datapath = '/root/data/asr/dict.txt'
    path = '/root/data/asr/data_list/st-cmds/train.syllable.txt'
    vv = get_vocabs(datapath)
    print(len(vv))







