#-*- coding:utf-8 -*-
#author:zhuxuechao

'''
此函数是为了提取语音的文件列表以及文件字典的脚本，为后续的文件处理做准备；
'''
def get_wav_list(filename):
    dic_filelist = {}
    list_wavmark = []
    with open(filename, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            res = line.split()
            dic_filelist[res[0]] = res[1]
            list_wavmark.append(res[0])
    return dic_filelist, list_wavmark

def get_wav_symbol(filename):
    dic_symbollist = {}
    list_symbolmark = []
    with open(filename, 'r',encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            res = line.split()
            dic_symbollist[res[0]] = res[1:]
            list_symbolmark.append(res[0])
    return dic_symbollist, list_symbolmark

if __name__ == '__main__':
    filename = '/root/data/asr/data_list/thchs30/train.syllable.txt'
    a, b = get_wav_list(filename)
    c, d = get_wav_symbol(filename)
    print(a['B8_439'])
    print(c['B8_439'])

