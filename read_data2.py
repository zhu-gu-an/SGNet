#-*- coding:utf-8 -*-
# author:zhuxuechao
"""
采样三次:语谱图(200维)
关于构造函数：
①包括类的属性
②在构造函数中，也可以是函数
"""
from random import shuffle
from general_function.file_wav_lst import *
from general_function.feature_extract import *
from general_function.pad_lst import *

class DataSpeech():

    def __init__(self, path, type):
        """
        初始化参数
        path：数据存放位置根目录
        """
        self.datapath = path
        self.type = type

        self.slash = '/'
        if self.slash != self.datapath[-1]:
            self.datapath = self.datapath + self.slash

        self.dic_wavlist = {}
        self.dic_symbollist = {}

        self.symbolnum = 0  # 记录拼音符号数量
        self.list_symbol = self.get_symbollist()  # 全部汉语拼音符号列表

        self.list_wavnum = []  # wav文件标记列表
        self.list_symbolnum = []  # symbol标记列表

        self.datanum = 0  # 记录数据量
        self.load_datalist()
        pass

    def load_datalist(self):
        """
        加载用于计算的数据列表
        参数：
        type：选取的数据集类型
        train 训练集
        dev   验证集
        test  测试集
        """
        if self.type == 'train':
            filename_wavlist = 'data_aishell_word' +self.slash + 'train.wav.txt'
            filename_symbollist ='data_aishell_word' + self.slash + 'train.syllable.txt'
        elif self.type == 'dev':
            filename_wavlist = 'data_aishell_word' + self.slash + 'dev.wav.txt'
            filename_symbollist = 'data_aishell_word' + self.slash + 'dev.syllable.txt'
        elif self.type == 'test':
            filename_wavlist = 'data_aishell_word' + self.slash + 'test.wav.txt'
            filename_symbollist = 'data_aishell_word' + self.slash + 'test.syllable.txt'
        else:
            filename_wavlist = 'test.w.txt'  # 其他，为了可视化
            filename_symbollist = 'test.s.txt'

        # 读取数据列表，wav文件列表和其对应的符号列表
        self.dic_wavlist , self.list_wavnum = get_wav_list(self.datapath + filename_wavlist)
        self.dic_symbollist , self.list_symbolnum = get_wav_symbol(self.datapath + filename_symbollist)
        self.datanum = self.get_datanum()

    def get_datanum(self):
        """
        获取数据的数量
        当wav数量和symbol数量一致的时候返回正确的值，否则返回-1，代表出错。
        """
        num_wavlist = len(self.dic_wavlist)
        num_symbollist = len(self.dic_symbollist)
        if num_wavlist == num_symbollist:
            datanum = num_wavlist
        else:
            datanum = -1
        return datanum

    def get_data(self, n_start):
        """
        读取数据，返回神经网络输入值和输出值矩阵(可直接用于神经网络训练的那种)
        参数：
            n_start：从编号为n_start数据开始选取数据
        返回：
            三个包含wav特征矩阵的神经网络输入值，和一个标定的类别矩阵神经网络输出值
        """
        filename = self.dic_wavlist[self.list_wavnum[n_start]]
        list_symbol = self.dic_symbollist[self.list_symbolnum[n_start]]

        feat_out = []
        for i in list_symbol:
            n = self.symbol_to_num(i)
            feat_out.append(n)  # 将字符转为数值向量
        data_input = compute_fbank(filename)
        data_label = np.array(feat_out)
        return data_input, data_label

    def data_generator(self, batch_size=8):
        """
        数据生成器函数
        batch_size:一次喂进网络的数据数量；
        shuffle_list:根据打乱数据的顺序查找训练音频的索引；
        """
        datanum = self.get_datanum()
        shuffle_list = [i for i in range(datanum)]
        shuffle(shuffle_list)
        for i in range(datanum // batch_size):
            wav_data_lst = []
            label_data_lst = []
            begin = i * batch_size
            end = begin + batch_size
            sub_list = shuffle_list[begin:end]
            # 将batch_size的信号时频图和标签数据，存放到两个list中去
            for index in sub_list:
                fbank, label = self.get_data(index)
                pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
                pad_fbank[:fbank.shape[0], :] = fbank
                wav_data_lst.append(pad_fbank)
                label_data_lst.append(label)

            pad_wav_data, input_length= wav_padding(wav_data_lst=wav_data_lst)
            pad_label_data, label_length = label_padding(label_data_lst)

            inputs = {'the_inputs': pad_wav_data,
                      'the_labels': pad_label_data,
                      'input_length': input_length,
                      'label_length': label_length,

                      }
            outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
            yield inputs, outputs

    def get_symbollist(self):
        """
        加载拼音字典列表，用于标记拼音
        返回一个列表list类型变量
        """
        list_symbol = []
        with open('aishell_dict.txt', 'r', encoding="utf-8") as fr:
            lines = fr.readlines()
            for line in lines:
                res = line.split()
                list_symbol.append(res[0])
        fr.close()
        list_symbol.append('_')
        self.symbolnum = len(list_symbol)
        return list_symbol

    def symbol_to_num(self, symbol):
        """
        拼音转为数字
        """
        if symbol != '':
            return self.list_symbol.index(symbol)
        else:
            return self.symbolnum

    def get_symbol_num(self):
        """
        获取拼音字典的数量
        """
        return len(self.list_symbol)

if __name__ == '__main__':
    datapath = './visualization_data'
    Data = DataSpeech(path=datapath, type='other')
    a, b = Data.get_data(0)
    import time
    # t = time.time()
    # a = Data.data_generator(batch_size=1)

    # print(c, d)
    # s = time.time()
    # a = Data.get_datanum()
    # a = Data.get_symbollist()
    # print(s-t)
    print(a)







