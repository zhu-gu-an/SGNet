#-*- coding:utf-8 -*-
import numpy as np

def wav_padding(wav_data_lst):
    """
    wav_data_lst:batch_size的wav列表。
    当pad之后，取最大的为该batch的帧数；
    """
    wav_lens = [len(data) for data in wav_data_lst]
    wav_max_len = max(wav_lens)
    wav_lens = np.array([leng//8 for leng in wav_lens])
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))  # (batch_size,最大帧数,200,1)
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]

    return new_wav_data_lst, wav_lens

def label_padding(label_data_lst):
    """
    label_data_lst:batch_size的label标签。
    当pad之后，取最大的为该batch的长度；
    """
    label_lens = np.array([len(label) for label in label_data_lst])
    max_label_len = max(label_lens)
    new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
    for i in range(len(label_data_lst)):
        new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
    return new_label_data_lst, label_lens
