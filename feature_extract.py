#-*- coding:utf-8 -*-
# author:zhuxuechao
from scipy.fftpack import fft
from general_function.add_noise_mode import *
import numpy as np
from python_speech_features import *
import scipy.io.wavfile
import matplotlib.pyplot as plt
"""
时域图谱:
分帧(25ms/10ms)-加窗(汉明窗)-fft(快速傅里叶变换)-log 
Fbank：
分帧(25ms/10ms)-加窗(汉明窗)-fft(快速傅里叶变换)-mel滤波器(40维)-log(余弦变换)
MFCC：
分帧(25ms/10ms)-加窗(汉明窗)-fft(快速傅里叶变换)-mel滤波器(26维+13维)-log-DCT(余弦变换)
"""

def compute_fbank(file, noise=False):

    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    fs, wavsignal = scipy.io.wavfile.read(file)
    if noise:
        wavsignal = augment(wavsignal, './noise', SNR=[-5, 15])  # 加噪

    time_window = 25  # 单位ms
    wav_arr = np.array(wavsignal)
    range0_end = int(len(wavsignal)/fs*1000 - time_window) // 10  # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype=np.float)
    for i in range(0, range0_end):
        p_start = i * 160   # 160表示帧移
        p_end = p_start + 400  # 400表示帧长
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    return data_input

def get_fbank(path):
    fs,signal = scipy.io.wavfile.read(path)
    fbank = logfbank(signal, fs, winlen=0.025, winstep=0.01, nfilt=40, nfft=512, lowfreq=0, highfreq=None, preemph=0.97)
    return fbank

def get_mfcc(path):
    fs, signal = scipy.io.wavfile.read(path)
    wav_feature = mfcc(signal, fs, numcep=13, winlen=0.025, winstep=0.01,
                       nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97)
    d_mfcc_feat = delta(wav_feature, 1)
    d_mfcc_feat2 = delta(wav_feature, 2)
    feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
    return feature

if __name__ == '__main__':
    file = 'D:\\asr-aishell\\data_aishell/train/S0002/BAC009S0002W0122.wav'
    import time
    t = time.time()
    a = compute_fbank(file, noise=True)
    b = compute_fbank(file)
    s = time.time()
    print(a.shape)

    plt.imshow(a.T, origin="lower")
    plt.show()
    plt.imshow(b.T, origin="lower")
    plt.show()
    # t = time.time()
    # b = get_fbank(file)
    # s = time.time()
    # print(b.shape, s-t)
    # plt.imshow(b.T, origin="lower")
    # plt.show()
    # t = time.time()
    # c = get_mfcc(file)
    # s = time.time()
    # print(c.shape, s-t)
    # plt.imshow(c.T, origin="lower")
    # plt.show()
