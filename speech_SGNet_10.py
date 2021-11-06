# -*- coding:utf-8 -*-
# author:zhuxuechao

"""
语谱图
采样三次
"""
from general_function.plotting import *
from general_function.file_dict import *
from general_function.feature_extract import *
from general_function.edit_distance import *
from random import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization,Dropout,\
    MaxPool2D, Activation,Reshape, Dense, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from read_data2 import DataSpeech
import kenlm
lm_model = kenlm.Model('./5gram.klm')

def GLU(inputs):
    a, b = tf.split(inputs, 2, axis=-1)
    b = tf.nn.sigmoid(b)
    return tf.multiply(a, b)
def conv(inputs):
    dw_conv = tf.keras.layers.SeparableConv1D(
        filters=2*256, kernel_size=32, strides=1,
        padding="same", depth_multiplier=1)(inputs)
    glu = GLU(dw_conv)
    liner = Dense(256)(glu)
    swish = Activation('swish')(liner)
    do = tf.keras.layers.Dropout(0.1)(swish)
    res_add = tf.add(inputs, do)
    return res_add
def Shrinkage(inputs,out_channels,downsample_strides):

    residual = tf.keras.layers.BatchNormalization()(inputs)
    residual = tf.keras.layers.Activation(tf.keras.activations.swish)(residual)
    residual_1 = Conv2D(out_channels, 3, strides=(downsample_strides, downsample_strides),
                        padding='same', kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(1e-4))(residual)

    # Calculate global means
    residual_abs = tf.abs(residual_1)
    abs_mean = tf.keras.layers.GlobalAveragePooling2D()(residual_abs)

    # Calculate scaling coefficients
    d1 = Dense(out_channels, activation=None, kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(1e-4))(abs_mean)
    bn = tf.keras.layers.BatchNormalization()(d1)
    swish = tf.keras.layers.Activation(tf.keras.activations.swish)(bn)
    d2 = Dense(out_channels, activation='sigmoid', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(1e-4))(swish)

    # Calculate thresholds
    thres = tf.keras.layers.multiply([abs_mean, d2])

    # Soft thresholding
    sub = tf.keras.layers.subtract([residual_abs, thres])
    zeros = tf.keras.layers.subtract([sub, sub])
    n_sub = tf.keras.layers.maximum([sub, zeros])
    residual_1 = tf.keras.layers.multiply([tf.sign(residual_1), n_sub])
    return residual_1


def Residual_shrinkage_block(inputs, out_channels, downsample_strides=1, residual_path= False):
    identity = inputs
    residual_1 = Shrinkage(inputs=inputs,out_channels=out_channels,downsample_strides=downsample_strides)
    residual_2 = Shrinkage(inputs=residual_1,out_channels=out_channels,downsample_strides=1)

    # shortcut
    residual_2 = tf.keras.layers.BatchNormalization()(residual_2)
    residual_2 = tf.keras.layers.Activation(tf.keras.activations.swish)(residual_2)
    residual_2 = tf.keras.layers.Conv2D(out_channels, kernel_size=1, padding='same')(residual_2)

    if residual_path:
        identity = Conv2D(out_channels, 1, strides=(downsample_strides, downsample_strides),
                                          padding='same')(inputs)
        identity = tf.keras.layers.BatchNormalization()(identity)
    residual = tf.keras.layers.add([residual_2, identity])
    return residual
def ctc_lambda_func(args):
    """
    CTCLoss Function: tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        :params y_true: 数字标记的tensor
        :params y_pred: 每个frame 各个class的概率
        :params input_length: y_pred的每个sample的序列长度
        :params label_length: y_true的序列长度
    """
    y_true, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

class Amodel():
    def __init__(self, datapath,lm_weight):
        super(Amodel, self).__init__()
        self.vocab_size = 4330
        self.datapath = datapath
        self.lm_weight = lm_weight
        self.model, self.ctc_model = self._model_init()
    def _model_init(self):
        inputs = Input(name='the_inputs', shape=(None, 200, 1))
        x = Conv2D(64, 3, padding='same', activation='swish', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(1e-4))(inputs)
        x = Dropout(0.1)(x)

        # RSCN
        x = Residual_shrinkage_block(x, out_channels=64, downsample_strides=2, residual_path=True)
        x = Dropout(0.1)(x)
        x = Residual_shrinkage_block(x, out_channels=128, downsample_strides=2, residual_path=True)
        x = Dropout(0.1)(x)
        x = Residual_shrinkage_block(x, out_channels=256, downsample_strides=2, residual_path=True)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
        x = Dropout(0.1)(x)
        x = Reshape((-1, 6400))(x)
        x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2())(x)
        x = Dropout(0.1)(x)

        # GCFN
        for i in range(10):
            x = conv(x)


        x = Dense(units=self.vocab_size, use_bias=True, kernel_initializer='he_uniform',
                       kernel_regularizer=regularizers.l2(1e-7))(x)
        outputs = Activation(activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        # self.model.summary()
        labels = Input(name='the_labels', shape=[None], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([labels, outputs, input_length, label_length])
        ctc_model = Model(inputs=[inputs, labels,input_length, label_length], outputs=loss_out)
        # ctc_model.summary()
        opt = Adam(lr=0.000001)
        ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)
        return model, ctc_model

    def test_model(self, datapath='', str_dataset='dev', data_count=1):
        """
        测试函数
        """
        data = DataSpeech(self.datapath, str_dataset)
        num_data = data.get_datanum()
        shuffle_list = [i for i in range(num_data)]
        shuffle(shuffle_list)
        if data_count <= 0 or data_count > num_data:
            data_count = num_data
        try:
            words_num = 0.
            word_error_num = 0.
            for i in range(data_count):
                data_input, data_labels = data.get_data(shuffle_list[i])
                pre = self.predict(data_input=data_input)
                words_n = data_labels.shape[0]
                words_num += words_n
                edit_distance = get_edit_distance(data_labels, pre)
                if edit_distance <= words_n:
                    word_error_num += edit_distance
                else:
                    word_error_num += words_n
            errors = word_error_num / words_num
            print('[*Test Result] Speech Recognition ' + str_dataset + ' set word error ratio : ' + str(
                errors * 100), '%')
            return 1-errors
        except StopIteration:
            print('=======================Error StopIteration 01======================')

    def predict(self, data_input):
        """
        预测函数
        """
        batch_size = 1
        pad_fbank = np.zeros((data_input.shape[0] // 8 * 8 + 8, data_input.shape[1]))
        pad_fbank[:data_input.shape[0], :] = data_input
        pad_wav_len = len(pad_fbank)
        new_lst = np.zeros((1, pad_wav_len, 200, 1), dtype=np.float)
        new_lst[0, :pad_fbank.shape[0], :, 0] = pad_fbank   # 'the_inputs'

        base_pred = self.model.predict(x=new_lst, steps=1)
        base_pred = base_pred[:, :, :]

        in_len = np.zeros((batch_size), dtype=np.int32)
        in_len[0] = pad_wav_len // 8
        r = tf.keras.backend.ctc_decode(base_pred, in_len, greedy=False, beam_width=10, top_paths=10)
        list_symbol_dic = get_vocabs(datapath='aishell_dict.txt')

        final_path = []
        max = -float('inf')
        for i in range(10):
            am_score = r[1][0][i]
            r1 = tf.keras.backend.get_value(r[0][i])
            r1 = r1[0]   # r1:path

            r11 = []
            for i in r1:
                if i != -1:
                    r11.append(i)
            r_str = []
            for i in r11:
                r_str.append(list_symbol_dic[i])

            lm_score = lm_model.score(' '.join(r_str))
            total = am_score+self.lm_weight*lm_score
            if max < total:
                max = total
                final_path = r11
        return final_path

if __name__ == '__main__':
    pass
