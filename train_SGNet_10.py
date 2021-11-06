# -*- coding:utf-8 -*-
# author:zhuxuechao

from speech_SGNet_10 import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

modelpath = './speech_model/'
datapath = './data_list'

speech = Amodel(datapath=datapath)
# speech.ctc_model.load_weights(modelpath + 'speech_model_SGNet_10_e_26_step_model.h5')
speech.train_model(datapath=datapath, epoch=200, batch_size=8)
