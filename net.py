# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:35:27 2016

@author: shiwu_001
"""
import os.path as osp
from caffe import Net, TRAIN
from .config import CAFFE_ROOT

DEPLOY_PATH = osp.join(CAFFE_ROOT, "examples/cifar10/resnet20_cifar10_1st_deploy.prototxt")
MODEL_PATH = osp.join(CAFFE_ROOT, "models/resnet_cifar10/resnet20_cifar10_1st_bz128_B_iter_32000.caffemodel")

class MyNet(Net):
    def __init__(self, deploy=None, model=None,
                 phase=TRAIN, pretrained=False):
        if deploy == None:
            deploy = DEPLOY_PATH
        if model == None:
            model = MODEL_PATH
        if pretrained:
            super(MyNet, self).__init__(deploy, model, phase)
        else:
            super(MyNet, self).__init__(deploy, phase)
        self.phase = phase
    
    def set_dataloader(self, dataloader, data_blob='data', label_blob='label'):
        self.dataloader = dataloader
        self.data_blob = data_blob
        self.label_blob = label_blob
        
    def load_data(self, batchid=None):
        self.dataloader.fill_input(net=self, batchid=batchid,
                                   data_blob=self.data_blob,
                                   label_blob=self.label_blob)