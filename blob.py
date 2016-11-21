# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:27:02 2016

@author: shiwu_001
"""

from config import DTYPE
import pycuda.gpuarray as garr
from numpy import zeros

class Blob(object):
    def __init__(self, blob, copy=False):
        self.blob_ = blob
        self.hold_ = copy
        self.last_data_cpu_ = True
        self.last_diff_cpu_ = True
        if copy:
            self.data_ = zeros(blob.shape, dtype=DTYPE)
            self.gpu_data_ = garr.zeros(shape=blob.shape, dtype=DTYPE)
            self.diff_ = zeros(blob.shape, dtype=DTYPE)
            self.gpu_diff_ = garr.zeros(shape=blob.shape, dtype=DTYPE)
        else:
            self.data_ = None
            self.gpu_data_ = garr.GPUArray(shape=blob.shape, dtype=DTYPE,
                                           gpudata=blob.gpu_data_ptr)
            self.diff_ = None
            self.gpu_diff_ = garr.GPUArray(shape=blob.shape, dtype=DTYPE,
                                           gpudata=blob.gpu_diff_ptr)
    
    @property
    def data(self):
        if self.hold_:
            if not self.last_data_cpu_:
                self.gpu_data_.get(self.data_)
        else:
            self.data_ = self.blob_.data
        self.last_data_cpu_ = True
        return self.data_
    
    @property
    def gpu_data(self):
        if self.hold_:
            if self.last_data_cpu_:
                self.gpu_data_.set(self.data)
        else:
            if self.last_data_cpu_:
                # call gpu_data to update data on the device
                self.blob_.gpu_data_ptr
        self.last_data_cpu_ = False
        return self.gpu_data_
    
    @property
    def diff(self):
        if self.hold_:
            if not self.last_diff_cpu_:
                self.gpu_diff_.get(self.diff_)
        else:
            self.diff_ = self.blob_.diff
        self.last_diff_cpu_ = True
        return self.diff_
    
    @property
    def gpu_diff(self):
        if self.hold_:
            if self.last_diff_cpu_:
                self.gpu_diff_.set(self.diff)
        else:
            if self.last_diff_cpu_:
                # call gpu_diff to update diff on the device
                self.blob_.gpu_diff_ptr
        self.last_diff_cpu_ = False
        return self.gpu_diff_