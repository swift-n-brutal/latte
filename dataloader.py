# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 17:25:55 2016

@author: shiwu_001
"""

import caffe
import numpy as np
import lmdb

class DataLoader(object):
    def __init__(self):
        self.mean = np.zeros(3)
        self.std = np.ones(3)
        self.nimages = 0
        self.key_length = 5
        self.data = None
        self.label = None
        self.transformer = None
    
    def __getitem__(self, key):
        if key in ['data', 'label', 'transformer', 'mean', 'std']:
            return eval('self.%s' % key)
        else:
            raise Exception('Invalid key: %s' % key)
            return None
    
    def __setitem__(self, key, item):
        if key in ['data', 'label', 'transformer', 'mean', 'std']:
            exec('self.%s = item' % key)
        else:
            raise Exception('Invalid key: %s' % key)
        
    def _init_loader(self, path):
        env = lmdb.open(path, readonly=True)
        self.nimages = env.stat()['entries']
        print "Load %d images from %s" % (self.nimages, path)
        txn = env.begin()
        return txn
        
    def _load_image(self, txn, index, key_length):
        raw_datum = txn.get(eval("'%%0%dd' %% index" % key_length))
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)
        flat_x = np.fromstring(datum.data, dtype=np.uint8)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        y = datum.label
        return x,y
    
    def _load_batch(self, txn, batch, dest_x, dest_y):
        for i,index in enumerate(batch):
            x,y = self._load_image(txn, index, self.key_length)
            dest_x[i,...] = x
            dest_y[i] = y

    def compute_meanstd(data, verbose=False):
        print "Computing mean"
        mean = np.mean(data, axis=(0,2,3))
        print "Computing std"
        std = np.std(data, axis=(0,2,3))
        return mean, std
    
    def load_dataset(self, path, recompute=False, transform=False):
        self.path = path
        txn = self._init_loader(path)
        # get the shape
        x,y = self._load_image(txn, 0, self.key_length)
        c,h,w = x.shape
        data = np.zeros((self.nimages, c, h, w))
        label = np.zeros((self.nimages))
        batch = np.arange(self.nimages)
        self._load_batch(txn, batch, data, label)
        self.data = data
        self.label = label
        if recompute:
            self.mean, self.std = self.compute_meanstd(self.data, verbose=True)
            print "Mean", self.mean
            print "Std", self.std
        if transform:
            self.transform_dataset(self)
    
    def transform_dataset(self, dataset, meanstd=None):
        if meanstd is None:
            dataset['data'] -= self.mean.reshape(1,3,1,1)
            dataset['data'] /= self.std.reshape(1,3,1,1)
        else:
            dataset['data'] -= meanstd['mean']
            dataset['data'] /= meanstd['std']

class CifarDataLoader(DataLoader):
    def __init__(self, path, net, phase, data_blob='data'):
        super(CifarDataLoader, self).__init__()
        self.mean = np.array([125.3, 123.0, 113.9])
        self.std = np.array([63.0, 62.1, 66.7])
        self.key_length = 5
        if phase == caffe.TRAIN:
            self.load_dataset(path=path,
                              transform=True)
            self.transformer = CifarTransformer({
                data_blob: net.blobs[data_blob].data.shape})
            self.transformer.set_pad(data_blob, 4)
            self.transformer.set_mirror(data_blob, True)
        elif phase == caffe.TEST:
            self.load_dataset(path=path,
                              transform=True)
        else:
            raise Exception("Invalid phase: %s" % str(phase))
        
    def _load_batch_from_dataset(self, batchid, dest_x, dest_y, data_blob):
        if self.transformer is None:
            for i,bid in enumerate(batchid):
                dest_x[i,...] = self.data[bid,...]
                if dest_y is not None:
                    dest_y[i] = self.label[bid]
        else:
            for i,bid in enumerate(batchid):
                dest_x[i,...] = self.transformer.process(
                    data_blob, self.data[bid,...])
                if dest_y is not None:
                    dest_y[i] = self.label[bid]
        
    def sample_batch(self, batchsize):
        return np.random.randint(self.nimages, size=batchsize)
    
    def fill_input(self, net, batchid=None,
                   data_blob='data', label_blob='label'):
        batchsize = net.blobs[data_blob].num
        if batchid is None:
            batchid = self.sample_batch(batchsize)
        else:
            assert(batchsize == len(batchid))
        if label_blob is not None:
            self._load_batch_from_dataset(batchid, net.blobs[data_blob].data,
                                          net.blobs[label_blob].data,
                                          data_blob)
        else:
            self._load_batch_from_dataset(batchid, net.blobs[data_blob].data,
                                          None, data_blob)

class CifarTransformer(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self.pad = {}
        self.pad_value = {}
        self.mean = {}
        self.std = {}
        self.mirror = {}
        self.center = {}
        
    def __check_input(self, in_):
        if in_ not in self.inputs:
            raise Exception("{} is not one of the net inputs: {}".format(
                in_, self.inputs))
    
    def process(self, in_, data):
        self.__check_input(in_)
        data_in = np.copy(data).astype(np.float32)
        mean = self.mean.get(in_)
        std = self.std.get(in_)
        pad = self.pad.get(in_)
        pad_value = self.pad_value.get(in_)
        mirror = self.mirror.get(in_)
        center = self.center.get(in_)
        in_dims = self.inputs[in_][2:]
        if mean is not None:
            data_in -= mean
        if std is not None:
            data_in /= std
        if pad is not None:
            if pad_value is None:
                pad_value = 0
            data_in = np.pad(data_in, ((0,0), (pad,pad), (pad,pad)),
                             'constant', constant_values=pad_value)
        if data_in.shape[1] >= in_dims[0] and data_in.shape[2] >= in_dims[1]:
            if center is not None and center:
                h_off = int((data_in.shape[1] - in_dims[0]+1) / 2)
                w_off = int((data_in.shape[2] - in_dims[1]+1) / 2)
            else:
                h_off = np.random.randint(data_in.shape[1] - in_dims[0]+1)
                w_off = np.random.randint(data_in.shape[2] - in_dims[1]+1)
            data_in = data_in[:,h_off:h_off+in_dims[0],
                              w_off:w_off+in_dims[1]]
        else:
            print 'Image is smaller than input: (%d,%d) vs (%d,%d)' \
                % (data_in.shape[1],data_in.shape[2], in_dims[0],in_dims[1])
        if mirror is not None and mirror and np.random.randint(2) == 1:
            data_in = data_in[:,:,::-1]
        return data_in
    
    def set_mean(self, in_, mean):
        self.__check_input(in_)
        ms = mean.shape
        if mean.ndim == 1:
            # broadcast channels
            if ms[0] != self.inputs[in_][1]:
                raise ValueError('Mean channels incompatible with input.')
            mean = mean[:, np.newaxis, np.newaxis]
        else:
            # elementwise mean
            if len(ms) == 2:
                ms = (1,) + ms
            if len(ms) != 3:
                raise ValueError('Mean shape invalid')
            if ms != self.inputs[in_][1:]:
                raise ValueError('Mean shape incompatible with input shape.')
        self.mean[in_] = mean

    def set_std(self, in_, std):
        self.__check_input(in_)
        ss = std.shape
        if std.ndim == 1:
            # broadcast channels
            if ss[0] != self.inputs[in_][1]:
                raise ValueError('Std channels incompatible with input.')
            std = std[:, np.newaxis, np.newaxis]
        else:
            # elementwise mean
            if len(ss) == 2:
                ss = (1,) + ss
            if len(ss) != 3:
                raise ValueError('Std shape invalid')
            if ss != self.inputs[in_][1:]:
                raise ValueError('Std shape incompatible with input shape.')
        self.std[in_] = std
        
    def set_pad(self, in_, pad):
        self.__check_input(in_)
        self.pad[in_] = pad
        
    def set_pad_value(self, in_, pad_value):
        self.__check_input(in_)
        self.pad_value[in_] = pad_value
    
    def set_mirror(self, in_, mirror):
        self.__check_input(in_)
        self.mirror[in_] = mirror

    def set_center(self, in_, center):
        self.__check_input(in_)
        self.center[in_] = center

if __name__ == '__main__':
    a = np.random.randn(1,2,3,4)
    tf = CifarTransformer({'data': a.shape})