# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:23:50 2016

@author: shiwu_001
"""

import sys
import os.path as osp
CAFFE_ROOT ='../../..'
sys.path.insert(0, osp.join(CAFFE_ROOT, 'examples'))
import latte
import caffe
import time
import numpy as np

def get_blobs_act(net):
    act = list()
    for i,l in enumerate(net.layers):
        if l.type == "ReLU":
            blob_ids = net._top_ids(i)
            blob_name = net._blob_names[blob_ids[0]]
            act.append(np.copy(net.blobs[blob_name].data))
    return act

def get_blobs_sgn_from_act(act):
    sgn = list()
    for a in act:
        sgn.append(a != 0)
    return sgn
    
def get_blobs_sgn(net):
    act = get_blobs_act(net)
    return get_blobs_sgn_from_act(act)

def get_trans(sgn0, sgn1):
    return np.sum(np.logical_xor(sgn0,sgn1))
    
def get_trans_pos(sgn0, sgn1):
    return np.sum(np.logical_and(np.negative(sgn0), sgn1))
        
def get_trans_neg(sgn0, sgn1):
    return np.sum(np.logical_and(sgn0, np.negative(sgn1)))

def test2(deploy, model=None, net=None,
          start_iter=0, stop_iter=None,
          lrs = [0.05, 0.005, 0.0005],
          maxiters = [32000, 48000, 64000],
          mom=0.9, decay=0.0001, dist_type='L2',
          display=100, test_interval=None, test_deploy=None):
    data_blob = 'data'
    label_blob = 'label'
    latte.set_device(1)
    if net == None:
        net = latte.MyNet(deploy, model, pretrained=(model!=None))
    elif model != None:
        net.copy_from(model)
    dataset = latte.CifarDataLoader(osp.join(CAFFE_ROOT, 'examples', 'cifar10/cifar10_train_lmdb'),
                                    net, phase=caffe.TRAIN,
                                    data_blob=data_blob)
    net.set_dataloader(dataset, data_blob, label_blob)
    if test_interval != None:
        test_net = latte.MyNet(test_deploy, phase=caffe.TEST, pretrained=False)
        testset = latte.CifarDataLoader(osp.join(CAFFE_ROOT, 'examples', 'cifar10/cifar10_test_lmdb'),
                                        test_net, phase=caffe.TEST,
                                        data_blob=data_blob)
        test_net.set_dataloader(testset, data_blob, label_blob)
        test_loss = []
        test_top1 = []
    else:
        test_net = None
        testset = None
    solver = latte.SGDSolver(net, test_net)
    #
    loss = []
    top1 = []
    trans = []
    i = start_iter
    for stage, maxiter in enumerate(maxiters):
        lr = lrs[stage]
        if stop_iter != None and maxiter > stop_iter:
            maxiter = stop_iter
        start_time = time.time()
        while i < maxiter:
            i += 1
            ofw, _, _ = solver.step(lr=lr, mom=mom, decay=decay, dist_type=dist_type)
            loss.append(np.copy(ofw['loss']))
            top1.append(np.copy(ofw['accuracy_top1']))
#            sgn0 = get_blobs_sgn(net)
#            net.forward()
#            sgn1 = get_blobs_sgn(net)
#            trans.append(np.array([get_trans(s0,s1) for s0,s1 in zip(sgn0, sgn1)]))
            if i % display == 0:
                end_time = time.time()
                print "%05d(%.3f) |" % (i, end_time - start_time),
                print "loss: %.6f |" % loss[-1],
                print "top1: %.4f |" % top1[-1],
                print "trans: %d" % 0#np.sum(trans[-1])
                start_time = end_time
            if test_interval != None and i % test_interval == 0:
                test_net.share_with(net)
                test1, test2 = solver.test()
                test_loss.append(test1)
                test_top1.append(test2)
                end_time = time.time()
                print "Test (%.3f) |" % (end_time - start_time),
                print "loss: %.6f |" % test1,
                print "top1: %.4f |" % test2
                start_time = end_time
            
    ret = {'net': net,
           'loss': loss,
           'top1': top1,
           'trans': trans}
    if test_interval != None:
        ret['test_loss'] = test_loss
        ret['test_top1'] = test_top1
#    np.savez("trans_over_time.npz", trans=np.array(trans))
    return ret

def save_test2_ret(ret, name='ret.npz', net_name='ret_net.caffemodel'):
    ret['net'].save(net_name)
    ret['net'] = net_name
    eval("np.savez('%s', %s)" % (name, ','.join(["%s=ret['%s']" % (k,k) for k in ret.keys()])))

if __name__ == "__main__":
    ret = test2(osp.join(CAFFE_ROOT, 'examples', 'cifar10/resnet20_cifar10_1st_deploy.prototxt'),
          lrs = [0.1, 0.01, 0.001],
          maxiters = [64000, 96000, 128000],
          dist_type='L2',
          display=100, test_interval=800,
          test_deploy=osp.join(CAFFE_ROOT, 'examples', 'cifar10/resnet20_cifar10_1st_test_deploy.prototxt'))
    save_test2_ret(ret)