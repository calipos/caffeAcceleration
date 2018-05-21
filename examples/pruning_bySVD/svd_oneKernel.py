#coding=utf-8

import os.path as osp
import sys
import copy
import os
import numpy as np
import numpy.linalg as linalg

CAFFE_ROOT = '/home/lbl/lbl_trainData/git/caffe_pruning'
if osp.join(CAFFE_ROOT,'python') not in sys.path:
        sys.path.insert(0,osp.join(CAFFE_ROOT,'python'))

import caffe

caffe.set_mode_cpu()
original_net = caffe.Net('./bn.prototxt', './pelee_bn.caffemodel', caffe.TEST)
#original_net = caffe.Net('./models/result.prototxt', './models/result.caffemodel', caffe.TEST)

im = np.random.random((1,3,304,304))
data_original_net = original_net.blobs['data']
data_original_net.data[...] = im

theLayer = original_net._layer_by_name('stem1')
weights=theLayer.blobs[0].data
dim=weights.shape
weights=np.transpose(weights,[1,2,0,3])
weights=weights.reshape(dim[1]*dim[2],dim[0]*dim[3])

V,sig,H = linalg.svd(weights)
acc=[sig[0]]
for i in range(1,len(sig)):
	acc.append((sig[i]+acc[i-1]))
acc= acc/np.sum(sig)
print acc
for i in range(len(sig)):
	if acc[i]>0.95:
		rank=i+1
		break
print rank
#exit(0)
#rank=len(sig)-1
V=V[:,0:rank]
H=H[0:rank,:]
sig=np.diag(sig[0:rank])
H=sig.dot(H)
#print V.shape
#print sig
#print H.shape
H=H.reshape(rank,dim[0],dim[3],1)
V=V.reshape(dim[1],1,dim[2],rank)
H = np.transpose(H, [1, 0, 3, 2])
V = np.transpose(V, [3, 0, 2, 1])
#print weights.shape
#print V.shape
#print H.shape
split_net = caffe.Net('./split.prototxt', './pelee_bn.caffemodel', caffe.TEST)
theLayer = split_net._layer_by_name('stem11')
dim11=theLayer.blobs[0].data.shape
for n in range(dim11[0]):
	for c in range(dim11[1]):
		for h in range(dim11[2]):
			for w in range(dim11[3]):
				split_net._layer_by_name('stem11').blobs[0].data[n,c,h,w]=V[n,c,h,w]
theLayer = split_net._layer_by_name('stem12')
dim12=theLayer.blobs[0].data.shape
for n in range(dim12[0]):
	for c in range(dim12[1]):
		for h in range(dim12[2]):
			for w in range(dim12[3]):
				split_net._layer_by_name('stem12').blobs[0].data[n,c,h,w]=H[n,c,h,w]


data_split_net = split_net.blobs['data']
data_split_net.data[...] = im

original_net.forward()
print original_net.blobs['stem1'].data[0,0,0,0:20]
split_net.forward()
print split_net.blobs['stem12'].data[0,0,0,0:20]

original_net.save("bn.caffemodel")
split_net.save("split.caffemodel")
