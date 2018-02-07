# -*- coding:utf-8 -*-
import sys
import _init_paths

import os
import os.path as osp
import cv2
import numpy as np
import time
import copy
import caffe

net_file = '/home/kyxu/kyxu/Projects/AnSheng/ClassifyModel/cnn-models-master/VGG19_cvgj/deploy.prototxt'
caffe_model = '/home/kyxu/kyxu/Projects/AnSheng/ClassifyModel/cnn-models-master/VGG19_cvgj/models/models_iter_5000.caffemodel'

label_file = '/home/kyxu/kyxu/Projects/AnSheng/ClassifyModel/cnn-models-master/VGG19_cvgj/classname.txt'
labels = []
file_label = open(label_file, 'r')
while 1:
    line = file_label.readline()
    line = line.strip('\n')
    if not line:
        break
    labels.append(line)
print labels

test_file = '/home/kyxu/kyxu/Projects/AnSheng/ClassifyModel/Data/train-data/test.txt'
file_in = open(test_file, 'r')
imgs_infos = []
while 1:
    line = file_in.readline()
    line = line.strip('\n')
    if not line:
        break
    index = line.find(' ')
    img_path = line[0:index]
    label = line[index+1:]
    img_info = []
    img_info.append(img_path)
    img_info.append(label)
    imgs_infos.append(img_info)

caffe.set_mode_gpu()
# classifier = caffe.Classifier(net_file, caffe_model, '224, 224')

# print imgs_infos
# '''
net = caffe.Net(net_file, caffe_model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
# transformer.set_raw_scale('data', 255)
# transformer.set_channel_swap('data', (2,1,0))

right_num = 0
wrong_num = 0
num = 0
for img_info in imgs_infos:
    print img_info[0]
    image = cv2.imread(img_info[0])
    net.blobs['data'].reshape(1,3,224,224)
    net.blobs['data'].data[...] = [transformer.preprocess('data',image)]
    start = time.time()
    out = net.forward()
    print("Done in %.2f s." % (time.time() - start))
    result = net.blobs['prob'].data[0].flatten()
    # result_label = result.argsort()
    # label_index = result_label[-1]
    label_index = result.argsort()[-1]

    print result, label_index, img_info[1]
    # print result_label, labels[result_label], img_info[1]
    if label_index == int(img_info[1]):
        right_num += 1
    else:
        wrong_num += 1
accuracy_rate = float(right_num) / float(right_num + wrong_num)
print right_num, wrong_num, accuracy_rate
# '''
