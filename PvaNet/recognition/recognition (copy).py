#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""
# add the directory - "tools"
import sys
#sys.path.append('/home/cvrsg/pva-faster-rcnn/tools/')

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

from caffe.proto import caffe_pb2
from google.protobuf import text_format

# CLASSES = ('__background__',
#            'Croissant', 'Pineapple', 'Coconut', 'Cheese',
#            'Baguette', 'Walnut', 'PorkFloss')
#
# NETS = {'vgg16': ('VGG16',
#                   'VGG16_faster_rcnn_final.caffemodel'),
#         'zf': ('ZF',
#                   'ZF_faster_rcnn_final.caffemodel')}

def demo(net, im, _t, CLASSES):
    """Detect object classes in an image using pre-computed object proposals."""
    #im_file = os.path.join('/home/cvrsg/pva-faster-rcnn/data', 'demo', image_name)
    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im = cv2.imread(im_file)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, _t)
    timer.toc()
    #print ('Detection took {:.3f}s for '
    #       '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    #CONF_THRESH = 0.8
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    detectrions_result = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        detections = get_detections(cls, dets, thresh=CONF_THRESH)
        #detections = get_detections(cls_ind, dets, thresh=CONF_THRESH)
        if len(detections) == 0:
            continue
        else:
            detectrions_result.extend(detections)
    return detectrions_result

def get_detections(class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return []
    detections = []
    for i in inds:
        bbox1 = dets[i, :4]
        bbox = bbox1.tolist()
        score1 = dets[i, -1]
        score = score1.tolist()
        detection = [class_name, bbox[0], bbox[1], bbox[2], bbox[3], score]
        detections.append(detection)
        
    return detections
def set_mode(mode = 'gpu', gpu_id = 0):
    if mode == 'gpu':
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    elif mode == 'cpu':
        caffe.set_mode_cpu()
    else:
        print "Please set mode = 'cpu' or 'gpu'"

def load_net(cfg_file, net_pt, net_weight):
    cfg_from_file(cfg_file)
    net = caffe.Net(net_pt, net_weight, caffe.TEST)
    return net


def load_net_v2():
    cfg_file = '/home/cvrsg/pva-faster-rcnn/models/pvanet/cfgs/submit_160715.yml'
    net_pt = '/home/cvrsg/pva-faster-rcnn/models/pvanet/full/test.pt'
    net_weight = '/home/cvrsg/pva-faster-rcnn/models/pvanet/full/test.model'

    cfg_from_file(cfg_file)
    cfg.GPU_ID = 0
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

    net = caffe.Net(net_pt, net_weight, caffe.TEST)
    return net

def detect(net, img, CLASSES):
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    detections = demo(net, img, _t, CLASSES)
    return detections

def take_picture(camera_id,width,height):
    capture = cv2.VideoCapture(camera_id)
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,width)  
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,height)  
    success, frame = capture.read()
    while success == False:
        success, frame = capture.read()
    return frame

# change num_classes in test_prototxt by len of CLASSES (=> cls_len)
def change_test_prototxt(test_prototxt_file, cls_len):
    net = caffe_pb2.NetParameter()
    f = open(test_prototxt_file, 'r')
    text_format.Merge(f.read(), net)
    # judge whether num_classes is equal to len(CLASSES)
    num_str_left = ''
    for layer in net.layer:
        if layer.name == 'cls_score':
            num_cls = layer.inner_product_param.num_output
            if num_cls == cls_len:
                f.close()
                return True
            else:
                break
    # if not equal
    for layer in net.layer:
        if layer.name == 'cls_score':
            layer.inner_product_param.num_output = cls_len
        if layer.name == 'bbox_pred':
            layer.inner_product_param.num_output = cls_len * 4

    f = open(test_prototxt_file, 'w')
    f.write(text_format.MessageToString(net))
    f.close()
    return False
