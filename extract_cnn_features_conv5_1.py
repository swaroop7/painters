#!/usr/bin/env python
#coding:utf-8
 
import caffe
import cv2 
import numpy as np
import os
 
EPSILON = 1.0e-8
 
TRAIN_FILE_PATH      = "/home/ubuntu/neural-style/train_conv5_1.txt"
TEST_FILE_PATH       = "/home/ubuntu/neural-style/test_conv5_1.txt"
TOTAL_LIST_PATH      = "/home/ubuntu/neural-style/total_list_15_in_local_machine.txt"
DEPLOY_PROTOTXT_PATH = '/home/ubuntu/neural-style/models/VGG_ILSVRC_19_layers_deploy.prototxt'
CAFFEMODEL_PATH      = '/home/ubuntu/neural-style/models/VGG_ILSVRC_19_layers.caffemodel'
 
#LABEL_A = "livingroom"
#LABEL_B = "industrial"
 
#
def extract_feature(net, image_path):
    if ".jpg" not in image_path:
        raise Exception("invalid path")
 
    image = caffe.io.load_image(
        image_path, 
        color = True, 
    )   
 
    scores = net.predict([image], oversample = False)
    #return net.blobs['pool1'].data[0].tolist()
    #print net.blobs['conv1_1'].data[0].shape
    coslist = np.ndarray.flatten(net.blobs['conv5_1'].data[0])
    #print net.blobs['fc7'].data[0].shape
    print len(coslist)
    #print np.ndarray(net.params['conv1_1'][0]).shape
    return coslist.tolist()

 
#
def make_format(label, feature):
    line = label + " "
    for (i, f) in enumerate(feature):
        if abs(f) > EPSILON:
            line += "{0}:{1} ".format(i+1, f)
    return line + "\n"
 

if __name__ == "__main__":
        
    train_file = open(TRAIN_FILE_PATH, "w")
    test_file  = open(TEST_FILE_PATH, "w")
    total_list = TOTAL_LIST_PATH
    try:
        with open(total_list) as infile:
            net = caffe.Classifier(DEPLOY_PROTOTXT_PATH, CAFFEMODEL_PATH)
            for (i, line) in enumerate(infile):
                print(i)
                tokens = line.split()
                if len(tokens) != 3:
                    raise Exception("invalid line")
                
                name = tokens[0]
                #if (not LABEL_A in name) and (not LABEL_B in name):
                #    continue
 
                feature = extract_feature(net, name)
                contents = make_format(tokens[1], feature)
                if tokens[2] == "test":
                    test_file.write(contents) 
                else:
                    train_file.write(contents)
                    
    except Exception as inst:
        print(inst.args)
