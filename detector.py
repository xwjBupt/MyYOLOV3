from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import DarkNet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    '''
    parse the arguement to the detect module
    :return:
    '''

    parser = argparse.ArgumentParser(description='YOLO V3 Detector module')

    parser.add_argument('--images',dest='images',help = 'images ditectory',default='./data',type = str)
    parser.add_argument('--det',dest='det',help='ditectory to store the detection result',default='./det',type = str)
    parser.add_argument('--bs',dest='bs',help='Batch size',default=1,type = int)
    parser.add_argument('--confidence',dest= 'confidence',help= 'Object Confidence to filter the prediction',default=0.5,type=float )
    parser.add_argument('--nms_thresh',dest='nms_thresh',help='nms_threshhold',default= 0.6,type=float)
    parser.add_argument('--cfg_file',dest='cfg_file',help='the directory to cfg file',default='./cfg/cfgfile',type= str)
    parser.add_argument('--weight',dest='weight',help='the directory to weight binary file',default='./weights/yolov3.weights',type = str)
    parser.add_argument('--reso',dest='reso',help='input resolution to the network,increase to increase accuracy,decrease to increase speed',default=416,type=int)

    return parser.parse_args()

def load_classes(namesfile):
    fp = open(namesfile,'r')
    names = fp.read().split('\n')[:-1]
    return names

def letterbox_image(img,inp_dim):
    '''
    resize the input image with unchanged aspect ratio using padding
    :param img:
    :param inp_dim:
    :return: resized image
    '''

    img_w,img_h = img.shape[1],img.shape[0]
    w,h = inp_dim
    new_w = int(img_w * min(w/img_w,h/img_h))
    new_h = int(img_h * min(w/img_w,h/img_h))
    resized_img = cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1],inp_dim[0],3),128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_img
    return canvas

def prep_image(img,inp_dim):
    img = cv2.resize(img,(inp_dim,inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.Tensor(img).float().div(255.0).unsqueeze(0)
    return img





if __name__ == '__main__':

    args = arg_parse()
    images = args.images
    bs = args.bs
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    cuda = torch.cuda.is_available()
    classes = load_classes('./cfg/coco.names')
    num_classes = 80
    #set up the network
    print('loading the network')
    model = DarkNet(args.cfg_file)
    model.load_weights(args.weight)
    print('load the network successfully')

    model.net_info['height'] = args.reso
    inp_dim = int(model.net_info['height'])
    assert inp_dim % 32 == 0
    assert inp_dim >32

    if cuda:
        model.cuda()

    #set the model into eval mode
    model.eval()
    read_dir = time.time()

    try:
        imlist = [img for img in os.listdir(images)]

    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'),images))

    except FileNotFoundError:
        print('No such file or directory with the name {}'.format(images))
        exit()

    if not os.path.exists(args.det):
        os.mkdirs(args.det)


    loaded_ims = [cv2.imread(x) for x in imlist]
    load_batch = time.time()
    im_batches = list(map(prep_image, loaded_ims,[inp_dim for x in range(len(imlist))]))

    #list containing dimensions of original images
    im_dim_list = [(x.shape[1],x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    if cuda:
        im_dim_list = im_dim_list.cuda()


    write = 0
    start_det_loop = time.time()
    for i, batch in enumerate(im_batches):

        start = time.time()
        if cuda:
            batch = batch.cuda()


        prediction = model(batch)
        prediction = write_results(prediction,confidence,num_classes=80,nms_conf= nms_thresh)
        end = time.time()
        if type(prediction) == 0:
            for im_num,image in enumerate(imlist[i*bs:min((i+1)*bs,len(imlist))]):
                im_id = i * bs + im_num
                print ("{0:20s} predicted in {1:6.3f}".format(image.split('/')[-1],(end - start)/bs))
                print ('{0:20s} {1:s}'.format('objetc detected',' '))
                print('--'*20)

            continue

        prediction[:0] += i*bs

        if not write:
            output  = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))

        for im_num, image in enumerate(imlist[i * bs:min((i + 1) * bs, len(imlist))]):
            im_id = i *bs + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f}".format(image.split('/')[-1], (end - start) / bs))
            print('{0:20s} {1:s}'.format('objetc detected', ' '.join(objs)))
            print('--' * 20)

        if cuda:
            torch.cuda.synchronize()

        try:
            output
        except NameError:
            print('No detection were made')
            exit()

        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
        scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])


        output_recast = time.time()
        class_load = time.time()
        colors = pkl.load(open('pallete', 'rb'))

        draw = time.time()


        def write(x, results):

            c1 = tuple(x[1:3].int())
            c2 = tuple(x[3:5].int())
            img = results[int(x[0])]
            cls = int(x[-1])
            color = random.choice(colors)
            label = "{0}".format(classes[cls])
            cv2.rectangle(img, c1, c2, color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
            return img

        list(map(lambda x: write(x, loaded_ims), output))
        det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))
        list(map(cv2.imwrite, det_names, loaded_ims))
        end = time.time()

        print("SUMMARY")
        print("----------------------------------------------------------")
        print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
        print()
        print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
        print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
        print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
        print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
        print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
        print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(imlist)))
        print("----------------------------------------------------------")

    torch.cuda.empty_cache()



