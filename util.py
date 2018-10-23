from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.variable
import numpy as np
import cv2

def predict_transform(prediction,inp_dim,anchors,num_classes,cuda = True):
    batch_size =prediction.size (0)
    stride = inp_dim // prediction.size(2)
    grid_size = prediction.size(2)
    #grid_size = inp_dim // stride          #
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size,bbox_attrs*num_anchors,grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    ##调用view需要tensor是连续的，但是使用view，transpose后得到的tensor是不连续的，
    # 若要再次使用view，则要使用contiguous将tensor重新变为内存连续状态
    prediction = prediction.view(batch_size,grid_size*grid_size*num_anchors,bbox_attrs)
    anchors = [(a[0]/stride,a[1]/stride) for a in anchors]

    #sigmoid the center_x,center_y and confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    prediction = prediction.cuda()
    #add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid)
    x_offsets = torch.FloatTensor(a).view(-1,1)
    y_offsets = torch.FloatTensor(a).view(-1,1)

    if cuda:
        x_offsets = x_offsets.cuda()
        y_offsets = y_offsets.cuda()

    x_y_offset = torch.cat((x_offsets,y_offsets),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)


    prediction[:,:,:2] += x_y_offset

    anchors = torch.FloatTensor(anchors)
    if cuda:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size ,1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    prediction[:,:,5:5 + num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))
    prediction[:,:,:4] *= stride

    return prediction





























