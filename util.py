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
    #grid_size = inp_dim // stride       #~=46
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    #now the prediction = [1,255,13,13]
    prediction = prediction.view(batch_size,bbox_attrs*num_anchors,grid_size*grid_size)
    #now the prediction = [1,255,169]
    prediction = prediction.transpose(1,2).contiguous()
    #now the prediction = [1,169,255]
    prediction = prediction.view(batch_size,grid_size*grid_size*num_anchors,bbox_attrs)
    #now the prediction = [1,507,85] (3*(4+1+80))

    ##调用view需要tensor是连续的，但是使用view，transpose后得到的tensor是不连续的，
    # 若要再次使用view，则要使用contiguous将tensor重新变为内存连续状态

    anchors = [(a[0]/stride,a[1]/stride) for a in anchors]

    #sigmoid the center_x,center_y and confidence
    # 0:x, 1:y, 2:x_offsets, 3:y_offsets, 4:confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    prediction = prediction.cuda()
    #add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid)
    x_offsets = torch.FloatTensor(a).view(-1,1)
    y_offsets = torch.FloatTensor(b).view(-1,1)

    if cuda:
        x_offsets = x_offsets.cuda()
        y_offsets = y_offsets.cuda()

    x_y_offset = torch.cat((x_offsets,y_offsets),1)
    x_y_offset = x_y_offset.repeat(1,num_anchors)
    x_y_offset = x_y_offset.view(-1,2)
    x_y_offset = x_y_offset.unsqueeze(0)


    prediction[:,:,:2] += x_y_offset  #x_y_offset是每个grid的起始位置

    anchors = torch.FloatTensor(anchors)
    if cuda:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size ,1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    prediction[:,:,5:5 + num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))
    prediction[:,:,:4] *= stride

    return prediction


def write_results(prediction,confidence,num_classes,nms_conf = 0.6):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)
    write = 0


    for ind in range(batch_size):
        image_pre = prediction[ind]
        max_conf,max_conf_score = torch.max(image_pre[:,5:5+num_classes],1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pre[:,:5],max_conf,max_conf_score)
        image_pre = torch.cat(seq,1)
        non_zero_ind = (torch.nonzero(image_pre[:,4]))

        try:
            image_pred_ = image_pre[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        img_classes = unique(image_pred_[:,-1])
        for cls in img_classes:
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            conf_sort_index = torch.sort(image_pred_class[:,4],descending = True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            #now it is time for nms
            for i in range(idx):
                #get the ious of all boxes that come after the one we are looking at in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0),image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                #zero out all the detections that have IOU > threshhold
                iou_mask = (ious<nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                #remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0),1).fill_(ind)

            #repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind,image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = 1
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    try:
        return output
    except:
        return 0



def bbox_iou(box1,box2):
    '''

    :param box1:
    :param box2:
    :return: returns the ios of two bounding boxes
    '''

    b1_x1,b1_y1,b1_x2,b1_y2 = box1[:,0],box1[:,1],box1[:,2],box1[:,3]
    b2_x1,b2_y1,b2_x2,b2_y2 = box2[:,0],box2[:,1],box2[:,2],box2[:,3]

    #get the corrdinates of the intersection over unioun

    inter_x1 = torch.max(b1_x1,b2_x1)
    inter_y1 = torch.max(b1_y1,b2_y1)
    inter_x2 = torch.max(b1_x2,b2_x2)
    inter_y2 = torch.max(b1_y2,b2_y2)

    #intersection area
    inter_area = torch.clamp(inter_x2-inter_x1+1,min = 0)*torch.clamp(inter_y2-inter_y1+1,min = 0)

    #union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area + inter_area)

    return iou


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.Tensor(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res




























