from __future__ import division
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable
import numpy
import pprint

def parse_cfg(cfgfile):
    """
        Takes a configuration file

        Returns a list of blocks. Each blocks describes a block in the neural
        network to be built. Block is represented as a dictionary in the list

        """
    cfgfile = open('./cfg/yolov3.cfg','r')

    #lines = cfgfile.read().split('\n')
    lines = cfgfile.readlines()
    lines = [x for x in lines if len(x)>0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]
    lines = [x for x in lines if x !='']
    block = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            if len(block)!=0:
                blocks.append(block)
                block ={}
            block['type']=line[1:-1].rstrip()
        else:
            key,value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    pprint.pprint (blocks)
    return  blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    pre_filters = 3
    output_fileters = []


cfg = './cfg/yolov3.cfg'
parse_cfg(cfg)



