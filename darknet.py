from __future__ import division
import torch
import torchvision
import torch.nn as nn
from util import *
import torch.nn.functional as F
from torch.autograd import variable
import numpy
import pprint
from tensorboardX import SummaryWriter

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors


class DarkNet(nn.Module):

    def __init__(self,cfgfile):
        super(DarkNet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info,self.module_list = create_modules(self.blocks)

    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}
        write = 0

        for i , module in enumerate(modules):
            module_type = (module['type'])

            if module_type == 'convolutional' or module_type == 'upsample':

                x = self.module_list[i](x)

            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]

                if (layers[0]>0):
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]

                else:
                    if (layers[1]>0):
                        layers[1] -= i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1,map2),1)


            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors

                #get the input dimensions
                inp_dim = int(self.net_info['height'])

                #get the numble of classes
                num_classes = int (module['classes'])

                x = x.data
                cuda = torch.cuda.is_available()
                x = predict_transform(x,inp_dim,anchors,num_classes,cuda)
                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections,x),1)

            outputs[i] = x

        return detections

    def load_weights(self,weightfile):

        fp = open(weightfile,'rb')
        header = np.fromfile(fp,dtype=np.int32,count = 5)
        self.header = torch.Tensor(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp,dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']

            #if type is convlutional,then load the weights
            #otherwise ignore
            if module_type =='convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.Tensor(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.Tensor(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.Tensor(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.Tensor(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    #cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weights.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_bn_biases = conv.bias.numel()
                    conv_biases = torch.Tensor(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    conv_biases = conv_biases.view_as(conv.bias.dta)
                    conv.bias.data.copy(conv_biases)

                num_weights = conv.weights.numel()
                conv_weights = torch.Tensor(weights[ptr:ptr+num_weights])
                prt = ptr+num_weights

                conv_weights = conv_weights.view_as(conv.weights.data)
                conv.weights.data.copy_(conv_weights)










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
    # pprint.pprint (blocks)
    return  blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    pre_filters = 3
    output_filters = []
    for index,x in enumerate(blocks[1:]):

        module = nn.Sequential()

        #check the type of block
        #create a new module for the block
        #append to module_list
        if (x['type']=='convolutional'):
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize=0
                bias = True
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int (x['stride'])
            if padding:
                pad = (kernel_size-1)//2
            else:
                pad = 0

            #add the convlutional layer
            conv = nn.Conv2d(pre_filters,filters,kernel_size,stride,pad,bias= bias)
            module.add_module("conv_{0}".format(index), conv)

            #add the batch_normalize layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            #add the activation
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1,inplace= True)
                module.add_module('leaky_{0}'.format(index),activn)

        #if its a upsample layer
        elif (x['type']=='upsample'):
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor= 2,mode='bilinear')
            #upsample = nn.functional.interpolate(input,scale_factor=2, mode='bilinear')
            module.add_module('upsamle_{0}'.format(index),upsample)

        #if its a route layer
        elif(x['type']=='route'):
            x['layers'] = x['layers'].split(',')
            #start of a route
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            #positive anotation
            if start>0:
                start = start - index
            if end>0:
                end = end - index
            route = EmptyLayer()
            module.add_module('route_{0}'.format(index),route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{0}'.format(index),shortcut)

        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x ['anchors'].split(',')
            anchors = [int (a) for a in anchors]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{0}'.format(index),detection)


        module_list.append(module)
        pre_filters = filters
        output_filters.append(filters)

    return (net_info,module_list)

def get_test_input(im):
    img = cv2.imread(im)

    img = cv2.resize(img,(416,416))
    img = img[:,:,::-1].transpose((2,0,1))
    img = img[np.newaxis,:,:,:]/255.0
    img = torch.Tensor(img)
    return img




if __name__ == '__main__':

    cfg = './cfg/yolov3.cfg'
    model = DarkNet(cfg)
    imgfile = 'dog-cycle-car.png'
    img = get_test_input(imgfile)
    cuda = torch.cuda.is_available()
    pred = model(img)
    print(pred)
    print('#'*20)






