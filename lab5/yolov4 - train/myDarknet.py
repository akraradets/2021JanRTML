import darknet
import torch
import torch.nn as nn
from util import *

class MyDarknet(nn.Module):
    def __init__(self, cfgfile):
        super(MyDarknet, self).__init__()
        # load the config file and create our model
        self.blocks = darknet.parse_cfg(cfgfile)
        self.net_info, self.module_list = darknet.create_modules(self.blocks)
        
    def forward(self, x, CUDA:bool):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        
        write = 0
        # run forward propagation. Follow the instruction from dictionary modules
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            # print(f"{module}:{x.shape}")
            if module_type in ["convolutional","upsample",'maxpool']:
                # do convolutional network
                x = self.module_list[i](x)
    
            elif module_type == "route":
                # concat layers
                # print(module)
                # {'type': 'route', 'layers': ['-1', '-7']}
                layers = module["layers"]
                # layers = [int(a) for a in layers]
                for index, number in enumerate(layers):
                    layers[index] = int(number)
                    if(layers[index] > 0):
                        layers[index] = layers[index] - i
                # print(layers)
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                elif len(layers) == 2:
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                elif len(layers) == 4:
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    map3 = outputs[i + layers[2]]
                    map4 = outputs[i + layers[3]]
                    x = torch.cat((map1, map2, map3, map4), 1)
                else:
                    raise ValueError(f"len of {layers} not defined")
                    
    
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                # residual network
                x = outputs[i-1] + outputs[i+from_]
    
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                #Transform 
                x = x.data
                # predict_transform is in util.py
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)
            else:
                print(f"{module} {len(module_type.split('::'))}")
                raise ValueError(f"Opertation:{module_type} is not defined")
            outputs[i] = x
        
        return detections

    def load_weights(self, weightfile, backbone=False):
        '''
        Load pretrained weight
        '''
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        ''' 
            0: looking for start point
            1: Loading weight
            2: End point
        '''
        stage = 1
    
        if(backbone):
            stage = 0

        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            if(backbone):
                # print(stage, self.blocks[i + 1])
                if(stage == 2): break

                if("backbone" in self.blocks[i + 1] and int(self.blocks[i + 1]["backbone"]) == 0):
                    stage = 1
                elif("backbone" in self.blocks[i + 1] and int(self.blocks[i + 1]["backbone"]) == 1):
                    stage = 2

                if(stage == 0): continue
            
            print(self.blocks[i + 1])
            # Load weight
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
            