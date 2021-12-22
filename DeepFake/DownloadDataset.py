#Python imports
import torch
import torch.nn as tnn
from torch.autograd import Variable
import sys
import itertools
from torch.utils.data import DataLoader
from Dataset import ImageSet
import datetime
import torchvision.utils as tv
import os.path
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

#Import network elements
import network_elements as net

#stdout code blocks reload's print statement
#prevstd = sys.stdout
#sys.stdout = None
reload(net)
print("Reloading net")
#sys.stdout = prevstd


# <h3>Set parameters</h3>

# In[ ]:


#The number of channels in the input image
im_in_channels = 3

#The number of channels in the output image
im_out_channels = 3

#The size of the largest size of the input image
im_size = 128

LEARNING_RATE = 0.0002

LR_DECAY_START = 100
LR_DECAY_END = 200
NUM_EPOCHS = 200

load_partial_net = False
CURR_EPOCH = 0

# <h3>Import dataset</h3>

# In[ ]:


#You can choose the following data sets: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
imageSet = ImageSet();
dataset = "vangogh2photo"
imageSet.downloadData(dataset)
training_transforms = [transforms.Resize(int(im_size*1.12), Image.BICUBIC),
                  transforms.RandomCrop(im_size),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))];
imageSet.loadData(dataset, 'train', im_size, training_transforms)
imgLoader = DataLoader(imageSet,shuffle=True)
