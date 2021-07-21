from __future__ import print_function, division

import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torchvision import models
import torchvision
import os
import numpy as np
from tempfile import TemporaryFile
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from skimage import transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
#import vlc
from playsound import playsound
import cma
import cv2


##############################################################################
##############################################################################
##############################################################################

class ConvNet1(nn.Module):
        def __init__(self):
            super(ConvNet1, self).__init__()


            # First Layer: (Check for Identity-Layer, or Convolution Layer)
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 32, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU())

            # Second Layer: (Check for Identity-Layer, Transpose Convolution, or Convolution Layer)     
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU())
            
            # Third Layer: (Check for Identity-Layer, or Convolution Layer)     
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU())

            # Fourth Layer: (Check for Identity-Layer, Transpose Convolution, or Convolution Layer) 
            self.layer4 = nn.Sequential(
                nn.Conv2d(128, 256, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU())

            self.fc1 = nn.Linear(256*7*7, 512)
            self.fc2 = nn.Linear(512, 7 * 7) 


        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            return  out


##############################################################################
##############################################################################
##############################################################################


class ConvNet2(nn.Module):
        def __init__(self):
            super(ConvNet2, self).__init__()


            # First Layer: (Check for Identity-Layer, or Convolution Layer)
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 32, 
                        kernel_size=1, stride=1, padding=0),
                nn.ReLU())

            # Second Layer: (Check for Identity-Layer, Transpose Convolution, or Convolution Layer)     
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU())
            
            # Third Layer: (Check for Identity-Layer, or Convolution Layer)     
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU())

            # Fourth Layer: (Check for Identity-Layer, Transpose Convolution, or Convolution Layer) 
            self.layer4 = nn.Sequential(
                nn.Conv2d(128, 256, 
                        kernel_size=3, stride=1, padding=1),
                nn.ReLU())

            self.fc1 = nn.Linear(256*7*7, 512)
            self.fc2 = nn.Linear(512, 7 * 7) 


        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = self.fc2(out)
            return  out


##############################################################################
##############################################################################
##############################################################################


class ConvNet3(nn.Module):
        def __init__(self):
            super(ConvNet3, self).__init__()


            # First Layer: (Check for Identity-Layer, or Convolution Layer)
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 32, 
                        kernel_size=5, stride=1, padding=2),
                nn.ReLU())
            self.drop1 = nn.Dropout(0.1)    

            # Second Layer: (Check for Identity-Layer, Transpose Convolution, or Convolution Layer)     
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, 
                        kernel_size=5, stride=1, padding=2),
                nn.ReLU())
            self.drop2 = nn.Dropout(0.2)    

            
            # Third Layer: (Check for Identity-Layer, or Convolution Layer)     
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, 
                        kernel_size=5, stride=1, padding=2),
                nn.ReLU())
            self.drop3 = nn.Dropout(0.25)    

            # Third Layer: (Check for Identity-Layer, or Convolution Layer)     
            self.layer4 = nn.Sequential(
                nn.Conv2d(128, 256, 
                        kernel_size=5, stride=1, padding=2),
                nn.ReLU())    
            self.drop4 = nn.Dropout(0.25)    

            self.fc1 = nn.Linear(256*7*7, 2048)
            self.drop5 = nn.Dropout(0.3)            
            self.fc2 = nn.Linear(2048, 7*7) 


        def forward(self, x):
            out = self.drop1(self.layer1(x))
            out = self.drop2(self.layer2(out))
            out = self.drop3(self.layer3(out))
            out = self.drop4(self.layer4(out))
            out = out.view(out.size(0), -1)
            out = self.drop5(F.relu(self.fc1(out)))
            out = self.fc2(out)
            return  out



class ConvNet4(nn.Module):
        def __init__(self):
            super(ConvNet4, self).__init__()


            # First Layer: (Check for Identity-Layer, or Convolution Layer)
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, 
                        kernel_size=5, stride=1, padding=2),
                nn.ReLU())

            # Second Layer: (Check for Identity-Layer, Transpose Convolution, or Convolution Layer)     
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, 
                        kernel_size=5, stride=1, padding=2),
                nn.ReLU())
            
            # Third Layer: (Check for Identity-Layer, or Convolution Layer)     
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, 
                        kernel_size=5, stride=1, padding=2),
                nn.ReLU())
       
            self.fc1 = nn.Linear(64*7*7,256)
            self.fc2 = nn.Linear(256, 7*7) 


        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            return  out            