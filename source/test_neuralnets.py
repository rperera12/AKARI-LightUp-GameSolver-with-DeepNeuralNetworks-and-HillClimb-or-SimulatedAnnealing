from __future__ import print_function, division

import torch
import torch.nn as nn
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
from torchsummary import summary
import cma
import cv2

# import from project functions
from copy import deepcopy


from board import GameBoard
from utils import Generate_Board, get_img_path, get_play_path, generate_dataset

# --------imports from system------
import os
import sys
from operator import itemgetter
import numpy as np
from playsound import playsound

from dataload import Boardloader
from models import ConvNet1, ConvNet2, ConvNet3, ConvNet4


trans=transforms.Compose([transforms.ToTensor()])
data = Boardloader('Train_Set/Train_Set/Initial_Sates_','Label_Set/Label_Set/Optimized_Data_',trans)



dataloader = DataLoader(data, batch_size=10,
                            shuffle=True, num_workers=0)
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = ConvNet4()
model = model.to(device)

# check keras-like model summary using torchsummary
summary(model, input_size=(1, 7, 7))



model.load_state_dict(torch.load('./logs/Pretrained_Models/Annealing_Conv4_Akari.pt'))

results_sample = next(iter(dataloader))
inn = results_sample["initial"].type(torch.cuda.FloatTensor)
outt = results_sample["labels"].cpu().numpy()

with torch.no_grad():
    pred = model(inn)
    print(pred.shape)

pred = pred[5,:]
pred = pred.reshape((7,7))
pred = pred.cpu().numpy() #.transpose((1,0))
inn = inn.cpu().numpy()


pieces_dir = get_img_path()
in123 = np.copy(inn[5,0,:,:])
testing123 = np.copy(outt[5,:,:]) #.transpose((1,0))

pred123 = np.zeros([7,7])
testing4 = np.zeros([7,7])
for kk in range(7):
    for jj in range(7):
        pred123[6-kk,jj] = pred[kk,jj]
        testing4[6-kk,jj] = testing123[kk,jj]
pred123 = np.rint(pred123)

print(in123)
print(testing123)
print(pred123)


pieces_dir = get_img_path()
Generate_Board(pieces_dir, in123)
Generate_Board(pieces_dir, testing4)
Generate_Board(pieces_dir, pred123)


