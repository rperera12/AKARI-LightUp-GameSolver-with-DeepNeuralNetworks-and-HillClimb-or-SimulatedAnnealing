from __future__ import print_function, division

import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
from tempfile import TemporaryFile
import time
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

from torch_dataload import Boardloader
from models import ConvNet1, ConvNet2, ConvNet3, ConvNet4

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('images/akari_experiment_1')


trans=transforms.Compose([transforms.ToTensor()])
data = Boardloader('Train_Set_Annealing/Initial_Sates_','Label_Set_Annealing/Optimized_Data_',trans)

test_sample = data[0]
x = (test_sample['initial']).numpy()
y = (test_sample['labels']).numpy()

print(x.shape)
print(x)
print(y.shape)
print(y)


train_ds, valid_ds = torch.utils.data.random_split(data, (9900, 21))
train_dl = DataLoader(train_ds, batch_size=10,
                            shuffle=True, num_workers=0)
valid_dl = DataLoader(valid_ds, batch_size=7, shuffle=True)

sample_b = next(iter(train_dl))
xb = sample_b['initial']
yb = sample_b['labels']
print(xb.shape)
print(yb.shape)       


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = ConvNet4()
model = model.to(device)

# check keras-like model summary using torchsummary

summary(model, input_size=(1, 7, 7))


def train(model, train_dl, valid_dl, loss_fn, optimizer, epochs=1):
    start = time.time()

    train_loss, valid_loss = [], []

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0

            step = 0

            # iterate over data
            for data in dataloader:
                x = data['initial']
                y = data['labels']

                # flatten pts
                y = y.view(y.size(0), -1)

                y = y.type(torch.cuda.FloatTensor)
                x = x.type(torch.cuda.FloatTensor)
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no 
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y)

                running_loss += loss*dataloader.batch_size 

                if step % 10 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  AllocMem (Mb): {}'.format(step, loss, torch.cuda.memory_allocated()/1024/1024))
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)

            print('{} Loss: {:.4f} '.format(phase, epoch_loss))

            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    torch.save(model.state_dict(),'./logs/Pretrained_Models/Annealing_Conv4_Akari.pt')    
    
    return train_loss, valid_loss 


learning_rate = 0.001
loss_fn = nn.SmoothL1Loss() # Huber Loss Funtion
opt = torch.optim.Adam(model.parameters(), lr=learning_rate) 

train_loss, valid_loss = train(model, train_dl, valid_dl, loss_fn, opt, epochs=50)


plt.figure(figsize=(10,8))
plt.plot(train_loss, label='Train loss')
plt.plot(valid_loss, label='Valid loss')
plt.legend()
plt.show()



model.load_state_dict(torch.load('./logs/Pretrained_Models/Annealing_Conv4_Akari.pt'))
dataloader = valid_dl
results_sample = next(iter(dataloader))
inn = results_sample["initial"].type(torch.cuda.FloatTensor)
outt = results_sample["labels"].cpu().numpy()

with torch.no_grad():
    pred = model(inn)

pred = pred[0,:]
pred = pred.reshape((7,7))
pred = pred.cpu().numpy() #.transpose((1,0))
inn = inn.cpu().numpy()


pieces_dir = get_img_path()


in123 = np.copy(inn[0,0,:,:])
testing123 = np.copy(outt[0,:,:]) #.transpose((1,0))

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


