from stat import ST_ATIME
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import scipy.io as sio
import numpy as np
import os
import math
import time
import random
from helper import *
import torchsummary as summary
from network import A2SAN
import logging
import scipy.io
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = 'Samson'
if dataset == 'Samson':
    image_file = r'Samson_dataset.mat'
    P, L, col = 3, 156, 95
    pixel = col**2
    LR, EPOCH, batch_size = 1e-3, 1000, 1
    step_size, gamma = 45, 0.9
    a, b= 1, 5e-1
    weight_decay_param = 1e-4
    drop = 0.08
data = sio.loadmat(image_file)


HSI = torch.from_numpy(data["Y"])  # mixed abundance
GT = torch.from_numpy(data["A"])  # true abundance
M_true = data['M']

band_Number = HSI.shape[0]
endmember_number, pixel_number = GT.shape

HSI = torch.reshape(HSI, (L, col, col))
GT = torch.reshape(GT, (P, col, col))

model = 'A2SAN'
if model == 'A2SAN':
    net = A2SAN(L,P)
else:
    logging.error("So such model in our zoo!")

MSE = torch.nn.MSELoss(size_average=True)
    
# load data
train_dataset = MyTrainData(img=HSI, gt=GT, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False) 

optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_param)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

best_loss =1
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):
        
        scheduler.step()
        x = x.cuda()
        net.train().cuda()
        abu, end, re, xo = net(x)

        re = torch.reshape(re,(L,col,col))
        abu = torch.reshape(abu,(P,col,col))
        reloss = reconstruction_SADloss(x, re)
        
        abu_neg_error = torch.mean(torch.relu(-abu))
        abu_sum_error = torch.mean((torch.sum(abu, dim=0) - 1) ** 2)
        abu_error = abu_neg_error + abu_sum_error 
  
        total_loss = a*reloss + b*abu_error

        optimizer.zero_grad()

        total_loss.backward()
        optimizer.step()
        
        loss = total_loss.cpu().data.numpy()

        if loss < best_loss:
            state = {
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'loss': loss,
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, "./Samsonresult/A2SAN/SamsonA2SANbest_model.pth.tar")
            best_loss = loss
        
        if epoch % 100 == 0:
            print(
                "Epoch:",
                epoch,
                "| loss: %.4f" % total_loss.cpu().data.numpy(),
                "| abu_error: %.4f" % abu_error.cpu().data.numpy(),
            )

  
checkpoint = torch.load("./Samsonresult/A2SAN/SamsonA2SANbest_model.pth.tar")
best_loss = checkpoint['best_loss']
loss = checkpoint['loss']
epoch = checkpoint['epoch']

net.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

net.eval()
abu, end, re, xo = net(x)

xo = xo.detach().cpu().numpy()
end = end.squeeze(0)
end = end.detach().cpu().numpy()
abu = torch.reshape(abu,(P,col,col))
re = torch.reshape(re, (L,col*col))
re = re.detach().cpu().numpy()

abundance_GT = GT
GT_endmember = M_true

abu, abundance_GT = norm_abundance_GT(abu, abundance_GT, endmember_number, col)
end, GT_endmember = norm_endmember(end, M_true, endmember_number)

abundance_input, endmember_input, RMSE_abundance, SAD_endmember = arange_A_E(abu, abundance_GT, end, GT_endmember, endmember_number,)

print("RMSE", RMSE_abundance)
print("mean_RMSE", RMSE_abundance.mean())
print("endmember_SAD", SAD_endmember)
print("mean_SAD", SAD_endmember.mean())

end = endmember_input
abu = abundance_input

plot_endmember(GT_endmember, end, endmember_number)
plot_abundance(abundance_GT, abu, endmember_number)
abu = np.reshape(abu,(P,col*col))
abundance_GT = np.reshape(abundance_GT,(P,col*col))
