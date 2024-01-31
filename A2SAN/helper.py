import matplotlib.pyplot as plt
import torch
import numpy as np
#Define Dataset
class MyTrainData(torch.utils.data.Dataset):
  def __init__(self, img, gt, transform=None):
    self.img = img.float()
    self.gt = gt.float()
    self.transform=transform

  def __getitem__(self, idx):
    return self.img,self.gt

  def __len__(self):
    return 1

def reconstruction_SADloss(output, target):
    
    _,band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss
  

