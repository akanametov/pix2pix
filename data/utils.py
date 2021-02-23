##############################################
##########   Import Libraries   ##############
##############################################

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

##############################################
##########   Facades Dataset   ###############
##############################################

class FacadesDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.array(Image.open(self.files[idx]))
        label = ToTensor()(data[:, :256, :])
        image = ToTensor()(data[:, 256:, :])
        return image, label