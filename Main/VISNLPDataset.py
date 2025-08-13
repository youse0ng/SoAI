from torch.utils.data import Dataset,DataLoader
import joblib
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os
from torch import tensor
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from warnings import filterwarnings
filterwarnings('ignore')

class VisualNLPCustomDataset(Dataset):
    def __init__(self,dataframe,image_path):
        self.data = joblib.load(dataframe)
        self.image_path = image_path
        self.mapping = {'선천성유문협착증':'Pyloric Stenosis','공기액체음영':'air-fluid level',
                        '기복증':'Abdominal distension','변비':'Constipation','정상':'Normal'}
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = os.path.join(self.image_path, row['ImagePath'], row['Filename'])
        image = Image.open(image_path).convert('RGB')
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        meta = self.mapping[row['ImagePath'].split('.')[-1]]
        return image_tensor, meta, row['Caption']