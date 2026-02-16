import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class APTOSDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform = None):
        self.df = dataframe.reset_index(drop = True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['id_code']
        label = self.df.iloc[idx]['diagnosis']

        img_path = os.path.join(self.img_dir, img_id + '.png')
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        return image, label