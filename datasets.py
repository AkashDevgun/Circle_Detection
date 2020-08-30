import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


# Define Class for Circle Image Dataset to load circle while training
class CircleImages(Dataset):
    def __init__(self, file_path, transforms):
        self.train_data = pd.read_csv(file_path)
        self.transforms = transforms

    # Fetch item while training and applies transformation
    def __getitem__(self, item):
        img = np.load(self.train_data.iloc[item][0])
        target = [self.train_data.iloc[item][i] for i in range(1, 4)]

        if self.transforms:
            img = np.expand_dims(np.asarray(img), axis=0)
            img = torch.from_numpy(np.array(img, dtype=np.float32))
            target = torch.from_numpy(np.array(target, dtype=np.float32))
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.train_data)