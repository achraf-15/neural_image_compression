import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os

class PreprocessedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img



class KodakDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = sorted(glob.glob(os.path.join(root_dir, '*.png')))
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img