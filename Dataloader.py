from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import os

default_transform = transforms.Compose([
    transforms.ToTensor(),  # convert to [0,1]
])

class PreprocessedDataset(Dataset):
    def __init__(self, root_dir, transform=default_transform):
        exts = ("*.jpg", "*.jpeg", "*.png")
        self.images = []
        for ext in exts:
            self.images.extend(glob.glob(os.path.join(root_dir, ext)))
        self.images = sorted(self.images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img



class KodakDataset(Dataset):
    def __init__(self, root_dir, transform=default_transform):
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