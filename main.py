import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class VEDAI_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, "train", fname) for fname in os.listdir(os.path.join(root_dir, "train")) if fname.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name)
        label = self.get_label(img_name)  # Implement label fetching logic
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def get_label(self, img_name):
        # Implement logic to fetch labels based on the image name
        return label
