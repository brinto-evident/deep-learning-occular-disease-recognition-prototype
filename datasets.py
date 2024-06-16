import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os


class ODIR5K(Dataset):
    def __init__(self, root_dir, transform=None, crop=False):
        super(ODIR5K, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.data = self._load_data()
        self.crop = crop

    def get_labels(self):
        print(self.class_to_idx)
    
    def _load_data(self):
        data = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for filename in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, filename)
                data.append((img_path, self.class_to_idx[cls_name]))
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if label == 0:
            label = torch.FloatTensor([1,0,0,0])
        elif label == 1:
            label = torch.FloatTensor([0,1,0,0])
        elif label == 2:
            label = torch.FloatTensor([0,0,1,0])
        else:
            label = torch.FloatTensor([0,0,0,1])

        if self.crop: # crop black pixels
            image = np.array(image)

            # Mask of coloured pixels.
            mask = image > 0

            # Coordinates of coloured pixels.
            coordinates = np.argwhere(mask)

            # Binding box of non-black pixels.
            x0, y0, s0 = coordinates.min(axis=0)
            x1, y1, s1 = coordinates.max(axis=0) + 1 # slices are exclusive at the top.

            # Get the contents of the bounding box.
            image = image[x0:x1, y0:y1]
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])
    
    dataset = ODIR5K('dataset', transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    dataset.get_labels()
    

    for (image, label) in iter(dataloader):
        print(image[5].size())
        print(label[5])
        break
        # image = image[idx].view(3, 256, 256).permute(1, 2, 0)
        
        # plt.figure(figsize=(6, 6))
        # plt.imshow(image)
        # plt.tight_layout()
        # plt.savefig('figure/input.png')

        # break
