import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class PolypDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".tif"))

#         mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.tif"))
#         if not os.path.isfile(mask_path):
#             mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.jpg"))
#         if not os.path.isfile(mask_path):
#             mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".jpg"))
#         if not os.path.isfile(mask_path):
#             mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
#         if not os.path.isfile(mask_path):
#             mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_polyp.tif"))
            
        image = np.array(Image.open(img_path).convert("RGB"))
        image=(image/255.).astype(np.float32)
        mask = np.array(Image.open(mask_path).convert("L"))
        mask=(mask/255.).astype(np.float32)
        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0
    
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image,mask