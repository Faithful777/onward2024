import os
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, image_paths, target_paths, path, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.path = path

    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(520, 520))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Transform to tensor
        image = TF.to_tensor(image)
        
        # Z-score normalization
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, mask

    def __getitem__(self, index):
        print(self.image_paths[index])
        #print(os.path.join(image_folder, os.path.splitext(self.image_paths[index])[0] + '.npy'))
        image_folder = os.path.join(self.path, 'image')
        label_folder = os.path.join(self.path, 'label')
        image = Image.open(os.path.join(image_folder, self.image_paths[index]))
        label = np.load(os.path.join(label_folder, os.path.splitext(self.image_paths[index])[0] + '_gt.npy'))
        print(os.path.join(label_folder, os.path.splitext(self.image_paths[index])[0] + '_gt.npy'))
        #label = np.load(os.path.join(label_folder, self.target_paths[index]))
        label = np.expand_dims(label, axis=0)
        label = torch.from_numpy(label)
        x, y = self.transform(image, label)
        return x, y

    def __len__(self):
        return len(self.image_paths)
