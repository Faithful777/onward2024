import os
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
        mask = TF.to_tensor(mask)
        return image, mask

    def __getitem__(self, index):
        print(self.image_paths[index])
        print(self.target_paths[index])
        image_folder = os.path.join(self.path, 'image')
        label_folder = os.path.join(self.path, 'label')
        image = Image.open(os.path.join(image_folder, self.image_paths[index]))
        label = np.load(os.path.join(label_folder, self.target_paths[index]))
        label = TF.to_tensor(label)
        x, y = self.transform(image, label)
        return x, y

    def __len__(self):
        return len(self.image_paths)
