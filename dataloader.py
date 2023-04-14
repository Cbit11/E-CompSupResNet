import cv2 as cv
import torch
from torch.utils.data import Dataset,random_split, DataLoader
import os
import numpy as np 

class ImageDataset(Dataset):
    def __init__(self, lr_images_path, hr_images_path):
        self.lr_images_path = lr_images_path
        self.hr_images_path = hr_images_path

    def __len__(self):
        return len(os.listdir(self.lr_images_path))

    def __getitem__(self, idx):
        lr_img_path = os.path.join(self.lr_images_path, os.listdir(self.lr_images_path)[idx])
        hr_img_path = os.path.join(self.hr_images_path, os.listdir(self.hr_images_path)[idx])

        lr_img = np.array(cv.imread(lr_img_path))
        hr_img = np.array(cv.imread(hr_img_path))

        lr_img = torch.tensor(lr_img,dtype=torch.float,requires_grad=True)*(1/255)
        lr_img= torch.reshape(lr_img, (3,24,24))
        hr_img = torch.tensor(hr_img,dtype=torch.float, requires_grad=True)*(1/255)
        hr_img= torch.reshape(hr_img, (3,192,192))
        # lr_img = get_image_patches(hazy_img)
        # hazefree_img = get_image_patches(hazefree_img)

        return lr_img, hr_img