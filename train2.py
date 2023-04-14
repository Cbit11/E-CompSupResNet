import os
import json
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,random_split
import sys
#sys.path.append(r"C:\Users\ratho\Desktop\summer_try2\dataset")
import dataloader
import torchvision
import model1
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import StructuralSimilarityIndexMeasure
import save_img


hr_img_pth= r"C:\Users\ratho\Desktop\summer_try2\hr_image"
lr_img_pth= r"C:\Users\ratho\Desktop\summer_try2\lr_image"
num_epochs= 40
batch_size=1

dataset=dataloader.ImageDataset(lr_img_pth, hr_img_pth)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
# use random_split to split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
val_loader =torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

l1_loss= torch.nn.L1Loss().to(device)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
mse= torch.nn.MSELoss()
def psnr(lr, hr):
    Psnr= 10*torch.log10(1/mse(lr,hr))
    return Psnr
def ssim_loss(lr, hr):
     ssim_l = 0.5*(1- ssim(lr,hr))
     return ssim_l
def criterion(lr, hr):
        #loss_ssim = 0.5*(ssim_loss(lr,hr))
        criterion = 10*l1_loss(lr, hr)+0.05*lpips(lr,hr)+0.5*ssim_loss(lr,hr)
        return criterion

model = model1.myNetwork().to(device)
print(model)
pram=0
print("Generator model - (prams-"+str(pram)+")\n")
print("\nTraining starts ... \n")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
save_dir = r"C:\Users\ratho\Desktop\summer_try2\save_img"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.to(device)
iter_count = 0
for epoch in range(num_epochs):
    for i, (lr, hr) in enumerate(train_loader):
        # print(os.getcwd())
        # break
        lpips_val= 0
        ssim_val= 0
        intermediate_psnr=0
        lr= lr.to(device)
        hr= hr.to(device)

        outputs= model(lr)

        loss = criterion(outputs, hr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        intermediate_psnr+= psnr(outputs,hr)
        ssim_val+= ssim(outputs, hr)
        lpips_val+= lpips(outputs, hr)
       
        if iter_count % 4 == 0:
            img = torchvision.utils.make_grid(outputs)
            img = img / 2 + 0.5  # unnormalize
            torchvision.utils.save_image(img, f'{save_dir}/epoch_{epoch}_iter_{i}.jpg')
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], LPIPS: {lpips_val.item():.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], ssim: {ssim_val.item():.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Psnr: {intermediate_psnr.item():.4f}')

        # Forward pass
        #images = lr.view(-1, 28 * 28)
        # lr= lr.to(device)
        # hr= hr.to(device)
        # outputs = model(lr)
        # # print(outputs.shape)
        # loss = criterion(outputs, hr)

        # # Backward and optimize
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # # Save images every 4 iterations
        # # if iter_count % 4 == 0:
        # #     img = torchvision.utils.make_grid(outputs)
        # #     img = img / 2 + 0.5  # unnormalize
        # #     torchvision.utils.save_image(img, f'{save_dir}/epoch_{epoch}_iter_{iter_count}.jpg')
        #     # img = ((outputs)*225.).detach().numpy().reshape([batch_size,192,192,3])
        #     # cv2.imwrite("outputs/real/"+str(epoch)+"_"+str(i)+"_"+".jpg",img)

        # iter_count += 1
        # intermediate_psnr+= psnr(outputs, hr)
        # lpips_val+=lpips(outputs,hr)
        # ssim_val+= ssim(outputs, hr)
        # # Print training statistics
        
        # if (i+1) % 10 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        #     print(str(i)+"th intermediate psnr value : "+str(intermediate_psnr/4))
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], LPIPS: {lpips_val.item():.4f}')
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], ssim: {ssim_val.item():.4f}')
