import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, Normalize
import numpy as np
from torchvision import utils
from torchvision import datasets, models, transforms
import os
# Create a dataset from ImageFolder

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'C:/Users/ddcfd/Desktop/vdeo'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# Save as npy files

train_data = []

for img, lbl in image_datasets['train']:
    train_data.append([np.array(img), np.array(lbl)])

train_data_np = np.array(train_data)
np.save('train_data' + '.npy', train_data_np)

val_data = []

for img, lbl in image_datasets['val']:
    val_data.append([np.array(img), np.array(lbl)])

val_data_np = np.array(val_data)
np.save('val_data' + '.npy', val_data_np)
