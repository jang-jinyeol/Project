from facenet_pytorch import MTCNN
import facedetector_m
import os
import torch
from torchvision import models
import torch.nn as nn
import argparse

# Construct the argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--blur", type=bool, default=True, choices=[True,False])
args = vars(ap.parse_args())

############### INITIATE CLASSIFIER #############
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, 2), torch.nn.Sigmoid())

model.load_state_dict(torch.load('model-resnet18-2.pth'))
model.eval()

######### INITIATE NETWORKS ######
#mtcnn = MTCNN()
mtcnn = MTCNN(device='cuda')

fcd = facedetector_m.FaceDetectorClass(mtcnn, classifier=model)

if args["blur"] == True:
    fcd.run(blur_setting=True)
if args["blur"] == False:
    fcd.run(blur_setting=False)
