from facenet_pytorch import MTCNN
from PIL import Image
import os

path = 'C:/Users/ddcfd/Desktop/vdeo/test'

mtcnn = MTCNN(device='cuda')

num_files = list(range(43))

save_paths = [f'detected_{i}.jpg' for i in num_files]

for file_, new_path in zip(sorted(os.listdir(path)), save_paths):
    if file_[-1] == 'g':
        im = Image.open(path + '/' + file_)
        mtcnn(im, save_path=new_path)
