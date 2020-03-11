from skimage.io import imread, imsave
import glob
import numpy as np
from model import UNet 
from modules import NeuralNetwork
import torch
import torch.nn as nn 
import torch.optim as optim
import os 
import matplotlib.pyplot as plt 
import json


model_path = '/groups/scicompsoft/home/preibischs/conda-projects/Kaspar_data/models/unet_depth3_disttrans/'

with open(model_path+'loss.json', 'r') as f:
    loss = json.load(f)
train_loss_total = loss['train_loss_total']
eval_loss_total = loss['eval_loss_total']
plt.figure()
plt.plot(range(len(train_loss_total)), train_loss_total, 'k', label='Train')
plt.plot(range(len(eval_loss_total)), eval_loss_total, 'b', label='Test')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.savefig(model_path+'/loss.pdf')
plt.show()

# test data
data_path = '/groups/scicompsoft/home/preibischs/conda-projects/Kaspar_data/test/'
img_files = glob.glob(data_path+'*.tif')

# checkpoint
ckpt_list = ['model_ckpt_50.pt', 'model_ckpt_100.pt']

# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth = 3
model = UNet(in_channels=1, base_filters=16, out_channels=1, depth=depth)
if torch.cuda.device_count()>1:
    print('---Using {} GPUs---'.format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
model.to(device)
# criterion
criterion = nn.MSELoss(reduction='mean')
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=0.00005, nesterov=True)

network = NeuralNetwork(model, criterion, optimizer, device)
input_sz = (108, 108) 
step = (68, 68)

# Load checkpoints
for ckpt in ckpt_list:
    print("Test checkpoint {}".format(ckpt))
    for idx in range(len(img_files)):
        curr_name = img_files[idx]
        img = imread(curr_name).astype(float)
        mu = img.mean(axis=(1,2))
        sigma = img.std(axis=(1,2))
        img = (img - mu.reshape(len(mu),1,1)) / sigma.reshape(len(sigma),1,1)
        result = np.zeros(img.shape, dtype='float32')#'uint8')
        for i in range(img.shape[0]):
            curr_slice = img[i,:,:]
            out = network.test_model(model_path+ckpt, curr_slice, input_sz, step)
            #out[out>0]=255
            #out[out<=0]=0
            #out = np.uint8(out)
            result[i,:,:] = out
        save_name = 'result_{}_{}.tif'.format(os.path.splitext(ckpt)[0], os.path.splitext(os.path.basename(curr_name[0]))[0])
        imsave(data_path+save_name, result)