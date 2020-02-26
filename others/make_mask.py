from skimage.io import imsave
import numpy as np 
import glob
import os


folder = '/groups/podgorski/podgorskilab/synapse_segmentation/processed_cellpose/train/'
file_name = glob.glob(folder+'*.npy')

for mask_file in file_name:
    mask_data = np.load(mask_file, allow_pickle=True).tolist()
    mask = mask_data['masks']
    mask[mask!=-1] = 255
    mask[mask==-1] = 0
    mask = np.uint8(mask)
    save_name = os.path.splitext(os.path.basename(mask_file))[0] + '.tif'
    imsave(folder+save_name, mask)