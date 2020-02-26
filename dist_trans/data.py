from skimage.io import imread, imsave
from scipy.ndimage.morphology import distance_transform_cdt
import torch
from torch.utils.data import Dataset
import numpy as np 
from torchvision import transforms
import os
from skimage.filters import threshold_triangle


class FlipSample():
    """
    Randomly flip image and mask
    """
    def __call__(self, sample):
        img = sample[0]
        mask = sample[1]
        opt = np.random.randint(2)
        if opt:
            img = np.flip(img, axis=opt)
            mask = np.flip(mask, axis=opt)
        return img, mask


class RotSample():
    """
    Rotate image and mask with k (1,2,3) times of 90 degrees
    """
    def __call__(self, sample):
        img = sample[0]
        mask = sample[1]
        k = np.random.randint(4)
        if k:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
        return img, mask


class ToTensor():
    """
    Convert image and mask into tensor in shape [channel, x, y]
    """
    def __call__(self, sample):
        img = sample[0]
        mask = sample[1]
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        img = torch.from_numpy(img.copy())
        mask = torch.from_numpy(mask.copy())
        return img, mask


class GenerateData(Dataset):
    """
    Generate training and validation dataset using multiple image and mask pairs
    """
    def __init__(self, img_mask_name, crop_sz=(64,64), num_data=10000, transform=None):
        """
        Args:
        img_mask_name: list of name pairs of image and mask data, each pair is a tuple of (img_name, mask_name)
            For mask, 1-positive samples, 0-negative samples
        crop_sz: cropping size 
        num_data: generated sample size
        transform: data augmentation 
        """
        self.img_all = {}
        self.mask_all = {}
        self.seg_all = {}  # segmented image, used as a reference to generate training samples by random cropping
        self.num_pairs = len(img_mask_name)
        for i in range(self.num_pairs):
            curr_name = img_mask_name[i]
            assert os.path.exists(curr_name[0]) and os.path.exists(curr_name[1]), \
                'Image or mask does not exist!'
            img = imread(curr_name[0])
            thres = threshold_triangle(img)
            seg_img = np.zeros(img.shape, dtype=img.dtype)
            seg_img[img>thres] = 1
            self.seg_all[str(i)] = seg_img

            img = img.astype(float)
            mu = img.mean(axis=(1,2))
            sigma = img.std(axis=(1,2))
            img = (img - mu.reshape(len(mu),1,1)) / sigma.reshape(len(sigma),1,1)
            self.img_all[str(i)] = img

            msk_data = np.load(curr_name[1], allow_pickle=True).tolist()
            mask = msk_data['masks']
            mask[mask!=-1] = 1
            mask[mask==-1] = 0
            obj_dist = distance_transform_cdt(mask, metric='chessboard')
            bg = np.zeros(mask.shape, dtype=mask.dtype)
            bg[mask==0] = 1
            bg_dist = distance_transform_cdt(bg, metric='chessboard')
            bg_dist[bg_dist>4] = 4
            mask_dist = (obj_dist - bg_dist).astype(float)

            self.mask_all[str(i)] = mask_dist

        self.crop_sz = crop_sz
        self.num_data = num_data
        self.transform = transform
    
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        pair_idx = np.random.randint(self.num_pairs)
        img = self.img_all[str(pair_idx)]
        mask = self.mask_all[str(pair_idx)]
        seg = self.seg_all[str(pair_idx)]
        sz = img.shape
        slice_idx = np.random.randint(sz[0])
        img_slice = img[slice_idx,:,:]
        mask_slice = mask[slice_idx,:,:]
        seg_slice = seg[slice_idx,:,:]

        # generate data with acceptance rate
        is_accept = False
        while is_accept is not True:
            accept_prob = np.random.random()
            loc_x = np.random.randint(0, sz[1]-self.crop_sz[0])
            loc_y = np.random.randint(0, sz[2]-self.crop_sz[1])
            sample_mask = mask_slice[loc_x:loc_x+self.crop_sz[0], loc_y:loc_y+self.crop_sz[1]]
            sample_seg = seg_slice[loc_x:loc_x+self.crop_sz[0], loc_y:loc_y+self.crop_sz[1]]
            if np.count_nonzero(sample_mask) > 10 or np.count_nonzero(sample_seg) > 200 or accept_prob > 0.99:
                is_accept = True
                sample_img = img_slice[loc_x:loc_x+self.crop_sz[0], loc_y:loc_y+self.crop_sz[1]]    
        
        # data augmentation
        if self.transform is not None:
            sample_img, sample_mask = self.transform([sample_img, sample_mask])
        
        return sample_img, sample_mask


if __name__ == "__main__":

    def view_data(save_name, img):
        img = img.numpy()
        img_ch0 = np.zeros(img.shape[1:], dtype=img.dtype)
        img_ch0 = np.float32(img[0,:,:])
        imsave(save_name, img_ch0)
        return None

    data_path = '/groups/podgorski/podgorskilab/synapse_segmentation/processed_cellpose/train/'
    img_mask_name = [(data_path+'refPRE2__00001_REF_Ch1_1127503.tif', data_path+'refPRE2__00001_REF_Ch1_1127503_manual.npy'),(data_path+'refPRE2__00001_REF_Ch1_0927C.tif', data_path+'refPRE2__00001_REF_Ch1_0927C_manual.npy')]
    Data = GenerateData(img_mask_name, transform=transforms.Compose([FlipSample(), RotSample(), ToTensor()]))

    print(len(Data))

    data = Data.__getitem__(100)
    print(len(data))
    img = data[0]
    mask = data[1]
    print(img.shape)
    print(img.dtype)
    print(mask.shape)
    print(mask.dtype)

    save_path = '/groups/podgorski/podgorskilab/synapse_segmentation/dingx/'
    view_data(save_path+'img.tif', img)
    view_data(save_path+'mask.tif', mask)