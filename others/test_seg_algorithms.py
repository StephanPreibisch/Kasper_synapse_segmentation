from skimage.filters import threshold_otsu, threshold_yen, threshold_isodata, threshold_li
from skimage.filters import threshold_mean, threshold_niblack, threshold_sauvola, threshold_triangle
from skimage.io import imread, imsave
import numpy as np 


folder_path = "/groups/podgorski/podgorskilab/synapse_segmentation/processed_cellpose/train/"

img = imread(folder_path+'refPRE2__00001_REF_Ch1_0927C.tif')

thres_otsu = threshold_otsu(img)
img_otsu = np.zeros(img.shape, dtype=img.dtype)
img_otsu[img>thres_otsu] = 255
imsave(folder_path+"img_otsu.tif", img_otsu)

thres_yen = threshold_yen(img)
img_yen = np.zeros(img.shape, dtype=img.dtype)
img_yen[img>thres_yen] = 255
imsave(folder_path+"img_yen.tif", img_yen)

thres_isodata = threshold_isodata(img)
img_isodata = np.zeros(img.shape, dtype=img.dtype)
img_isodata[img>thres_isodata] = 255
imsave(folder_path+"img_isodata.tif", img_isodata)

thres_li = threshold_li(img)
img_li = np.zeros(img.shape, dtype=img.dtype)
img_li[img>thres_li] = 255
imsave(folder_path+"img_li.tif", img_li)

thres_mean = threshold_mean(img)
img_mean = np.zeros(img.shape, dtype=img.dtype)
img_mean[img>thres_mean] = 255
imsave(folder_path+"img_mean.tif", img_mean)

thres_niblack = threshold_niblack(img)
img_niblack = np.zeros(img.shape, dtype=img.dtype)
img_niblack[img>thres_niblack] = 255
imsave(folder_path+"img_niblack.tif", img_niblack)

thres_sauvola = threshold_sauvola(img)
img_sauvola = np.zeros(img.shape, dtype=img.dtype)
img_sauvola[img>thres_sauvola] = 255
imsave(folder_path+"img_sauvola.tif", img_sauvola)

thres_triangle = threshold_triangle(img)
img_triangle = np.zeros(img.shape, dtype=img.dtype)
img_triangle[img>thres_triangle] = 255
imsave(folder_path+"img_triangle.tif", img_triangle)