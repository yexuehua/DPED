import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

#generate the npzdata
"""
ti = 15
low_imgs = nib.load(r"D:\python\DPED-master\niidata\head_wire_dataset\raw_data\2nd\train\pa"+str(ti)+"_low.nii")
gt_imgs = nib.load(r"D:\python\DPED-master\niidata\head_wire_dataset\raw_data\2nd\train\pa"+str(ti)+"_high.nii")
# load image data
low_imgs_data = low_imgs.dataobj
# conver the shape [512 512 15] to [15 512 512]
low_imgs_data = np.transpose(low_imgs_data,(2,1,0))
gt_imgs_data = gt_imgs.dataobj
gt_imgs_data = np.transpose(gt_imgs_data,(2,1,0))

# place the ratated image in here temporarily
low_90s = []
gt_90s = []
# select the valid image
low = low_imgs_data[3:21]
gt = gt_imgs_data[3:21]

# augmentation by rotating 90,180,270 degrees
for i in range(3):
    for lowi in low:
        low_90 = np.rot90(lowi,i+1)
        low_90s.append(low_90)
low_90s = np.array(low_90s)
low_auged = np.concatenate((low,low_90s),0)

for i in range(3):
    for gti in gt:
        gt_90 = np.rot90(gti,i+1)
        gt_90s.append(gt_90)
gt_90s = np.array(gt_90s)
gt_auged = np.concatenate((gt,gt_90s),0)

low_processed = []
gt_processed = []

p_size = 100
for l_img in low_auged:
    h, w = np.array(l_img.shape) // 100
    for i in range(h):
        for j in range(w):
            low_proc = l_img[i * p_size:(i + 1) * p_size, j * p_size:(j + 1) * p_size]
            low_processed.append(low_proc)

for gt_img in gt_auged:
    hi, wi = np.array(gt_img.shape) // 100
    for i in range(hi):
        for j in range(wi):
            gt_proc = gt_img[i * p_size:(i + 1) * p_size, j * p_size:(j + 1) * p_size]
            gt_processed.append(gt_proc)
low_processed = np.array(low_processed)
gt_processed = np.array(gt_processed)


print(gt_processed.shape)
np.savez(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\2nd\pa"+str(ti)+"_low.npz",low=low_processed)
np.savez(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\2nd\pa"+str(ti)+"_high.npz",gt=gt_processed)
"""
# low = np.load(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\2nd\.npz")
# gt = np.load(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\2nd\.npz")
# low1 = np.load("low_quality1.npz")
# gt1 = np.load("ground_truth1.npz")
# low = low['low']
# gt = gt['gt']
# low1 = low1['low']
# gt1 = gt1['gt']
# low_all = np.concatenate((low,low1),0)
# gt_all = np.concatenate((gt,gt1),0)
path = r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\2nd"
low_all = []
gt_all = []
for file in os.listdir(path):
    file_path = os.path.join(path,file)
    if file[-7:] == "low.npz":
        low = np.load(file_path)
        low = low['low']
        low_all.append(low)
    if file[-8:] == "high.npz":
        gt = np.load(file_path)
        gt = gt['gt']
        gt_all.append(gt)

low_all = np.array(low_all)
lownpz = np.concatenate((low_all[0],low_all[1],low_all[2]),axis=0)
for im in range(3,12):
    lownpz = np.concatenate((lownpz,low_all[im]),axis=0)
gt_all = np.array(gt_all)
gtnpz = np.concatenate((gt_all[0],gt_all[1]),axis=0)
for jm in range(2,12):
    gtnpz = np.concatenate((gtnpz,gt_all[jm]),axis=0)
print(lownpz.shape)
print(gtnpz.shape)
np.savez(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\2nd\low_quality_all.npz",low=lownpz)
np.savez(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\2nd\ground_truth_all.npz",gt=gtnpz)
#
# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(low_all[150],cmap='gray')
# fig.add_subplot(1,2,2)
# plt.imshow(low_all[1650],cmap='gray')
# plt.show()
