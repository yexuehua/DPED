import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from tqdm import tqdm
# np.set_printoptions(threshold=np.nan)

# the method of normalization,and restore the window width and window position
class nomalprocess():
    #              1 --> 2
    # (x-min1)(max2-min2)/(max1-min1)+min2
    def normal(self,pixels):
        self.mins = np.min(pixels)
        self.maxs = np.max(pixels)
        pixels = (255/(self.maxs-self.mins))*(pixels-self.mins)
        return pixels
#pixels = ((self.maxs-self.mins)/255)*pixels+self.mins
	# def denomal(self,pixels):
    #     pixels = ((self.maxs-self.mins)/255)*pixels+self.mins
    def denomal(self,pixels):
        pixels = ((self.maxs-self.mins)/255)*pixels+self.mins
        return pixels

def normal(pixels):
    nmins = np.min(pixels)
    nmaxs = np.max(pixels)
    pixels = (pixels-nmins)*(255/(nmaxs-nmins))
    return pixels

# define the path of the raw data
top_path = r"D:\python\DPED-master\dicom_data\youyi_dicomB"

#============step1: generate the npz of ACC4 and HB dataset=======================
# name the dataset
HB = []
ACC4 = []
# recursive the whole folder,store the correspoding file
# for root,dir,file in os.walk(top_path):
#     if len(file)!= 0:
#         file.sort(key=lambda x:int(x[2:]))
#         print(file)
#         print(root)
#         if root[-4:] == 'acc4':
#             for img in file:
#                 img_path = os.path.join(root,img)
#                 img_dcm = pydicom.read_file(img_path)
#                 img_data = img_dcm.pixel_array
#                 img_data = misc.imresize(img_data,(512,512))
#                 img_data = img_data.astype(np.int16)
#                 ACC4.append(img_data)
#         if root[-2:] == 'hb':
#             for img in file:
#                 img_path = os.path.join(root,img)
#                 img_dcm = pydicom.read_file(img_path)
#                 img_data = img_dcm.pixel_array
#                 HB.append(img_data)
# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(ACC4[1053],cmap='gray')
# fig.add_subplot(1,2,2)
# plt.imshow(HB[1053],cmap='gray')
# plt.show()
# ACC4 = np.array(ACC4)
# HB = np.array(HB)



# ACC4 = normal(ACC4)
# HB = normal(HB)
# ACC4 = ACC4.astype(np.int8)
# HB = HB.astype(np.int8)
# plt.ion()
# for i in range(len(ACC4)):
#     fig = plt.figure()
#     fig.add_subplot(1,2,1)
#     plt.imshow(ACC4[i],cmap='gray')
#     fig.add_subplot(1,2,2)
#     plt.imshow(HB[i],cmap='gray')
#     #plt.show()
#     plt.pause(0.5)
#     plt.close()
#
# np.savez(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\youyi_magic\DicomB_ACC4.npz",low=ACC4)
# np.savez(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\youyi_magic\DIcomB_HB.npz",gt=HB)


# import cv2
# def nothing(emp):
#     pass
#
# def jindu_imgpath(name,frames):
#     cv2.namedWindow(name,0)
#     cv2.resizeWindow(name, 800, 600)
#     loop_flag = 0
#     pos = 0
#     cv2.createTrackbar('time', name, 0, frames, nothing)
#     while True:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.waitKey(0)
#         if loop_flag == pos:
#             loop_flag = loop_flag + 1
#             cv2.setTrackbarPos('time', name, loop_flag)
#         else:
#             pos = cv2.getTrackbarPos('time', name)
#             loop_flag = pos
#         img1_new = ACC4[loop_flag]
#         img2_new = HB[loop_flag]
#         img_new = np.hstack([img1_new,img2_new])
#         cv2.imshow(name, img_new)
#
# jindu_imgpath('img', len(ACC4)-1)


#==============step2: extract every echoes in each npz file,the save as npz file respectively================
'''
# low_pro = np.load(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\youyi_magic\ACC4.npz")
# gt_pro = np.load(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\youyi_magic\HB.npz")
# low_processed = low_pro['low']
# gt_processed = gt_pro['gt']
# #low_processed = normal(low_processed)
# # gt_processed = normal(gt_processed)
# #
# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(low_processed[1055],cmap='gray')
# fig.add_subplot(1,2,2)
# plt.imshow(gt_processed[1055],cmap='gray')
# plt.show()

# # saving the vary echo to npz
# A01_syn = []
# B01_syn = []
#
# A02_syn = []
# B02_syn = []
#
# A03_syn = []
# B03_syn = []
#
# A04_syn = []
# B04_syn = []
#
# for p in tqdm(range(10)):
#     for itr in range(24):
#         A01_syn.append((gt_processed[384*p+4*itr+0]**2+gt_processed[384*p+4*itr+1]**2)**0.5)
#         B01_syn.append((gt_processed[384*p+4*itr+2]**2+gt_processed[384*p+4*itr+3]**2)**0.5)
#         A02_syn.append((gt_processed[384*p+4*itr+96]**2+gt_processed[384*p+4*itr+97]**2)**0.5)
#         B02_syn.append((gt_processed[384*p+4*itr+98]**2+gt_processed[384*p+4*itr+99]**2)**0.5)
#         A03_syn.append((gt_processed[384*p+4*itr+96*2+0]**2+gt_processed[384*p+4*itr+96*2+1]**2)**0.5)
#         B03_syn.append((gt_processed[384*p+4*itr+96*2+2]**2+gt_processed[384*p+4*itr+96*2+3]**2)**0.5)
#         A04_syn.append((gt_processed[384*p+4*itr+96*3+0]**2+gt_processed[384*p+4*itr+96*3+1]**2)**0.5)
#         B04_syn.append((gt_processed[384*p+4*itr+96*3+2]**2+gt_processed[384*p+4*itr+96*3+3]**2)**0.5)
#
# quality = 'high'
# np.savez("D:/python/DPED-master/niidata/head_wire_dataset/npz_data/youyi_magic/"+quality+"A01_syn.npz",data=A01_syn)
# np.savez("D:/python/DPED-master/niidata/head_wire_dataset/npz_data/youyi_magic/"+quality+"B01_syn.npz",data=B01_syn)
# np.savez("D:/python/DPED-master/niidata/head_wire_dataset/npz_data/youyi_magic/"+quality+"A02_syn.npz",data=A02_syn)
# np.savez("D:/python/DPED-master/niidata/head_wire_dataset/npz_data/youyi_magic/"+quality+"B02_syn.npz",data=B02_syn)
# np.savez("D:/python/DPED-master/niidata/head_wire_dataset/npz_data/youyi_magic/"+quality+"A03_syn.npz",data=A03_syn)
# np.savez("D:/python/DPED-master/niidata/head_wire_dataset/npz_data/youyi_magic/"+quality+"B03_syn.npz",data=B03_syn)
# np.savez("D:/python/DPED-master/niidata/head_wire_dataset/npz_data/youyi_magic/"+quality+"A04_syn.npz",data=A04_syn)
# np.savez("D:/python/DPED-master/niidata/head_wire_dataset/npz_data/youyi_magic/"+quality+"B04_syn.npz",data=B04_syn)
'''

#=====================Test the saved npz file========================================
low = np.load("D:/python/DPED-master/niidata/head_wire_dataset/npz_data/youyi_magic/low_synthetic/lowA01_syn.npz")
high = np.load("D:/python/DPED-master/niidata/head_wire_dataset/npz_data/youyi_magic/high_synthetic/highA01_syn.npz")
low_data = low["data"]
high_data = high["data"]
# print(low_data.shape)
# rm = np.arange(144,168,1)
# low_data = np.delete(low_data,rm,axis=0)
# high_data = np.delete(high_data,rm,axis=0)
# rm2 = np.arange(0,240,24)
# rm3 = rm2-1
# rm3 = np.concatenate([rm2,rm3])
# rm3 = np.sort(rm3)
# rm3 = rm3[1:]
# low_data = np.delete(low_data,rm3,axis=0)
# high_data = np.delete(high_data,rm3,axis=0)
# low_data = np.delete(low_data,59,axis=0)
# high_data = np.delete(high_data,59,axis=0)

# print(np.min(low_data),np.min(high_data))
# print(np.max(low_data),np.max(high_data))

# print(low_data[0],high_data[0])
low_data = normal(low_data)
high_data = normal(high_data)
low_data = normal(low_data)
high_data = normal(high_data)
low_data = np.uint8(low_data)
high_data = np.uint8(high_data)


# visualize the slice by opencv
''''''
import cv2

# low_data = cv2.normalize(low_data,dst=None)
# high_data = cv2.normalize(high_data,dst=None)
# print(low_data.dtype)
print(np.min(low_data),np.max(low_data))

def nothing(emp):
    pass

def jindu_imgpath(name,frames):
    cv2.namedWindow(name,0)
    cv2.resizeWindow(name, 800, 600)
    loop_flag = 0
    pos = 0
    cv2.createTrackbar('time', name, 0, frames, nothing)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.waitKey(0)
        pos = cv2.getTrackbarPos('time', name)
        img1_new = low_data[pos]
        img2_new = high_data[pos]
        img_new = np.hstack([img1_new,img2_new])
        cv2.imshow(name, img_new)

jindu_imgpath('img', len(low_data)-1)
''''''

# visualize the slice by plt
'''
# for i in range(100,len(low_data)):
#     fig = plt.figure()
#     fig.add_subplot(1,2,1)
#     plt.imshow(low_data[i],cmap='gray')
#     fig.add_subplot(1,2,2)
#     plt.imshow(high_data[i],cmap='gray')
#     plt.pause(1)
#     plt.close()
'''

# augment the dataset
''''
low = low_data
gt = high_data

low_90s = []
gt_90s = []
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
np.savez(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\youyi_magic\all_lowsyn.npz",low=low_processed)
np.savez(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\youyi_magic\all_highsyn.npz",gt=gt_processed)
'''

# low_pro1 = np.load(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\youyi_magic\all_lowsyn.npz")
# gt_pro1 = np.load(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\youyi_magic\all_highsyn.npz")
# low_processed1 = low_pro1['low']
# gt_processed1 = gt_pro1['gt']
# print(low_processed1.shape)
# print(gt_processed1.shape)
#
# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(low_processed1[102],cmap='gray')
# fig.add_subplot(1,2,2)
# plt.imshow(gt_processed1[102],cmap='gray')
# plt.show()
#===============Addition: visualize the result======================================

# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(low_data[75],cmap='gray')
# fig.add_subplot(1,2,2)
# plt.imshow(high_data[75],cmap='gray')
# plt.show()
