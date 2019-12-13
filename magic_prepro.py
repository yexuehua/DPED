import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
# np.set_printoptions(threshold=np.nan)

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


top_path = r"D:\python\DPED-master\niidata\head_wire_dataset\raw_data\youyi_magic"

selected_num = np.arange(0,96,4)
selected_file = []
for i in selected_num:
    selected_file.append("IM"+str(i))
count = 0
hb = []
acc = []
for root,dir,file in os.walk(top_path):
    if len(file) != 0:
        images = []
        for img in selected_file:
            img_path = os.path.join(root,img)
            img = pydicom.read_file(img_path)
            image = img.pixel_array
            #image = normal(image)
            if root[-4:] == "acc4":
                image = misc.imresize(image,(512,512))
                image = image.astype(np.int16)
            images.append(image)
        if root[-2:] == "hb":
            hb.append(images)
        if root[-4:] == "acc4":
            acc.append(images)
hb = np.array(hb)
hb = np.reshape(hb,(-1,512,512))

acc = np.array(acc)
acc = np.reshape(acc,(-1,512,512))


# with open("hb2.txt",'w') as f:
#     for item in hb:
#         f.write("%s\n" % item)

low = acc
gt = hb

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

# low_processed = []
# gt_processed = []
#
# p_size = 100
# for l_img in low_auged:
#     h, w = np.array(l_img.shape) // 100
#     for i in range(h):
#         for j in range(w):
#             low_proc = l_img[i * p_size:(i + 1) * p_size, j * p_size:(j + 1) * p_size]
#             low_processed.append(low_proc)
#
# for gt_img in gt_auged:
#     hi, wi = np.array(gt_img.shape) // 100
#     for i in range(hi):
#         for j in range(wi):
#             gt_proc = gt_img[i * p_size:(i + 1) * p_size, j * p_size:(j + 1) * p_size]
#             gt_processed.append(gt_proc)
# low_processed = np.array(low_processed)
# gt_processed = np.array(gt_processed)

# print(gt_processed.dtype)
# print(low_processed.dtype)
# print(gt_processed.shape)
# print(low_processed.shape)
np.savez(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\youyi_magic\all_lowfull.npz",low=low_auged)
np.savez(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\youyi_magic\all_highfull.npz",gt=gt_auged)

# misc.imresize(acc,(240,512,512))
# hb1 = hb[126]
# acc1 = acc[35]

low_pro = np.load(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\youyi_magic\all_lowfull.npz")
gt_pro = np.load(r"D:\python\DPED-master\niidata\head_wire_dataset\npz_data\youyi_magic\all_highfull.npz")
low_processed = low_pro['low']
gt_processed = gt_pro['gt']
print(low_processed.shape)
print(gt_processed.shape)
a = low_processed[100]
b = gt_processed[100]

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(low_processed[102],cmap='gray')
fig.add_subplot(1,2,2)
plt.imshow(gt_processed[102],cmap='gray')
# fig.add_subplot(1,3,3)
# plt.imshow(misc.imresize(acc[125],(512,512)),cmap='gray')
# plt.show()
# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(low_processed[20052], cmap="gray")
# fig.add_subplot(1,2,2)
# plt.imshow(gt_processed[20052], cmap="gray")
plt.show()
