import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt

#=========================generate the npz data=================================
# data path
#low_path = r"D:\python\DPED-master\niidata\head_wire_dataset\raw_data\2nd\train\low"
#high_path = r"D:\python\DPED-master\niidata\head_wire_dataset\raw_data\2nd\train\high"
train_path = r"D:\python\DPED-master\niidata\head_wire_dataset\raw_data\2nd\train"

filenames = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path,f))]
i = 0
j = 0
low_datas = []
record = []
for file in filenames:
    if file[-7:] == 'low.nii':
        file_path = os.path.join(train_path,file)
        low_imgs = nib.load(file_path)
        low_data = low_imgs.dataobj
        low_data = np.transpose(low_data,(2,1,0))
        for img in low_data:
            if np.sum(img[0:100,-100:]) < 100 or np.sum(img[-100:,0:100]) < 100:
                record.append([i,j])

            j += 1
        low_datas.append(low_data)
    i += 1
low_datas = np.array(low_datas)
low_datas = np.reshape(low_datas,(-1,512,512))
print(low_datas.shape)
plt.imshow(low_datas[10],cmap="gray")
plt.show()
