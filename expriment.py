'''
# from scipy import misc
# import pydicom
# import numpy as np
#
# def normal(pixels):
#     high,wid = pixels.shape
#     mins = np.min(pixels)
#     maxs = np.max(pixels)
#     for i in range(high):
#         for j in range(wid):
#             pixels[i][j] = int((pixels[i][j]-mins)*(255/(maxs-mins)))
#     return pixels
#
# img = pydicom.read_file(r'D:\yexuehua\download\MR_Enhance\1_75815001\T1FLAIR_08.dcm')
# #inputs = img.pixel_array
# #inputs = normal(inputs)
# #misc.imsave('IM42.jpg',inputs)
# a = img.PixelData
# b = bytearray(a)
# print(img)#dir(img))
#
# # print(b.decode())
'''

# Change the kspace information
'''
import numpy as np
import pydicom
from scipy import misc
import utils
# img = pydicom.read_file(r'D:\yexuehua\data\MR_Enhance\1029921A3D\1022921A3D\IM65')
# data = img.pixel_array
#print(np.max(data))
data = misc.imread('liver_a_original.jpg')
shape = data.shape
#misc.imsave('liver_a_original.jpg',data)
kspace_img = np.fft.fftshift(np.fft.ifft2(data))
#maxk = np.max(kspace_img)
#misc.imsave('liver_a_kspace.jpg',abs(kspace_img))
kspace_img[0:round(shape[0]*0.9/2),:] = 0
kspace_img[(shape[0]-round(shape[0]*0.9/2)):shape[0],:] = 0

out = abs(np.fft.fft2(kspace_img))
misc.imsave('liver_a_10%.png',out)
'''

# read nii file and process it
'''
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

def k_downgrade(img):
    kspace = np.fft.fftshift(np.fft.ifft2(img))
    h,w = img.shape
    kspace[0:round(h*0.9/2),:] = 0
    kspace[h-round(h*0.9/2):h,:] = 0
    out = np.fft.fft2(kspace)
    return abs(out)

def zero_normal(img):
    mean = np.mean(img)
    std = np.std(img)
    normal = (img-mean)/std
    return normal
#top_path =
img = nib.load(r'D:\yexuehua\data\MRBrain_TrainData\zhaohongdeM686631_t1ce.nii')
img2 = img.dataobj
datas = np.array(img.dataobj)
datas = np.transpose(datas,(2,1,0))
postdata = []
for data in datas:
    processed = k_downgrade(data)
    postdata.append(processed)

# show the effect of k_space process
# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(postdata[10],cmap='gray')
# fig.add_subplot(1,2,2)
# plt.imshow(datas[10],cmap='gray')
# plt.show()

new_data = zero_normal(datas[18])
print(np.max(datas[12]),'\n',np.min(datas[12]))
print(np.max(new_data),'\n',np.min(new_data))
print(datas[18].shape)
'''

# find the image which is the size of 512*512
'''
import nibabel as nib
import numpy as np
import os

top_path = 'D:\yexuehua\data\MRBrain_TrainData'
patients = os.listdir(top_path)

for patient in patients:
    patient_path = os.path.join(top_path,patient)
    img = nib.load(patient_path)
    img_datas = np.array(img.dataobj)
    img_datas = np.transpose(img_datas,(2,1,0))
    print(patient,'\n',img_datas[2].shape,'\n')
'''

# saved as dataset to npz file
'''
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# img = nib.load(r'D:\yexuehua\data\MRBrain_TrainData\zhaohongdeM686631_t1ce.nii')
# img2 = img.dataobj
# datas = np.array(img.dataobj)
# datas = np.transpose(datas,(2,1,0))
# np.savez('zhao.npz',datas)
imge = np.load('zhao.npz')

plt.imshow(imge['arr_0'][10])
plt.show()
'''

# make a 100*100 patch dataset 10%-100%
'''
import nibabel as nib
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy import misc

global p_size
p_size = 100

# def k_downgrade(img):
#     kspace = np.fft.fftshift(np.fft.ifft2(img))
#     h,w = img.shape
#     kspace[0:round(h*0.9/2),:] = 0
#     kspace[h-round(h*0.9/2):h,:] = 0
#     out = np.fft.fft2(kspace)
#     return abs(out)

def k_downgrade(img,i):
    kspace = np.fft.fftshift(np.fft.ifft2(img))
    h,w = img.shape
    kspace[0:round(h*0.1*(10-i)/2),:] = 0
    kspace[h-round(h*0.1*(10-i)/2):h,:] = 0
    out = np.fft.fft2(kspace)
    return abs(out)


def zero_normal(img):
    mean = np.mean(img)
    std = np.std(img)
    normal = (img-mean)/std
    return normal

def normal(pixels):
    mins = np.min(pixels)
    maxs = np.max(pixels)
    pixels = (pixels-mins)*(255/(maxs-mins))
    return pixels


def get_patch(img,k_img,h_id,w_id):
    # ran = random.uniform(0,1)
    #ran>0.5:#adequatelly crop the image
    patch = img[h_id*p_size:(h_id+1)*p_size,w_id*p_size:(w_id+1):p_size]
    k_patch = k_img[h_id*p_size:(h_id+1)*p_size,w_id*p_size:(w_id+1):p_size]
    # else:
    #     patch = img[-1+(-1)*(h_id+1)*p_size:-1+(-1)*h_id*p_size,\
    #             -1+(-1)*(h_id+1)*p_size:-1+(-1)*h_id*p_size]
    #     k_patch = k_img[-1+(-1)*(h_id+1)*p_size:-1+(-1)*h_id*p_size,\
    #             -1+(-1)*(h_id+1)*p_size:-1+(-1)*h_id*p_size]
    return patch,k_patch
start_time = time.time()
top_path = 'D:\yexuehua\data\k_space_data\liver\A3D'
patients = os.listdir(top_path)
p = 0
for patient in patients:
    patient_path = os.path.join(top_path,patient)
    img = nib.load(patient_path)
    img_datas = np.array(img.dataobj)
    img_datas = np.transpose(img_datas,(2,1,0))
    print(patient)
    for img in tqdm(img_datas):
        h,w = np.array(img.shape)//100
        #k_img = k_downgrade(img)
        if np.min(img)<0 | np.max(img)==0:
            continue
        else:
            for i in range(h):
                for j in range(w):
                    orig = img[i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]
                    proc = k_img[i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]
                    for persent in range(10):
                        proce = k_downgrade(proc,persent+1)
                        proce = normal(proce)
                        #using zero normalization to normalize the image
                        out_paths = 'D:/python/DPED-master/dped/blackberry/training_data/A3D'
                        if not os.path.exists(out_paths):
                            os.mkdir(out_paths)
                        misc.imsave('D:/python/DPED-master/dped/blackberry/training_data/blackberry'+str(persent+1)+'/'+str(p)+'.png',proce)
                    p=p+1

# original = np.array(original)
# processd = np.array(processd)
# np.savez('MRI_dataset.npz',original=original,processd=processd)

# end_time = time.time()
# print("the whole time is",end_time-start_time)
# print(original.shape)
# print(processd.shape)
# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(original[600],cmap='gray')
# fig.add_subplot(1,2,2)
# plt.imshow(processd[600],cmap='gray')
# plt.show()
# print(processd.shape)
'''

# use the MRI.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import misc

start_time = time.time()
img = np.load('liver.npz')
pro = img['processd']

print(len(pro))
img = np.load('liver_orig.npz')
orig = img['original']
print(len(orig))
#orign = np.array([pro[9+10*i] for i in range(len(pro)//10)])
# pro1 = np.array([pro[0+10*i] for i in range(len(pro)//10)])
# pro2 = np.array([pro[1+10*i] for i in range(len(pro)//10)])
# orign = np.expand_dims(orign,axis=3)
# orign = np.concatenate((orign,orign,orign),axis=-1)
#
# pro1 = np.expand_dims(pro1,axis=3)
# pro1 = np.concatenate((pro1,pro1,pro1),axis=-1)
# misc.imsave('pro1.png',pro1[262])

fig = plt.figure()
end_time = time.time()
#print(ori.shape,pro.shape)
print("the whole time is",end_time-start_time)
fig.add_subplot(1,3,1)
plt.imshow(orig[36],cmap='gray')
fig.add_subplot(1,3,2)
plt.imshow(orig[37],cmap='gray')
fig.add_subplot(1,3,3)
plt.imshow(pro[37],cmap='gray')
plt.show()
'''

'''
from load_dataset import load_data
import matplotlib.pyplot as plt
import numpy as np
dataset  = 'MRI_dataset.npz'
test_data, test_answ,train_data, train_answ = load_data(dataset,0.1)

print(test_data.shape,train_data.shape)
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(np.reshape(train_data[702],[100,100,3]),cmap='gray')
fig.add_subplot(1,2,2)
plt.imshow(np.reshape(train_answ[702],[100,100,3]),cmap='gray')
plt.show()

'''

# generate 0-90% k_sapce image
'''
import numpy as np
import pydicom
from scipy import misc
img = pydicom.read_file(r'D:\python\DPED-master\visual_results\previous\liver k_space\P3D_k_dcm\100\100%_IM259')
data = img.pixel_array
#print(np.max(data))
shape = data.shape
#misc.imsave('brain_original.jpg',data)
kspace_img = np.fft.fftshift(np.fft.ifft2(data))
#maxk = np.max(kspace_img)
#misc.imsave('brain_kspace.jpg',abs(kspace_img))
for i in range(10):
    removal = kspace_img
    removal[0:round(shape[0]*0.1*i/2),:] = 0
    removal[(shape[0]-round(shape[0]*0.1*i/2)):shape[0],:] = 0
    out = abs(np.fft.fft2(removal))
    misc.imsave('fft/liver_pd_fft'+str((10-i))+'0.jpg',out)
'''

# generate 0-90% k_space DCM
'''
import numpy as np
import pydicom
from scipy import misc
import os

top_path = r'D:\yexuehua\data\k_space_data\liver imaging poor\01813792\01813792P3D'
dcms = os.listdir(top_path)
out_path = os.path.join(top_path,'k_dcm')
if not os.path.exists(out_path):
    os.mkdir(out_path)

for img in dcms:
    img_path = os.path.join(top_path,img)
    imge = pydicom.read_file(img_path)
    data = imge.pixel_array
    #print(np.max(data))
    shape = data.shape
    #misc.imsave('brain_original.jpg',data)
    kspace_img = np.fft.fftshift(np.fft.ifft2(data))
    #maxk = np.max(kspace_img)
    #misc.imsave('brain_kspace.jpg',abs(kspace_img))
    for i in range(10):
        out_paths = os.path.join(out_path,str((10-i)*10))
        if not os.path.exists(out_paths):
            os.mkdir(out_paths)
        removal = kspace_img
        removal[0:round(shape[0]*0.1*i/2),:] = 0
        removal[(shape[0]-round(shape[0]*0.1*i/2)):shape[0],:] = 0
        out = abs(np.fft.fft2(removal))
        out = np.uint16(out)
        out_name = str((10-i)*10)+'%_'+img
        out_name = os.path.join(out_paths,out_name)
        imge.PixelData = out.tobytes()
        imge.save_as(out_name)
'''


# make 10-100% dataset
'''
import nibabel as nib
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy import misc

global p_size
p_size = 100

def k_downgrade(img,i):
    kspace = np.fft.fftshift(np.fft.ifft2(img))
    h,w = img.shape
    kspace[0:round(h*0.1*(10-i)/2),:] = 0
    kspace[h-round(h*0.1*(10-i)/2):h,:] = 0
    out = np.fft.fft2(kspace)
    return abs(out)

def zero_normal(img):
    mean = np.mean(img)
    std = np.std(img)
    normal = (img-mean)/std
    return normal

def normal(pixels):
    high,wid = pixels.shape
    mins = np.min(pixels)
    maxs = np.max(pixels)
    pixels = (pixels-mins)*(255/(maxs-mins))
    return pixels


def get_patch(img,k_img,h_id,w_id):
    patch = img[h_id*p_size:(h_id+1)*p_size,w_id*p_size:(w_id+1):p_size]
    k_patch = k_img[h_id*p_size:(h_id+1)*p_size,w_id*p_size:(w_id+1):p_size]
    return patch,k_patch


start_time = time.time()
top_path = 'D:\yexuehua\data\k_space_data\liver\A_P'
patients = os.listdir(top_path)
processd = []
original = []
for patient in patients:
    patient_path = os.path.join(top_path,patient)
    img = nib.load(patient_path)
    img_datas = np.array(img.dataobj)
    img_datas = np.transpose(img_datas,(2,1,0))
    for img in tqdm(img_datas):
        h,w = np.array(img.shape)//100
        if np.min(img)<0 | np.max(img)==0:
            print("there is a negative value")
            continue
        else:
            for i in range(h):
                for j in range(w):
                    orig = img[i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]
                    #proc = img[i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]
                    for persent in range(3):
                        original.append(orig)
                        #proce = k_downgrade(proc,persent+1)
                        #processd.append(proce)
                    #orig = normal(orig)/255#using zero normalization to normalize the image
                    #proc = normal(proc)/255
                    #misc.imsave('D:/python/DPED-master/dped/iphone/training_data/canon'+'/'+str(p)+'.jpg',orig)
                    #misc.imsave('D:/python/DPED-master/dped/iphone/training_data/blackberry'+'/'+str(p)+'.jpg',proc)
                    #original.append(orig)

original = np.array(original)
#processd = np.array(processd)
#np.savez('liver.npz',processd=processd)
np.savez('liver_orig.npz',original=original)
# end_time = time.time()
# print("the whole time is",end_time-start_time)
# print(original.shape)
# print(processd.shape)
# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(original[600],cmap='gray')
# fig.add_subplot(1,2,2)
# plt.imshow(processd[600],cmap='gray')
# plt.show()
# print(original.shape)
# print(processd.shape)
'''


'''

import nibabel as nib
import numpy as np
import random
import utils
import time
import os
import matplotlib.pyplot as plt
data = np.load('MRI_dataset_all.npz')
processd = []
proc = data['processd']
original = np.array([proc[9+10*i] for i in range(len(proc)//10)])
original = utils.normal(original)
processd = np.array([proc[0+10*i] for i in range(len(proc)//10)]) # %10
print(processd[100])
processd = utils.normal(processd)
# expend the data to 3 channels
original = np.expand_dims(original,axis=3)
original = np.concatenate((original,original,original),axis=-1)
print(original.shape)
processd = np.expand_dims(processd,axis=3)
processd = np.concatenate((processd,processd,processd),axis=-1)
print(processd[100])

# reshape them to [batch,imgdata]
original = np.reshape(original,(original.shape[0],-1))/255
processd = np.reshape(processd,(processd.shape[0],-1))/255

test_size = int(original.shape[0]*0.1)
test_data = processd[0:test_size,:]
test_groud = original[0:test_size,:]
train_data = processd[test_size:-1,:]
train_groud = original[test_size:-1,:]
print(train_data.shape)

train_data = np.reshape(train_data, [-1, 100, 100, 3])
train_groud = np.reshape(train_groud,[-1,100,100,3])
print(train_data[100])
print(train_groud[100])
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(train_data[100],cmap='gray')
fig.add_subplot(1,2,2)
plt.imshow(train_groud[100],cmap='gray')
plt.show()

'''

# make a 100*100 patch dataset for a3d/p3d
'''
import nibabel as nib
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy import misc

global p_size
p_size = 100

# def k_downgrade(img):
#     kspace = np.fft.fftshift(np.fft.ifft2(img))
#     h,w = img.shape
#     kspace[0:round(h*0.9/2),:] = 0
#     kspace[h-round(h*0.9/2):h,:] = 0
#     out = np.fft.fft2(kspace)
#     return abs(out)

def k_downgrade(img,i):
    kspace = np.fft.fftshift(np.fft.ifft2(img))
    h,w = img.shape
    kspace[0:round(h*0.1*(10-i)/2),:] = 0
    kspace[h-round(h*0.1*(10-i)/2):h,:] = 0
    out = np.fft.fft2(kspace)
    return abs(out)


def zero_normal(img):
    mean = np.mean(img)
    std = np.std(img)
    normal = (img-mean)/std
    return normal

def normal(pixels):
    mins = np.min(pixels)
    maxs = np.max(pixels)
    pixels = (pixels-mins)*(255/(maxs-mins))
    return pixels


def get_patch(img,k_img,h_id,w_id):
    # ran = random.uniform(0,1)
    #ran>0.5:#adequatelly crop the image
    patch = img[h_id*p_size:(h_id+1)*p_size,w_id*p_size:(w_id+1):p_size]
    k_patch = k_img[h_id*p_size:(h_id+1)*p_size,w_id*p_size:(w_id+1):p_size]
    # else:
    #     patch = img[-1+(-1)*(h_id+1)*p_size:-1+(-1)*h_id*p_size,\
    #             -1+(-1)*(h_id+1)*p_size:-1+(-1)*h_id*p_size]
    #     k_patch = k_img[-1+(-1)*(h_id+1)*p_size:-1+(-1)*h_id*p_size,\
    #             -1+(-1)*(h_id+1)*p_size:-1+(-1)*h_id*p_size]
    return patch,k_patch
start_time = time.time()
top_path = 'D:\yexuehua\data\k_space_data\liver\P3D'
out_k_paths = 'D:/python/DPED-master/dped/blackberry/training_data/P3D_k'
out_o_paths = 'D:/python/DPED-master/dped/blackberry/training_data/P3D_o'
if not os.path.exists(out_k_paths):
    os.mkdir(out_k_paths)
if not os.path.exists(out_o_paths):
    os.mkdir(out_o_paths)
patients = os.listdir(top_path)
p = 0
for patient in patients:
    patient_path = os.path.join(top_path,patient)
    img = nib.load(patient_path)
    img_datas = np.array(img.dataobj)
    img_datas = np.transpose(img_datas,(2,1,0))
    for img in tqdm(img_datas):
        h,w = np.array(img.shape)//100
        k_img = k_downgrade(img,8)
        if np.min(img)<0 | np.max(img)==0:
            continue
        else:
            for i in range(h):
                for j in range(w):
                    orig = img[i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]
                    proc = k_img[i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]
                    proc = normal(proc)
                    orig = normal(orig)
                    misc.imsave(out_o_paths+'/'+str(p)+'.png',orig)
                    #using zero normalization to normalize the image
                    misc.imsave(out_k_paths+'/'+str(p)+'.png',proc)
                    p=p+1

# original = np.array(original)
# processd = np.array(processd)
# np.savez('MRI_dataset.npz',original=original,processd=processd)

# end_time = time.time()
# print("the whole time is",end_time-start_time)
# print(original.shape)
# print(processd.shape)
# fig = plt.figure()
# fig.add_subplot(1,2,1)
# plt.imshow(original[600],cmap='gray')
# fig.add_subplot(1,2,2)
# plt.imshow(processd[600],cmap='gray')
# plt.show()
# print(processd.shape)
'''


#make 80% k_space dcm
'''
import numpy as np
import pydicom
from scipy import misc
import os

top_path = r'D:\python\DPED-master\dped\blackberry\test_data\full_size_test_images\HA'
dcms = os.listdir(top_path)
out_path = os.path.join(top_path,'k_dcm')
if not os.path.exists(out_path):
    os.mkdir(out_path)

for img in dcms:
    img_path = os.path.join(top_path,img)
    imge = pydicom.read_file(img_path)
    data = imge.pixel_array
    #print(np.max(data))
    shape = data.shape
    #misc.imsave('brain_original.jpg',data)
    kspace_img = np.fft.fftshift(np.fft.ifft2(data))
    #maxk = np.max(kspace_img)
    #misc.imsave('brain_kspace.jpg',abs(kspace_img))
    out_paths = os.path.join(out_path,str((10-2)*10))
    if not os.path.exists(out_paths):
        os.mkdir(out_paths)
    removal = kspace_img
    removal[0:round(shape[0]*0.1*2/2),:] = 0
    removal[(shape[0]-round(shape[0]*0.1*2/2)):shape[0],:] = 0
    out = abs(np.fft.fft2(removal))
    out = np.uint16(out)
    out_name = str((10-2)*10)+'%_'+img
    out_name = os.path.join(out_paths,out_name)
    imge.PixelData = out.tobytes()
    imge.save_as(out_name)
'''

# convert to k-space
'''
import numpy as np
from scipy import misc
import numpy as np


img = misc.imread('brain_original.jpg',mode='L')
print(img)
fft_img = np.fft.fftshift(np.fft.fft2(img))
#fft_img = np.fft.ifft(img)
misc.imsave('brain_fft.png',20*np.log(np.abs(fft_img)))


'''

# create single 10-100% dcm by using dcm
'''
import numpy as np
import pydicom
from scipy import misc
import os
print('here')
imge = pydicom.read_file('IM135')
data = imge.pixel_array
#print(np.max(data))
shape = data.shape
#misc.imsave('brain_original.jpg',data)
kspace_img = np.fft.fftshift(np.fft.ifft2(data))
#maxk = np.max(kspace_img)
#misc.imsave('brain_kspace.jpg',abs(kspace_img))
for i in range(10):
    removal = kspace_img
    removal[0:round(shape[0]*0.1*i/2),:] = 0
    removal[(shape[0]-round(shape[0]*0.1*i/2)):shape[0],:] = 0
    out = abs(np.fft.fft2(removal))
    out = np.uint16(out)
    out_name = str((10-i)*10)+'%_'+'IM135.dcm'
    imge.PixelData = out.tobytes()
    imge.save_as(out_name)
'''

# # convert dcm to png
# '''
# import pydicom
# from scipy import misc
# import os
# import numpy as np
# 
# toppath = r'C:\Users\212725145\Desktop\dcm'
# imgs = os.listdir(toppath)
# for img in imgs:
#     imgpath = os.path.join(toppath,img)
#     dcm = pydicom.read_file(imgpath)
#     data = dcm.pixel_array
#     outname = os.path.join(toppath,img)
#     outname = outname.replace('.dcm','.png')
#     misc.imsave(outname,data)
# '''

# the usage of nifti
'''
import nibabel as nib
import numpy as np

imgs = nib.load('niidata/1082611_HA.nii')


print(imgs.shape)
'''

# calculate the number of images
''''''
import nibabel as nib
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy import misc


start_time = time.time()
top_path = 'D:\yexuehua\data\k_space_data\liver\A_P'
patients = os.listdir(top_path)
processd = []
original = []
c = 0
for patient in patients:
    patient_path = os.path.join(top_path,patient)
    img = nib.load(patient_path)
    img_datas = np.array(img.dataobj)
    img_datas = np.transpose(img_datas,(2,1,0))
    for img in tqdm(img_datas):
       c = c+1
print(c)




''''''


import pydicom
from scipy import misc
import os
import numpy as np

dcm = pydicom.read_file('your_dicom_name')
data = dcm.pixel_array

misc.imsave("output_jpg_name",data)
