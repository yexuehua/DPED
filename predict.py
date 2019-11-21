import tensorflow as tf
from scipy import misc
import numpy as np
import sys
import cv2
from load_dataset import load_test_data, load_batch
from ssim import MultiScaleSSIM
from models import resnet
import utils
import vgg
from tqdm import tqdm
import os
import time
import nibabel as nib

batch_size = 32

#pixels = (255/(self.maxs-self.mins))*(pixels-self.mins)
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


def k_downgrade(img,i):
    kspace = np.fft.fftshift(np.fft.ifft2(img))
    h,w = img.shape
    kspace[0:round(h*0.1*(10-i)/2),:] = 0
    kspace[h-round(h*0.1*(10-i)/2):h,:] = 0
    out = np.fft.fft2(kspace)
    return abs(out)

# calculate the time
strat_time = time.time()
x = tf.placeholder(tf.float32,[None, None, None, 3])
enhanced = resnet(x)

# data process
data = nib.load(r'D:\python\DPED-master\dped\blackberry\test_data\full_size_test_images\HA\HA.nii')
imgs = np.array(data.dataobj)
imgs = np.transpose(imgs,(2,1,0))
originals = []
for img in imgs:
    original = normal(img)
    originals.append(original)
originals = np.array(originals)
originals = np.expand_dims(originals,axis=3)
originals = np.concatenate((originals,originals,originals),axis=-1)
print(np.max(originals))
print(originals.shape)
k_imgs = []
enhanced4d = []

for img in imgs:
    k_img = list(k_downgrade(img,8))
    k_img = normal(k_img)
    k_imgs.append(k_img)
k_imgs = np.array(k_imgs)
k_imgs = np.expand_dims(k_imgs,axis=3)
k_imgs = np.concatenate((k_imgs,k_imgs,k_imgs),axis=-1)
k_imgs = np.float16(k_imgs)/255
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,"saved_parameter/artery/blackberry_iteration_9000.ckpt")
    real_count = len(k_imgs)/batch_size
    count = int(np.ceil(real_count))
    for i in range(count):
        if (real_count-i)>0 and (real_count-i)<1:
            batch_data = k_imgs[i*batch_size:k_imgs.shape[0]]
        else:
            batch_data = k_imgs[i*batch_size:(i+1)*batch_size]
        enhanced3d = sess.run(enhanced,feed_dict={x:batch_data})
        enhanced3d = list(enhanced3d)
        enhanced4d.append(enhanced3d)
    enhanced4d = np.concatenate((enhanced4d[0],enhanced4d[1]),axis=0)
    ssmi = MultiScaleSSIM(originals,enhanced4d*255)

    loss_mse = np.mean(np.power(originals[0].flatten() - enhanced4d[0].flatten()*255, 2))
    loss_psnr = 10 * np.log10(255.0**2 / loss_mse)
    print(loss_psnr)
    print(ssmi)





