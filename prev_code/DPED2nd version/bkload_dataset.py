from __future__ import print_function
import nibabel as nib
import numpy as np
import random
import utils
import time
import os

def make_dataset(dir_path):
    patients = os.listdir(dir_path)
    processd = []
    original = []
    for patient in patients:
        patient_path = os.path.join(dir_path,patient)
        img = nib.load(patient_path)
        img_datas = np.array(img.dataobj)
        img_datas = np.transpose(img_datas,(2,1,0))
        for img in img_datas:
            h,w = np.array(img.shape)//100
            k_img = utils.k_downgrade(img)
            for i in range(h):#crop image to 100 * 100 patches
                for j in range(w):
                    orig = img[i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]
                    proc = k_img[i*p_size:(i+1)*p_size,j*p_size:(j+1)*p_size]
                    orig = utils.zero_normal(orig)#using zero normalization to normalize the image
                    proc = utils.zero_normal(proc)
                    processd.append(proc)
                    original.append(orig)
    original = np.array(original)#convert list to ndarray
    processd = np.array(processd)
    np.savez('MRI_dataset.npz',original=original,processd=processd)

def load_data(dataset,test_rate):
    data = np.load(dataset)
    original = data['original']
    processd = data['processd']

    # expend the data to 3 channels
    original = np.expand_dims(original,axis=3)
    original = np.concatenate((original,original,original),axis=-1)
    processd = np.expand_dims(processd,axis=3)
    processd = np.concatenate((processd,processd,processd),axis=-1)

    # reshape them to [batch,imgdata]
    original = np.reshape(original,(original.shape[0],-1))
    processd = np.reshape(processd,(processd.shape[0],-1))

    test_size = int(original.shape[0]*test_rate)
    test_data = processd[0:test_size,:]
    test_groud = original[0:test_size,:]
    train_data = processd[test_size:-1,:]
    train_groud = original[test_size:-1,:]

    return test_data, test_groud, train_data, train_groud


def load_batch(phone, dped_dir, TRAIN_SIZE, IMAGE_SIZE):

    train_directory_phone = dped_dir + str(phone) + '/training_data/' + str(phone) + '/'
    train_directory_dslr = dped_dir + str(phone) + '/training_data/canon/'

    NUM_TRAINING_IMAGES = len([name for name in os.listdir(train_directory_phone)
                               if os.path.isfile(os.path.join(train_directory_phone, name))])

    # if TRAIN_SIZE == -1 then load all images

    if TRAIN_SIZE == -1:
        TRAIN_SIZE = NUM_TRAINING_IMAGES
        TRAIN_IMAGES = np.arange(0, TRAIN_SIZE)
    else:
        TRAIN_IMAGES = np.random.choice(np.arange(0, NUM_TRAINING_IMAGES), TRAIN_SIZE, replace=False)

    train_data = np.zeros((TRAIN_SIZE, IMAGE_SIZE))
    train_answ = np.zeros((TRAIN_SIZE, IMAGE_SIZE))

    i = 0
    for img in TRAIN_IMAGES:

        I = np.asarray(imageio.imread(train_directory_phone + str(img) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        train_data[i, :] = I

        I = np.asarray(imageio.imread(train_directory_dslr + str(img) + '.jpg'))
        I = np.float16(np.reshape(I, [1, IMAGE_SIZE])) / 255
        train_answ[i, :] = I

        i += 1
        if i % 100 == 0:
            print(str(round(i * 100 / TRAIN_SIZE)) + "% done", end="\r")

    return train_data, train_answ
