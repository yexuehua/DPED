import tensorflow as tf
import numpy as np
from models import resnet
import os
import time
import pydicom
from tqdm import tqdm

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

# calculate the time
strat_time = time.time()
x = tf.placeholder(tf.float32,[None, None, 3])
x_imge = tf.expand_dims(x,axis=0)
enhanced = resnet(x_imge)


with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,"models/blackberry_iteration_7000.ckpt")
    # data process
    norpro = nomalprocess()
    test_dir = "dicom_data/"
    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

    for photo in tqdm(test_photos):
        img = pydicom.read_file(test_dir+photo)
        imge = img.pixel_array
        imge = norpro.normal(imge)
        # print(np.min(imge))
        imge = np.expand_dims(imge, axis=2)
        imge = np.concatenate((imge, imge, imge), axis=-1)
        # image = np.float16(misc.imresize(imge, res_sizes[phone])) / 255
        # img = misc.imread(test_dir + photo,mode='L')
        # img = np.expand_dims(img, axis=2)
        # img = np.concatenate((img, img, img), axis=-1)

        high,wid,_= imge.shape
        IMAGE_SIZE = high*wid*3

        image = np.float16(imge) / 255
        #image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
        #print(image_crop.shape)

        #image_crop_2d = np.reshape(image, [1, high,wid,3])
        #print(image_crop_2d.shape)

        # get enhanced image
        enhanced_2d = sess.run(enhanced, feed_dict={x: image})
        enhanced_image = np.reshape(enhanced_2d, [high, wid, 3])
        enhanced_image = enhanced_image[:,:,0]
        enhanced_image = enhanced_image*255
        enhanced_image = normal(enhanced_image)
        enhanced_image = norpro.denomal(enhanced_image)
        # print(np.max(enhanced_image))
        # print(np.min(enhanced_image))
        # image = image[:,:,0]
        # before_after = np.hstack((image, enhanced_image))
        enhanced_image = np.int16(enhanced_image)
        img.PixelData = enhanced_image.tobytes()
        if photo[-4:] == ".dcm" or photo[-6:] == ".dicom":
            photo = photo[0:-4]
        img.save_as("dicom_output/"+ "enhanced_" +photo+ ".dcm")
    print('process finished!')
endtime = time.time()
print("the whole time is %f" %(endtime-strat_time))
