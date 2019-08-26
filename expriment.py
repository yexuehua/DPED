from scipy import misc
import pydicom
import numpy as np

def normal(pixels):
    high,wid = pixels.shape
    mins = np.min(pixels)
    maxs = np.max(pixels)
    for i in range(high):
        for j in range(wid):
            pixels[i][j] = int((pixels[i][j]-mins)*(255/(maxs-mins)))
    return pixels

img = pydicom.read_file(r'D:\yexuehua\download\MR_Enhance\1_75815001\T1FLAIR_08.dcm')
#inputs = img.pixel_array
#inputs = normal(inputs)
#misc.imsave('IM42.jpg',inputs)
a = img.PixelData
b = bytearray(a)
print(img)#dir(img))

# print(b.decode())


