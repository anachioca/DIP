import numpy as np
import imageio
from skimage import morphology
from skimage.color import rgb2hsv
from skimage.morphology import disk


def RMSE(img, original_img):
    rmse = np.sqrt(np.mean((img - original_img) ** 2))
    print("%.4f" % rmse)


def RGBOpening(image, size):
    imageR = image[:, :, 0]
    imageG = image[:, :, 1]
    imageB = image[:, :, 2]

    img_mod = np.zeros(img.shape, dtype='float')
    img_mod[:, :, 0] = morphology.opening(imageR, morphology.disk(size)).astype(np.uint8)
    img_mod[:, :, 1] = morphology.opening(imageG, morphology.disk(size)).astype(np.uint8)
    img_mod[:, :, 2] = morphology.closing(imageB, morphology.disk(size)).astype(np.uint8)

    return img_mod


'Taking inputs and opening image:'
filename = str(input()).strip()
img = imageio.imread(filename)
k = int(input())
option = int(input())

if option == 1:
    img_final = RGBOpening(img, k)

if option == 3:
    img_final = RGBOpening(img, k)


intensity = img.sum(axis=2)
intensity_mod = img_final.sum(axis=2)
RMSE(intensity_mod, intensity)
RMSE(img_final, img)


