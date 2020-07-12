import numpy as np
import imageio
import matplotlib
from matplotlib import colors
from skimage import morphology
from skimage.morphology import disk


def RMSE(img, original_img):
    num_el = original_img.shape[0]*original_img.shape[1]
    rmse = np.sqrt(np.sum(np.square(np.subtract(img.astype(np.float64), original_img.astype(np.float64))))/num_el)
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

def composition(img, size):
    img = matplotlib.colors.rgb_to_hsv(img)
    imageH = img[:, :, 0]

    'Normalizing image'
    imageH = (imageH - imageH.min()) * 255 / (imageH.max() - imageH.min())

    'Gradient = Dilation - Erosion'
    imageH_dilation = morphology.dilation(imageH, morphology.disk(size)).astype(np.uint8)
    imageH_erosion = morphology.erosion(imageH, morphology.disk(size)).astype(np.uint8)
    imageH_gradient = imageH_dilation - imageH_erosion

    'Normalizing image'
    #imageH_gradient = (imageH_gradient - imageH_gradient.min()) * 255 / (imageH_gradient.max() - imageH_gradient.min())

    img_mod = np.zeros(img.shape, dtype='float')
    img_mod[:, :, 0] = imageH_gradient
    img_mod[:, :, 1] = morphology.opening(imageH, morphology.disk(size)).astype(np.uint8)
    img_mod[:, :, 2] = morphology.closing(imageH, morphology.disk(size)).astype(np.uint8)

    return img_mod


'Taking inputs and opening image:'
filename = str(input()).strip()
img = imageio.imread(filename)
k = int(input())
option = int(input())

if option == 1:
    img_final = RGBOpening(img, k)

if option == 2:
    img_final = composition(img, k)

if option == 3:
    img_middle = RGBOpening(img, 2*k)
    img_final = composition(img_middle, k)

RMSE(img_final, img)


