# Ana Laura Chioca Vieira 9866531
# Assignment 1 - SCC0251 2020/1


import numpy as np
import imageio


# Function that compares the reference image and the modified image
# and prints the root square error (RSE) in the screen.
def RSE(img_mod, img):
    error = np.square(np.subtract(img_mod.astype(np.float32), img.astype(np.float32)))
    rse = np.sqrt(np.sum(error))
    print("%.4f" % rse)


# Function that applies the Inversion transformation to the image
def Inversion(img):
    img_mod = 255 - img
    return img_mod


# Function that applies the Contrast Modulation transformation to the image
def ContrastModulation(img):
    c = int(input())
    d = int(input())
    const = ((d - c) / 255)
    img_mod = ((img) * const) + c
    return img_mod


# Function that applies the Logarithmic transformation to the image
def LogarithmicFunction(img):
    c_scale = 255 / (np.log2(1 + 255))
    img_mod = (c_scale * np.log2(1 + img.astype(np.int32))).astype(np.uint8)
    return img_mod


# Function that applies the Gamma Adjustment transformation to the image
def GammaAdjustment(img):
    w = int(input())
    lambd = float(input())
    img_mod = w * (img ** lambd)
    return img_mod


# Receiving the inputs and choosing the method:

filename = str(input()).rstrip()
method = int(input())
save = int(input())

if method == 1:
    img = imageio.imread(filename)
    img_mod = Inversion(img)
    RSE(img_mod, img)

    if save == 1:
        imageio.imwrite("output_img.png", img_mod)


if method == 2:
    img = imageio.imread(filename)
    img_mod2 = ContrastModulation(img)
    RSE(img_mod2, img)

    if save == 1:
        imageio.imwrite("output_img.png", img_mod2)


if method == 3:
    img = imageio.imread(filename)
    img_mod3 = LogarithmicFunction(img)
    RSE(img_mod3, img)

    if save == 1:
        imageio.imwrite("output_img.png", img_mod3)


if method == 4:
    img = imageio.imread(filename)
    img_mod4 = GammaAdjustment(img)
    RSE(img_mod4, img)

    if save == 1:
        imageio.imwrite("output_img.png", img_mod4)




