import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.fftpack import fftn, ifftn, fftshift


def gaussian_filter(k=3, sigma=1.0):
   arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
   x, y = np.meshgrid(arx, arx)
   filt = np.exp( -(1/2)*(np.square(x) + np.square(y))/np.square(sigma) )
   return filt/np.sum(filt)


def constrainedLeastSquare(img, H, P, gamma):
   aux = np.conjugate(H) / (np.square(np.abs(H)) + gamma * np.square(np.abs(P)))
   return np.multiply(aux, img)


'Taking inputs and opening image:'
filename = str(input()).strip()
img = imageio.imread(filename)
k = float(input())
sigma = float(input())
gamma = float(input())

f = gaussian_filter(k, sigma)  # filter

'padding the filter so that it has the same size of the image'
pad1 = (img.shape[0]//2)-f.shape[0]//2
fp = np.pad(f, (pad1, pad1-1), "constant",  constant_values=0)

'Computing the Fourrier Transform of the filter and the image'
f_fft = fftn(fp)  # Fourrier transform of the filter
img_fft = fftn(img)  # Fourrier transform of the image

max_value = np.amax(img_fft)

'Filtered image in the Fourier domain'
filtered_img = np.multiply(f_fft, img_fft)

'Normalizing image'
filtered_img = filtered_img/(filtered_img.max()/max_value)
max_value2 = np.amax(filtered_img)

'Restoring the blur using the Constrained Least Squares method'
p = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])  # laplacian operator
pad1 = (filtered_img.shape[0] // 2) - p.shape[0] // 2
p = np.pad(p, (pad1, pad1 - 1), "constant", constant_values=0)
P = fftn(p)

final_img = constrainedLeastSquare(filtered_img, f_fft, P, gamma)

'Normalizing and computing the inverse Transform'
final_img = final_img/(final_img.max()/max_value2)
final_img = np.real(fftshift(ifftn(final_img)))

'Standard Deviation'
print(np.round(np.std(final_img[:]), 1))
