'Ana Laura Chioca Vieira NUSP: 9866531'

import warnings
import numpy as np
import imageio

def DFT2D(f):

    ' create empty array of complex coefficients '
    F = np.zeros(f.shape, dtype=np.complex64)
    n, m = f.shape[0:2]
    ' creating indices for x, to compute multiplication using numpy (f*exp) '
    x = np.arange(n)

    for u in np.arange(n):
        for v in np.arange(m):
            for y in np.arange(m):
                F[u, v] += np.sum(f[:, y] * np.exp((-1j * 2 * np.pi) * (((u * x) / n) + ((v * y) / m))))

    return F / np.sqrt(n * m)

def IDFT2D(f):

    ' create empty array of complex coefficients '
    F = np.zeros(f.shape, dtype=np.complex64)
    n, m = f.shape[0:2]
    ' creating indices for x, to compute multiplication using numpy (f*exp) '
    x = np.arange(n)

    for u in np.arange(n):
        for v in np.arange(m):
            for y in np.arange(m):
                F[u, v] += np.sum(f[:, y] * np.exp((1j * 2 * np.pi) * (((u * x) / n) + ((v * y) / m))))

    return F / np.sqrt(n * m)

'Taking inputs and opening image:'
filename = str(input()).strip()
img = imageio.imread(filename)
T = float(input())

'Computing the fourier transform of the image:'
img_fft = DFT2D(img)

'Find the second peak of the fourier spectrum'
p2 = np.unique(abs(img_fft))[-2]

'Setting to 0 all coeficients below T% (|img_fft| < p2*T)'
a, b = img_fft.shape
num = 0

for i in range (a):
    for j in range (b):
        if abs(img_fft[i, j]) < p2*T:
            img_fft[i, j] = 0
            num+=1

'Computing the inverse fourier transform'
new_img = abs(IDFT2D(img_fft))
warnings.filterwarnings("ignore")

print("Threshold=%.4f" % (p2*T))
print("Filtered Coefficients=%d" % num)
print("Original Mean=%.2f" % np.mean(img))
print("New Mean=%.2f" % np.mean(new_img))