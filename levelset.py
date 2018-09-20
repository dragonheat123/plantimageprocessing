import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import color, io
import cv2



def grad(x):
    return np.array(np.gradient(x))


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def stopping_fun(x):
    return 1. / (1. + norm(grad(x))**2)


def stand_mat(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))



img = cv2.imread('C://Users//xxx//Desktop//smartfarm_data//panel1//img-100.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = stand_mat(img)*255
img = img - np.mean(img)
cv2.imshow('img',img)
plt.imshow(img)


sigma=2

# Smooth the image to reduce noise and separation between noise and edge becomes clear
img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)

F = stopping_fun(img_smooth)
cv2.imshow('f',F)

def default_phi(x):
    # Initialize surface phi at the border (5px from the border) of the image
    # i.e. 1 outside the curve, and -1 inside the curve
    phi = 1*np.ones(x.shape[:2])
    phi[5:-5, 5:-5] = 1.
    return phi

dt = 1.

phi = default_phi(img_smooth)

phi[100:300,100:300] = -1*np.ones(phi[100:300,100:300].shape[:2])
phi[200:400,200:400] = -1*np.ones(phi[200:400,200:400].shape[:2])
phi[400:600,100:300] = -1*np.ones(phi[400:600,100:300].shape[:2])

plt.imshow(phi)

for i in range(1000):
    dphi = grad(phi)
    dphi_norm = norm(dphi)

    dphi_t = F * dphi_norm

    phi = phi + dt * dphi_t
    cv2.imshow('phi',phi)
    cv2.waitKey(0)


