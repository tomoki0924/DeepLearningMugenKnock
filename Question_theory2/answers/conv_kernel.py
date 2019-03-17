import numpy as np
import cv2
import matplotlib.pyplot as plt

height, width = 64, 64
k_channel = 4

img = cv2.imread("akahara_0001.jpg")
img = cv2.resize(img, (width, height))

np.random.seed(0)

kernels = np.random.normal(0, 0.01, [3, 3, k_channel])

out = np.zeros((height-2, width-2, 4), dtype=np.float32)

for y in range(height-2):
    for x in range(width-2):
        for ki in range(k_channel):
            out[y, x, ki] = np.sum(img[y:y+3, x:x+3] * kernels[..., ki])

for i in range(k_channel):
    plt.subplot(1,4,i+1)
    plt.imshow(out[..., i], cmap='gray')

plt.show()
