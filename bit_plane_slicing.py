import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(16, 9))

img = cv2.imread('/home/rayhan/Downloads/image7.jpg', cv2.IMREAD_GRAYSCALE)
slices = []

for i in range(8):
    plane = (img >> i) & 1
    plane = plane * 255
    slices.append(plane)


plt.subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

for i, slice in enumerate(slices):
    plt.subplot(3, 3, i + 2)
    plt.imshow(slice, cmap='gray')
    plt.title(f'Slice {i}')

plt.tight_layout()
plt.show()
