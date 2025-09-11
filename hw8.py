import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('/home/rayhan/Downloads/image7.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(img, (5,5), 1.4)

# Perform Canny Edge Detection
edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

# Display results
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Blurred Image")
plt.imshow(blurred, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Canny Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.show()
