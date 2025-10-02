import cv2
import numpy as np
import matplotlib.pyplot as plt

#================= Structuring Elements =================
def get_structuring_elements():
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

    # Diamond SE (manually created)
    diamond = np.array([[0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0],
                        [0, 0, 1, 0, 0]], dtype=np.uint8)

    return {"Rectangular": rect, "Elliptical": ellipse, 
            "Cross": cross, "Diamond": diamond}

#================= Custom Morphological Functions =================
def custom_erode(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=255)
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            if np.all(region[kernel==1] == 255):
                result[i, j] = 255
            else:
                result[i, j] = 0
    return result

def custom_dilate(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2
    padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kh, j:j+kw]
            if np.any(region[kernel==1] == 255):
                result[i, j] = 255
            else:
                result[i, j] = 0
    return result

def custom_open(img, kernel):
    return custom_dilate(custom_erode(img, kernel), kernel)

def custom_close(img, kernel):
    return custom_erode(custom_dilate(img, kernel), kernel)

def custom_tophat(img, kernel):
    return cv2.subtract(img, custom_open(img, kernel))

def custom_blackhat(img, kernel):
    return cv2.subtract(custom_close(img, kernel), img)

#================= Main Function with Plotting =================
def main():
    # Load grayscale image
    img = cv2.imread("/home/rayhan/Downloads/image7.jpg", 0)   # Replace with your own image
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    struct_elems = get_structuring_elements()

    for name, kernel in struct_elems.items():
        print(f"\n==== {name} Structuring Element ====")

        # ---- OpenCV Built-in ----
        erosion_cv = cv2.erode(img_bin, kernel, iterations=1)
        dilation_cv = cv2.dilate(img_bin, kernel, iterations=1)
        opening_cv = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
        closing_cv = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)
        tophat_cv = cv2.morphologyEx(img_bin, cv2.MORPH_TOPHAT, kernel)
        blackhat_cv = cv2.morphologyEx(img_bin, cv2.MORPH_BLACKHAT, kernel)

        # ---- Custom ----
        erosion_custom = custom_erode(img_bin, kernel)
        dilation_custom = custom_dilate(img_bin, kernel)
        opening_custom = custom_open(img_bin, kernel)
        closing_custom = custom_close(img_bin, kernel)
        tophat_custom = custom_tophat(img_bin, kernel)
        blackhat_custom = custom_blackhat(img_bin, kernel)

        # ---- Plot Results ----
        titles = ["Original", "Erosion", "Dilation", "Opening", "Closing", "Top Hat", "Black Hat"]
        images_cv = [img_bin, erosion_cv, dilation_cv, opening_cv, closing_cv, tophat_cv, blackhat_cv]
        images_custom = [img_bin, erosion_custom, dilation_custom, opening_custom, closing_custom, tophat_custom, blackhat_custom]

        fig, axes = plt.subplots(2, 7, figsize=(20, 6))
        fig.suptitle(f"Morphological Operations with {name} Structuring Element", fontsize=16)

        for i in range(7):
            # Row 1: OpenCV
            axes[0, i].imshow(images_cv[i], cmap="gray")
            axes[0, i].set_title(f"CV - {titles[i]}", fontsize=8)
            axes[0, i].axis("off")

            # Row 2: Custom
            axes[1, i].imshow(images_custom[i], cmap="gray")
            axes[1, i].set_title(f"Custom - {titles[i]}", fontsize=8)
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(f"{name}_Morphology_Comparison.png")
        plt.show()

if __name__ == "__main__":
    main()
