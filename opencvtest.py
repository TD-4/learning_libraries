import cv2
import matplotlib.pyplot as plt
import numpy as np


def his(image):
    rows, cols = image.shape
    flat_gray = image.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    image = np.uint8(255 / (B - A + 0.1) * (image - A) + 0.5)
    return image

img = cv2.imread("images/IMG_3079.BMP", cv2.IMREAD_GRAYSCALE)
img = his(img)
# cv2.imshow("IMG", img)
# cv2.waitKey()


plt.imshow(img, cmap="gray")
# plt.text(226, 1, str(target.numpy()[0]))
plt.show()