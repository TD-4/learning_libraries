import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


def his(image):
    rows, cols = image.shape
    flat_gray = image.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    image = np.uint8(255 / (B - A + 0.1) * (image - A) + 0.5)
    return image


def cvt_g(path="E:\\Midle\\BlackDottrain", output="E:\\Test"):
    all_images = os.listdir(path)
    cout = 0
    for img_p in all_images:
        img = cv2.imdecode(np.fromfile(os.path.join(path,img_p),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        img = his(img)
        cv2.imwrite(os.path.join(output, img_p), img)
        cout +=1
        print("deal cout:", cout)
# cv2.imshow("IMG", img)
# cv2.waitKey()


# plt.imshow(img, cmap="gray")
# plt.text(226, 1, str(target.numpy()[0]))
# plt.show()

if __name__ == "__main__":
    img_path = "F:\\BDD_\\WLtrain\\T589M490BD41017770_画面1暗斑112141_674.812500_1071.625000_LABLEweakL_NG_SCORE0.700922.bmp"
    img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    img = his(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    print("here")
