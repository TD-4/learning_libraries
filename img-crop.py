"""
crop img from image
"""
from matplotlib import pyplot as plt
import numpy as np
import glob as glob
import cv2
import os
import shutil
import re
from matplotlib import pyplot as plt


def crop_img(root_path="", x1=0, y1=0, x2=0, y2=0):
    all_images = os.listdir(root_path)
    for img_path in [img for img in all_images if re.search(".bmp", img)]:
        img = cv2.imdecode(np.fromfile(os.path.join(root_path, img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        img = img[x1:x2, y1:y2]
        plt.imshow(img)
        plt.show()
        output_img_path = os.path.join(root_path, img_path.split(".")[0] + "_" + ".bmp")
        is_success_mask, img_buff_arr = cv2.imencode('.bmp', img)
        img_buff_arr.tofile(output_img_path)


if __name__ == "__main__":
    crop_img(root_path="F:\\复核图像\\测试2", x1=620, y1=850, x2=1000, y2=1150)
