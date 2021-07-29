import cv2
import numpy as np
import os
import re


def grey_scale(image):
    # img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gray = image
    rows, cols = img_gray.shape
    flat_gray = img_gray.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    print('A = %d,B = %d' % (A, B))
    output = np.uint8(255 / (B - A) * (img_gray - A) + 0.5)
    return output


def check_after(path=""):
    all_images = os.listdir(path)
    all_json = sorted([img for img in all_images if re.search(".json", img)])
    all_img = sorted([img for img in all_images if re.search(".bmp", img)])
    for img, json  in zip(all_img, all_json):
        print("img:", img[:-4])
        print("json:", json[:-5])
        assert img[:-4] == json[:-5]


if __name__ == "__main__":
    # -----Gray 2 histogram
    # path = r"F:\Data\LandingAI\筛前"
    # output = r"F:\Data\LandingAI\标后"
    # all_images = os.listdir(os.path.join(path))
    # for img_p in all_images:
    #     if img_p[-4:] != ".bmp": continue
    #     file_path = os.path.join(path, img_p)
    #     src = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    #     # src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #     result = grey_scale(src)
    #     # cv2.imshow('src', src)
    #     # cv2.imshow('result', result)
    #     cv2.imencode('.bmp', result)[1].tofile(os.path.join(output, img_p))

    # ---- check after labeled
    # check_after(r"F:\Data\LandingAI\标后")
