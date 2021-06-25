import numpy as np
import os
import cv2


def gen_mean_std(root_path=""):
    """
    获得mean & std
    """
    all_images = os.listdir(root_path)
    gray_channel = 0
    count = 0
    for img in all_images:
        img = cv2.imdecode(np.fromfile(os.path.join(root_path, img), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        H, W = np.asarray(img).shape
        gray_channel += np.sum(img)
        count += H * W
    gray_channel_mean = gray_channel / count

    gray_channel = 0
    count = 0
    for img in all_images:
        img = cv2.imdecode(np.fromfile(os.path.join(root_path, img), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img = img / 255.0
        H, W = np.asarray(img).shape
        gray_channel = gray_channel + np.sum((img - gray_channel_mean) ** 2)
        count += H * W
    gray_channel_std = np.sqrt(gray_channel / count)

    print("mean:", gray_channel_mean)
    print("std:", gray_channel_std)


if __name__ == "__main__":
    gen_mean_std(root_path="F:\\industry\\industry1_20\\industry1_20\\img")
