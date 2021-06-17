import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


def change_mask_val(path="E:\\迅雷下载\\landing_data_0610\\mask"):
    """
    修改每张mask图片的mask值，以适应小样本训练的要求
    :param path:
    :return:
    """
    all_images = os.listdir(path)
    count = 0
    for img_path in all_images:
        img = cv2.imdecode(np.fromfile(os.path.join(path,img_path),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        img = np.where(img >= 1, 2, 0)
        cv2.imwrite(os.path.join("mask", img_path), img)
        count += 1
    print("deal {} images".format(count))


def test_mask_val(path="E:\\迅雷下载\\landing_data_0610\\mask"):
    """
    检验 change_mask_val函数修改的mask值 是否符合要求
    :param path:
    :return:
    """
    all_images = os.listdir(path)
    count = 0
    for img_path in all_images:
        img = cv2.imdecode(np.fromfile(os.path.join(path, img_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        count += 1

        assert len(set(np.asarray(img).flatten())) <= 2

        if img.any() > 2:
            print("error!!!")
            break
        if count % 200 == 0:
            img = np.where(img ==2, 255, 0)
            plt.imshow(img, cmap='gray')
            plt.text(-50, -10, img_path)
            plt.show()
    print("deal {} images".format(count))


def get_images_path(path="E:\\迅雷下载\\crack\\\\img"):
    """
    获取处理的当前类 的所有图片ID，将结果保存在classX.txt中
    :param path:
    :return:
    """
    all_images = os.listdir(path)
    count = 0
    with open("class2.txt","a+") as file:
        for img_path in all_images:
            file.write(img_path.split(".")[0]+"\n")
            count += 1

    print("deal {} images".format(count))


if __name__ == "__main__":
    # change_mask_val()
    # test_mask_val("mask")
    get_images_path()