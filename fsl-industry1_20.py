from matplotlib import pyplot as plt
import numpy as np
import glob as glob
import cv2
import os
import shutil
import re


def crack_deal(images_path="", masks_path="",  output_imgs_path="",  output_masks_path="",output_masks_gray_path="",
                 English=True, pre_name="", fg_value=20, bg_value=0):
    """
    1、修改mask值
    2、切割（已切割好，略）
    3、筛选有object的内容（已筛选好，略）
    4、get images & masks
    """
    all_images = os.listdir(images_path)
    count = 0
    for img_path in all_images:
        img = cv2.imdecode(np.fromfile(os.path.join(images_path, img_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.fromfile(os.path.join(masks_path, img_path[:-4]+".png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        count += 1
        # 非正方形，退出
        H = img.shape[0]
        W = img.shape[1]
        assert H == W
        # 3、筛选
        if len(set(np.asarray(mask).flatten())) == 1:
            print("Only one label, skip!!!!")
            continue

        # 1、修改mask值
        tmp = set(np.asarray(mask).flatten())
        mask = np.where(mask == bg_value, 0, fg_value)
        tmp2 = set(np.asarray(mask).flatten())
        print("img_path:{} set {} --> {}".format(img_path, tmp, tmp2))

        # 2、略

        # 4、合并img和mask
        roiImg_mask_gray = mask
        roiImg_mask_gray = np.where(roiImg_mask_gray == 0, 255, fg_value)
        roiImg_mask_gray = np.hstack([img, roiImg_mask_gray])

        if English:
            output_img_path = os.path.join(output_imgs_path,
                                           pre_name + str(count) + ".bmp")
            output_mask_path = os.path.join(output_masks_path,
                                            pre_name + str(count) + ".png")
            output_mask_gray_path = os.path.join(output_masks_gray_path,
                                                 pre_name + str(count) + ".png")
        else:
            output_img_path = os.path.join(output_imgs_path,
                                           img_path.split(".")[0] + ".bmp")
            output_mask_path = os.path.join(output_masks_path,
                                                    img_path.split(".")[0] + ".png")
            output_mask_gray_path = os.path.join(output_masks_gray_path,
                                                 pre_name + str(count) + ".png")

        is_success_img, img_buff_arr = cv2.imencode('.bmp', img)
        is_success_mask, mask_buff_arr = cv2.imencode('.png', mask)
        is_success_mask, mask_gray_buff_arr = cv2.imencode('.png', roiImg_mask_gray)
        img_buff_arr.tofile(output_img_path)
        mask_buff_arr.tofile(output_mask_path)
        mask_gray_buff_arr.tofile(output_mask_gray_path)
    print("deal {} images".format(count))


def get_images_path(path="", output_path="", fg_value=""):
    """
    获取处理的当前类 的所有图片ID，将结果保存在classX.txt中
    :param path:
    :return:
    """
    all_images = os.listdir(path)
    count = 0
    output_path_ = os.path.join(output_path, "class{}.txt".format(fg_value))
    with open(output_path_,"a+") as file:
        for img_path in all_images:
            file.write(img_path.split(".")[0]+"\n")
            count += 1

    print("deal {} images".format(count))


def change_mask_val(path="", fg_value=0, fg_value_new=255):
    """
    修改每张mask图片的mask值，以适应小样本训练的要求
    :param path:
    :return:
    """
    all_images = os.listdir(path)
    count = 0
    for img_path in all_images:
        img = cv2.imdecode(np.fromfile(os.path.join(path, img_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        tmp = set(np.asarray(img).flatten())
        img = np.where(img == fg_value, fg_value_new, img)
        tmp2 = set(np.asarray(img).flatten())
        print("{} --> {}".format(tmp, tmp2))
        is_success_mask, mask_buff_arr = cv2.imencode('.png', img)
        mask_buff_arr.tofile(os.path.join(path, img_path))
        count += 1
    print("deal {} images".format(count))


def test_all(masks_path="", class_path="", class_value=0):
    count = 0
    with open(class_path, 'r') as file:
        for line in file.readlines():
            mask = cv2.imdecode(np.fromfile(os.path.join(masks_path, line.strip() + ".png"), dtype=np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            mask_set = set(np.asarray(mask).flatten())
            if len(mask_set)==3:
                print(mask_set)
                continue
            count += 1
            print(count, " = ", mask_set)
            assert mask_set - set({0}) == set({class_value})


if __name__ == "__main__":
    # industry1_8是正确的，不需要修改
    # industry9_19需要修改mask值
    # change_mask_val(path="F:\\industry\\industry1_20\\industry9_17\\ys1\\mask",
    #                 fg_value=1, fg_value_new=9)
    # change_mask_val(path="F:\\industry\\industry1_20\\industry9_17\\ys2\\mask",
    #                 fg_value=2, fg_value_new=10)
    # change_mask_val(path="F:\\industry\\industry1_20\\industry9_17\\ys3\\mask",
    #                 fg_value=3, fg_value_new=11)
    # change_mask_val(path="F:\\industry\\industry1_20\\industry9_17\\ys4\\mask",
    #                 fg_value=4, fg_value_new=12)
    # change_mask_val(path="F:\\industry\\industry1_20\\industry9_17\\ys5\\mask",
    #                 fg_value=5, fg_value_new=13)
    # change_mask_val(path="F:\\industry\\industry1_20\\industry9_17\\ys6\\mask",
    #                 fg_value=6, fg_value_new=14)
    # change_mask_val(path="F:\\industry\\industry1_20\\industry9_17\\ys7\\mask",
    #                 fg_value=7, fg_value_new=15)
    # change_mask_val(path="F:\\industry\\industry1_20\\industry9_17\\ys8\\mask",
    #                 fg_value=8, fg_value_new=16)
    # change_mask_val(path="F:\\industry\\industry1_20\\industry9_17\\ys9\\mask",
    #                 fg_value=9, fg_value_new=17)

    # change_mask_val(path="F:\\industry\\industry1_20\\industry18_19\\dz1\\mask",
    #                 fg_value=10, fg_value_new=18)
    # change_mask_val(path="F:\\industry\\industry1_20\\industry18_19\\dz1\\mask",
    #                 fg_value=11, fg_value_new=19)
    # change_mask_val(path="F:\\industry\\industry1_20\\industry18_19\\dz2\\mask",
    #                 fg_value=10, fg_value_new=18)
    # change_mask_val(path="F:\\industry\\industry1_20\\industry18_19\\dz2\\mask",
    #                 fg_value=11, fg_value_new=19)

    # crack
    # 1、修改mask值, 2、切割图片。重复每个文件夹；3、获得object图；4、得到images和mask对应图
    # crack_deal(images_path="F:\\industry\\industry1_20\\landing_data_0610\\img",
    #            masks_path="F:\\industry\\industry1_20\\landing_data_0610\\cls",
    #            output_imgs_path="F:\\industry\\industry1_20\\industry20\\img",
    #            output_masks_path="F:\\industry\\industry1_20\\industry20\\mask",
    #            output_masks_gray_path="F:\\industry\\industry1_20\\industry20\\mask_gray",
    #            pre_name="crack_", fg_value=20, bg_value=0)
    # # 5、获得此类的classX.txt
    # get_images_path(path="F:\\industry\\industry1_20\\industry20\\img",
    #                 output_path="F:\\industry\\industry1_20\\industry20", fg_value=20)

    # test all
    path = "F:\\industry\\industry1_20\\industry1_20\\classes"
    mask_path = "F:\\industry\\industry1_20\\industry1_20\\mask"
    for i in range(1, 21):
        p = os.path.join(path, "class{}.txt".format(str(i)))
        test_all(masks_path=mask_path, class_path=p, class_value=i)
