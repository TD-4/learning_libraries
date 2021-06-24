from matplotlib import pyplot as plt
import numpy as np
import glob as glob
import cv2
import os
import shutil
import re


def yinshua_deal(images_path="", masks_path="",  output_imgs_path="",  output_masks_path="",output_masks_gray_path="",
                 English=True, pre_name="", fg_value=255, bg_value=255):
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
        mask = cv2.imdecode(np.fromfile(os.path.join(masks_path, img_path[:-4]+"_label.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # 非正方形，退出
        H = img.shape[0]
        W = img.shape[1]
        assert H == W

        # 1、修改mask值
        tmp = set(np.asarray(mask).flatten())
        mask = np.where(mask == bg_value, 0, fg_value)
        tmp2 = set(np.asarray(mask).flatten())
        print("img_path:{} set {} --> {}".format(img_path, tmp, tmp2))

        # 2、3、略

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

        count += 1

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


def select_dz(img_path="", mask_path="", output_img_path1="", output_mask_path1="",output_img_path2="", output_mask_path2=""):
    """
    从端子中获取类别1和2
    """
    all_images = os.listdir(img_path)
    count = 0
    count_d = 0
    count_dd = 0
    for img_p in all_images:
        img = cv2.imdecode(np.fromfile(os.path.join(img_path, img_p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.fromfile(os.path.join(mask_path, img_p[:-4]+".png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        count += 1
        # 1、删除只有背景的图
        if len(set(np.asarray(mask).flatten())) == 1:
            continue
        # 2、修改mask
        tmp = set(np.asarray(mask).flatten())
        mask = np.where(mask == 1, 10, mask)
        mask = np.where(mask == 2, 11, mask)
        if set(np.asarray(mask).flatten()) & set({10}) and set(np.asarray(mask).flatten()) & set({11}):  # 里面有10和11
            count_dd += 1
            print("this img has 2 label")

        if set(np.asarray(mask).flatten()) & set({10}):    # 里面有10
            count_d += 1
            shutil.copy(os.path.join(img_path, img_p), os.path.join(output_img_path1, img_p))
            is_success_mask, mask_buff_arr = cv2.imencode('.png', mask)
            mask_buff_arr.tofile(os.path.join(output_mask_path1, img_p[:-4] + ".png"))
            # shutil.copy(os.path.join(mask_path, img_p[:-4] + ".png"),
            #             os.path.join(output_mask_path1, img_p[:-4] + ".png"))

        if set(np.asarray(mask).flatten()) & set({11}):    # 里面有11
            count_d += 1
            shutil.copy(os.path.join(img_path, img_p), os.path.join(output_img_path2, img_p))
            is_success_mask, mask_buff_arr = cv2.imencode('.png', mask)
            mask_buff_arr.tofile(os.path.join(output_mask_path2, img_p[:-4] + ".png"))
            # shutil.copy(os.path.join(mask_path, img_p[:-4] + ".png"),
            #             os.path.join(output_mask_path2, img_p[:-4] + ".png"))


    print("deal {} images, and {} has object, 共同拥有{} images".format(count, count_d, count_dd))


def get_img_mask(imgs_path="", masks_path="", masks_gray_path="", fg_value=10, bg_value=0):
    all_masks = os.listdir(masks_path)
    count = 0

    for mask_path in all_masks:
        mask = cv2.imread(os.path.join(masks_path, mask_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(os.path.join(imgs_path, mask_path.split(".")[0] + ".bmp"), cv2.IMREAD_GRAYSCALE)

        mask_gray = mask
        mask_gray = np.where(mask_gray == 10, 230, mask_gray)
        mask_gray = np.where(mask_gray == bg_value, 255, mask_gray)
        mask_gray = np.hstack([img, mask_gray])
        cv2.imwrite(os.path.join(masks_gray_path, mask_path), mask_gray)
        count += 1
    print("deal {} images".format(count))


if __name__ == "__main__":
    # # 1、修改mask值, 2、切割图片。重复每个文件夹；3、获得object图；4、得到images和mask对应图
    # names = ["版污", "串墨", "刀丝", "晶点", "起皱", "甩墨", "拖尾", "油墨污染", "针孔"]
    # value = 9
    # yinshua_deal(images_path="D:\\industry9_17\\ys\\{}\\def".format(names[value-1]),
    #              masks_path="D:\\industry9_17\\ys\\{}\\Label".format(names[value-1]),
    #              output_imgs_path="D:\\industry9_17\\ys_\\ys{}\\img".format(str(value)),
    #              output_masks_path="D:\\industry9_17\\ys_\\ys{}\\mask".format(str(value)),
    #              output_masks_gray_path="D:\\industry9_17\\ys_\\ys{}\\mask_gray".format(str(value)),
    #              pre_name="ys{}_".format(str(value)),
    #              fg_value=value,
    #              bg_value=255)
    # # 5、获得此类的classX.txt
    # get_images_path(path="D:\\industry9_17\\ys_\\ys{}\\img".format(str(value)),
    #                 output_path="D:\\industry9_17\\ys_\\ys{}".format(str(value)), fg_value=value)

    # 1、端子获取两类: 修改mask、分割（无）、选择有object的物体（无）
    # select_dz(img_path="F:\\industry\\industry9_19\\dz\\img",
    #           mask_path="F:\\industry\\industry9_19\\dz\\mask",
    #           output_img_path1="F:\\industry\\industry9_19\\ys_dz_\\dz1\\img",
    #           output_mask_path1="F:\\industry\\industry9_19\\ys_dz_\\dz1\\mask",
    #           output_img_path2="F:\\industry\\industry9_19\\ys_dz_\\dz2\\img",
    #           output_mask_path2="F:\\industry\\industry9_19\\ys_dz_\\dz2\\mask"
    #           )
    # 2、获得image和mask
    # get_img_mask(imgs_path="F:\\industry\\industry9_19\\ys_dz_\\dz1\\img",
    #              masks_path="F:\\industry\\industry9_19\\ys_dz_\\dz1\\mask",
    #              masks_gray_path="F:\\industry\\industry9_19\\ys_dz_\\dz1\\mask_gray", fg_value=10, bg_value=0)
    # get_img_mask(imgs_path="F:\\industry\\industry9_19\\ys_dz_\\dz2\\img",
    #              masks_path="F:\\industry\\industry9_19\\ys_dz_\\dz2\\mask",
    #              masks_gray_path="F:\\industry\\industry9_19\\ys_dz_\\dz2\\mask_gray", fg_value=10, bg_value=0)
    # 5、获得此类的classX.txt
    get_images_path(path="F:\\industry\\industry9_19\\ys_dz_\\dz1\\img",
                    output_path="F:\\industry\\industry9_19\\ys_dz_\\dz1\\", fg_value=10)
    get_images_path(path="F:\\industry\\industry9_19\\ys_dz_\\dz2\\img",
                    output_path="F:\\industry\\industry9_19\\ys_dz_\\dz2\\", fg_value=11)
