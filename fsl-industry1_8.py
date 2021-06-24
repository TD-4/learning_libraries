from matplotlib import pyplot as plt
import numpy as np
import glob as glob
import cv2
import os
import shutil
import re


def split_image_mask(images_path="", masks_path="", output_imgs_path="", output_masks_path="", output_masks_gray_path="",
                     English=True, pre_name="", fg_value=255,  bg_value=255):
    """
    1、修改mask值
    2、切割
    3、筛选有object的内容
    4、get images & masks
    """
    all_images = os.listdir(images_path)
    count = 0
    all_count = 0

    for img_path in all_images:
        img = cv2.imdecode(np.fromfile(os.path.join(images_path, img_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.fromfile(os.path.join(masks_path, img_path[:-4]+"_label.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if len(set(np.asarray(mask).flatten())) >= 5:
            print("标注的值多")
            continue
        # 非正方形，则填充变成正方形
        H = img.shape[0]
        W = img.shape[1]
        size = max(H, W)
        if H != W:
            padding_h1 = int((size-H)/2)
            padding_h2 = size - H - padding_h1
            padding_w1 = int((size-W)/2)
            padding_w2 = size - W - padding_w1
            img = np.pad(img, ((padding_h1, padding_h2), (padding_w1, padding_w2)), 'constant', constant_values=(0, 0))
            mask = np.pad(mask, ((padding_h1, padding_h2), (padding_w1, padding_w2)), 'constant', constant_values=(255, 255))
            # print("padding one image, because size error")

        # 正方形resize到1192
        H = img.shape[0]
        W = img.shape[1]
        if H == W and H == 1192:
            pass
        elif H == W:
            img = cv2.resize(img, (1192, 1192), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (1192, 1192), interpolation=cv2.INTER_CUBIC)    # 插值后，set(mask)可能不是2和255了，值变多了
        else:
            print("ERROR: size error")
            break

        # 1、修改mask
        tmp = set(np.asarray(mask).flatten())
        mask = np.where(mask == bg_value, 0, fg_value)
        tmp2 = set(np.asarray(mask).flatten())
        print("img_path:{} set {} --> {}".format(img_path, tmp, tmp2))
        H = img.shape[0]
        W = img.shape[1]
        assert H == W and H == 1192
        # get 8*8 roi
        num = 4
        hh = int(H/num)
        ww = int(W/num)
        # 2、分割
        for x in range(0, num):
            for y in range(0, num):
                roiImg_img = img[x * hh:(x+1)*hh, y*ww:(y+1)*ww]
                roiImg_mask = mask[x * hh:(x+1)*hh, y*ww:(y+1)*ww]

                # 3、选择只有object的图片
                if len(set(np.asarray(roiImg_mask).flatten())) == 1:  # 如果只有背景舍弃
                    continue

                # 4、合并img和mask
                roiImg_mask_gray = roiImg_mask
                roiImg_mask_gray = np.where(roiImg_mask_gray == 0, 255, fg_value)
                roiImg_mask_gray = np.hstack([roiImg_img, roiImg_mask_gray])

                if English:
                    output_img_path = os.path.join(output_imgs_path,
                                                   pre_name + str(count) + "_" + str(x) + "-" + str(y) + ".bmp")
                    output_mask_path = os.path.join(output_masks_path,
                                                    pre_name + str(count) + "_" + str(x) + "-" + str(y) + ".png")
                    output_mask_gray_path = os.path.join(output_masks_gray_path,
                                                         pre_name + str(count) + "_" + str(x) + "-" + str(y) + ".png")
                else:
                    output_img_path = os.path.join(output_imgs_path,
                                                   img_path.split(".")[0] + "_" + str(x)+"-"+str(y) + ".bmp")
                    output_mask_path = os.path.join(output_masks_path,
                                                   img_path.split(".")[0] + "_" + str(x)+"-"+str(y) + ".png")
                    output_mask_gray_path = os.path.join(output_masks_gray_path,
                                                         pre_name + str(count) + "_" + str(x) + "-" + str(y) + ".png")

                is_success_img, img_buff_arr = cv2.imencode('.bmp', roiImg_img)
                is_success_mask, mask_buff_arr = cv2.imencode('.png', roiImg_mask)
                is_success_mask, mask_gray_buff_arr = cv2.imencode('.png', roiImg_mask_gray)
                img_buff_arr.tofile(output_img_path)
                mask_buff_arr.tofile(output_mask_path)
                mask_gray_buff_arr.tofile(output_mask_gray_path)
                all_count += 1
        count += 1

    print("deal {} images, and get {} images".format(count, all_count))


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


if __name__ == "__main__":
    # 1、修改mask值, 2、切割图片。重复每个文件夹；3、获得object图；4、得到images和mask对应图
    names = ["0顶伤", "1凹凸点", "2膜翘", "3破损", "4气泡", "5异物", "6印痕"]
    split_image_mask(images_path="D:\\lcd\\6印痕\\Original",
                     masks_path="D:\\lcd\\6印痕\\\\Label",
                     output_imgs_path="D:\\lcd_\\lcd7\\img",
                     output_masks_path="D:\\lcd_\\lcd7\\mask",
                     output_masks_gray_path="D:\\lcd_\\lcd7\\mask_gray",
                     pre_name="lcd7_",
                     fg_value=7,
                     bg_value=255)
    # 5、获得此类的classX.txt
    get_images_path(path="D:\\lcd_\\lcd7\\img",
                    output_path="D:\\lcd_\\lcd7\\", fg_value=7)


