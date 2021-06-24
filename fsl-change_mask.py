from matplotlib import pyplot as plt
import numpy as np
import glob as glob
import cv2
import os
import shutil
import re


def split_image_mask(
        images_path="F:\\1.分割\\1-显示屏\\显示屏数据\\0顶伤\\original",
        masks_path="F:\\1.分割\\1-显示屏\\显示屏数据\\0顶伤\\label",
        output_imgs_path="F:\\lcd\\img",
        output_masks_path="F:\\lcd\\mask",
        English=True,
        pre_name="lcd0",
        fg_value=5,
        bg_value=255):
    """
    切割数据集中的图片和mask到一定size
    :param images_path:
    :param masks_path:
    :param output_imgs_path:
    :param output_masks_path:
    :return:
    """
    all_images = os.listdir(images_path)
    count = 0
    all_count = 0
    # 1、分割
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
            padding_w2 = size- W - padding_w1
            img = np.pad(img,((padding_h1, padding_h2), (padding_w1, padding_w2)), 'constant', constant_values=(0, 0))
            mask = np.pad(mask,((padding_h1, padding_h2), (padding_w1, padding_w2)), 'constant', constant_values=(255, 255))
            # print("padding one image, because size error")

        # 正方形resize到1192
        H = img.shape[0]
        W = img.shape[1]
        if H == W and H == 1192:
            pass
        elif H == W:
            img = cv2.resize(img, (1192, 1192), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (1192, 1192), interpolation=cv2.INTER_CUBIC)
        else:
            print("ERROR: size error")
            break

        # 2、修改mask
        mask = np.where(mask == bg_value, 0, fg_value)

        H = img.shape[0]
        W = img.shape[1]
        assert H == W and H == 1192
        # get 8*8 roi
        num = 8
        hh = int(H/8)
        ww = int(W/8)
        count_i = 0
        for x in range(0, num):
            for y in range(0, num):
                roiImg_img = img[x * hh:(x+1)*hh, y*ww:(y+1)*ww]
                roiImg_mask = mask[x * hh:(x+1)*hh, y*ww:(y+1)*ww]

                if len(set(np.asarray(roiImg_mask).flatten())) == 1:
                    continue
                if English:
                    output_img_path = os.path.join(output_imgs_path,
                                                   pre_name + str(count) + "_" + str(x) + "-" + str(y) + ".bmp")
                    output_mask_path = os.path.join(output_masks_path,
                                                    pre_name + str(count) + "_" + str(x) + "-" + str(y) + ".png")
                else:
                    output_img_path = os.path.join(output_imgs_path,
                                                   img_path.split(".")[0] + "_" + str(x)+"-"+str(y) + ".bmp")
                    output_mask_path = os.path.join(output_masks_path,
                                                   img_path.split(".")[0] + "_" + str(x)+"-"+str(y) + ".png")

                is_success_img, img_buff_arr = cv2.imencode('.bmp', roiImg_img)
                is_success_mask, mask_buff_arr = cv2.imencode('.png', roiImg_mask)
                img_buff_arr.tofile(output_img_path)
                mask_buff_arr.tofile(output_mask_path)
                count_i += 1
                all_count += 1
        count += 1

    print("deal {} images, and get {} images".format(count, all_count))


def change_mask_val(path="", fg_value=0, bg_value=255):
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
        # [set(np.asarray(cv2.imdecode(np.fromfile(os.path.join(path, i_p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)).flatten()) for i_p in all_images]   # 在“Evaluate debug”测试mask value有几个
        img = np.where(img == bg_value, 0, fg_value)
        assert len(set(np.asarray(img).flatten())) <= 2
        tmp2 = set(np.asarray(img).flatten())
        print("{} --> {}".format(tmp,tmp2))
        is_success_mask, mask_buff_arr = cv2.imencode('.png', img)
        mask_buff_arr.tofile(os.path.join(path, img_path))
        count += 1
    print("deal {} images".format(count))


def select_object2(
        img_path="F:\\lcd\\img",
        mask_path="F:\\lcd\\mask",
        output_img_path="F:\\lcd2\\img",
        output_mask_path="F:\\lcd2\\mask"):
    """
    检验 change_mask_val函数修改的mask值 是否符合要求
    :param path:
    :return:
    """
    all_images = os.listdir(img_path)
    count = 0
    count_d = 0
    for img_p in all_images:
        if re.search("ys_1_", img_p):
            continue
        img = cv2.imdecode(np.fromfile(os.path.join(img_path, img_p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.fromfile(os.path.join(mask_path, img_p[:-4]+".png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        count += 1

        if len(set(np.asarray(mask).flatten())) == 2:
            count_d += 1
            shutil.move(os.path.join(img_path, img_p), os.path.join(output_img_path, img_p).replace("ys", "lcd"))
            shutil.move(os.path.join(mask_path, img_p[:-4]+".png"), os.path.join(output_mask_path, img_p[:-4]+".png").replace("ys", "lcd"))
            print("get one: ", str(count))

    print("deal {} images, and {} has object".format(count, count_d))


def select_object(
        img_path="F:\\lcd\\img",
        mask_path="F:\\lcd\\mask",
        output_img_path="F:\\lcd2\\img",
        output_mask_path="F:\\lcd2\\mask"):
    """
    检验 change_mask_val函数修改的mask值 是否符合要求
    :param path:
    :return:
    """
    all_images = os.listdir(img_path)
    count = 0
    count_d = 0
    for img_p in all_images:
        img = cv2.imdecode(np.fromfile(os.path.join(img_path, img_p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.fromfile(os.path.join(mask_path, img_p[:-4]+".png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        count += 1

        if len(set(np.asarray(mask).flatten())) == 2:
            count_d += 1
            shutil.move(os.path.join(img_path, img_p), os.path.join(output_img_path, img_p))
            shutil.move(os.path.join(mask_path, img_p[:-4]+".png"), os.path.join(output_mask_path, img_p[:-4]+".png"))
            print("get one: ", str(count))

    print("deal {} images, and {} has object".format(count, count_d))


def get_img_mask(imgs_path="", masks_path="", masks_gray_path="", fg_value=1, bg_value=0):
    all_masks = os.listdir(masks_path)
    count = 0

    for mask_path in all_masks:
        mask = cv2.imread(os.path.join(masks_path, mask_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(os.path.join(imgs_path, mask_path.split(".")[0] + ".bmp"), cv2.IMREAD_GRAYSCALE)

        mask = np.where(mask == bg_value, 255, fg_value)
        print(set(np.asarray(mask).flatten()))
        assert len(set(np.asarray(mask).flatten())) <= 2
        # print(mask.shape)
        # print(img.shape)
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = np.hstack([img, mask])
        # plt.imshow(mask, cmap='gray')
        # plt.text(-50, -10, img_path)
        # plt.show()
        cv2.imwrite(os.path.join(masks_gray_path, mask_path), mask)
        count += 1

    print("deal {} images".format(count))


def get_images_path(path="", output_path="", value=""):
    """
    获取处理的当前类 的所有图片ID，将结果保存在classX.txt中
    :param path:
    :return:
    """
    all_images = os.listdir(path)
    count = 0
    output_path_ = os.path.join(output_path, "class{}.txt".format(value))
    with open(output_path_,"a+") as file:
        for img_path in all_images:
            file.write(img_path.split(".")[0]+"\n")
            count += 1

    print("deal {} images".format(count))


def yinshua_deal(images_path="", masks_path="",  output_imgs_path="",  output_masks_path="",
        English=True, pre_name="", fg_value=3, bg_value=0):
    """
    处理已经分割好的，但是没有修改mask值
    :param images_path:
    :param masks_path:
    :param output_imgs_path:
    :param output_masks_path:
    :param English:
    :param pre_name:
    :return:
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

        # 修改mask值
        mask = np.where(mask == bg_value, 0, fg_value)
        if English:
            output_img_path = os.path.join(output_imgs_path,
                                           pre_name + str(count) + ".bmp")
            output_mask_path = os.path.join(output_masks_path,
                                            pre_name + str(count) + ".png")
        else:
            output_img_path = os.path.join(output_imgs_path,
                                           img_path.split(".")[0] + ".bmp")
            output_mask_path = os.path.join(output_masks_path,
                                                    img_path.split(".")[0] + ".png")

        is_success_img, img_buff_arr = cv2.imencode('.bmp', img)
        is_success_mask, mask_buff_arr = cv2.imencode('.png', mask)
        img_buff_arr.tofile(output_img_path)
        mask_buff_arr.tofile(output_mask_path)
        count += 1

    print("deal {} images".format(count))


def pi_deal(images_path="", masks_path="", output_imgs_path="", output_masks_path="",output_masks_gray_path="",
        English=True, pre_name="pi_off_", bg_value=255, fg_value=4):
    """
    切割数据集中的图片和mask到一定size
    :param images_path:
    :param masks_path:
    :param output_imgs_path:
    :param output_masks_path:
    :return:
    """
    all_images = os.listdir(images_path)
    count = 0   # 统计一共处理多少张大图
    all_count = 0   # 统计一共得到多少张小图
    for img_path in all_images:
        img = cv2.imdecode(np.fromfile(os.path.join(images_path, img_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imdecode(np.fromfile(os.path.join(masks_path, img_path[:-4] + "_label.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if len(set(np.asarray(mask).flatten())) >= 5:   # 标注数值超过5个，舍弃这张照片
            print("skip, because marked is too 多 ", set(np.asarray(roiImg_mask).flatten()))
            continue

        # 3、修改mask值
        tmp = set(np.asarray(mask).flatten())
        mask = np.where(mask == bg_value, 0, fg_value)
        tmp2 = set(np.asarray(mask).flatten())
        print("{} --> {}".format(tmp, tmp2))

        H = img.shape[0]
        W = img.shape[1]
        assert H == 5000 and W == 8000
        hh = 125
        ww = 125
        # 1、切割
        for x in range(0, int(H/hh)):
            for y in range(0, int(W/ww)):
                roiImg_img = img[x * hh:(x+1)*hh, y*ww:(y+1)*ww]
                roiImg_mask = mask[x * hh:(x+1)*hh, y*ww:(y+1)*ww]

                # 2、选择只有object的图片
                if len(set(np.asarray(roiImg_mask).flatten())) == 1:  # 如果只有背景舍弃
                    continue

                # 4、合并img和mask
                roiImg_mask_gray = roiImg_mask
                roiImg_mask_gray = np.where(roiImg_mask_gray == 0, 255, fg_value)
                roiImg_mask_gray = np.hstack([roiImg_img, roiImg_mask_gray])

                if English: # 重命名
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


if __name__ == "__main__":
    # -------------------处理端子数据-------------------
    # 1、切割(略)
    # 2、修改mask值
    # change_mask_val(path="D:\\felixfu\\data\\FSL\\dz\\mask", fg_value=1, bg_value=0)
    # 3、选择有object的图片（略）
    # 4、测试： 获取有object的图片和mask是否正确, 测试mask与图片名字对应否，mask标注set长度是否为2，输出img和mask对应图
    # get_img_mask(imgs_path="D:\\felixfu\\data\\FSL\\dz\\img", masks_path="D:\\felixfu\\data\\FSL\\dz\\mask",
    #              masks_gray_path="D:\\felixfu\\data\FSL\\dz\\mask_gray", fg_value=1, bg_value=0)
    # 5、获得此类的classX.txt
    # get_images_path(path="D:\\felixfu\\data\\FSL\\dz\\img", output_path="D:\\felixfu\\data\\FSL\\dz", value=1)

    # -------------------处理crack数据-------------------
    # 1、切割(略)
    # 2、修改mask值
    # change_mask_val(path="D:\\felixfu\\data\\FSL\\crack\\mask", fg_value=2, bg_value=0)
    # 3、选择有object的图片
    # select_object(img_path="D:\\felixfu\\data\\FSL\\crack\\img", mask_path="D:\\felixfu\\data\\FSL\\crack\\mask",
    #               output_img_path="D:\\felixfu\\data\\FSL\\crack_\\img",
    #               output_mask_path="D:\\felixfu\\data\\FSL\\crack_\\mask")
    # 4、测试： 获取有object的图片和mask是否正确, 测试mask与图片名字对应否，mask标注set长度是否为2，输出img和mask对应图
    # get_img_mask(imgs_path="D:\\felixfu\\data\\FSL\\crack_\\img", masks_path="D:\\felixfu\\data\\FSL\\crack_\\mask",
    #              masks_gray_path="D:\\felixfu\\data\FSL\\crack_\\mask_gray", fg_value=2, bg_value=0)
    # 5、获得此类的classX.txt
    # get_images_path(path="D:\\felixfu\\data\\FSL\\crack_\\img",
    #                 output_path="D:\\felixfu\\data\\FSL\\crack_", value=2)

    # -------------------处理印刷数据-------------------
    # 1、切割(略)=印刷是移动数据 && 2、修改mask值
    # names = ["版污", "串墨", "刀丝", "晶点", "起皱", "甩墨", "拖尾", "油墨污染", "针孔"]
    # for i, name in enumerate(names):
    #     yinshua_deal(images_path="D:\\felixfu\\data\\ys\\{}\\def".format(name),
    #                  masks_path="D:\\felixfu\\data\\ys\\{}\\Label".format(name),
    #                  output_imgs_path="D:\\felixfu\\data\\FSL\\ys\\img",
    #                  output_masks_path="D:\\felixfu\\data\\FSL\\ys\\mask",
    #                  pre_name="ys_{}_".format(str(i)), fg_value=3, bg_value=255
    #                  )
    # 3、选择有object的图片（略）
    # 4、测试： 获取有object的图片和mask是否正确, 测试mask与图片名字对应否，mask标注set长度是否为2，输出img和mask对应图
    # get_img_mask(imgs_path="D:\\felixfu\\data\\FSL\\ys\\img", masks_path="D:\\felixfu\\data\\FSL\\ys\\mask",
    #              masks_gray_path="D:\\felixfu\\data\FSL\\ys\\mask_gray", fg_value=3, bg_value=0)
    # 5、获得此类的classX.txt
    # get_images_path(path="D:\\felixfu\\data\\FSL\\ys\\img",
    #                 output_path="D:\\felixfu\\data\\FSL\\ys", value=3)

    pass  # -------------------处理PI数据-------------------分割--修改mask--选择object--test--getlabel
    # 1、切割(略)=印刷是移动数据 && 2、修改mask值
    # names = ["PI--BLU OFF", "PI--BLU ON"]
    # for i, name in enumerate(names): #D:\felixfu\data\FSL_b\pi\PI--BLU OFF
    #    pi_deal(images_path="D:\\felixfu\\data\\FSL_b\\pi\\{}\\Original".format(name),
    #            masks_path="D:\\felixfu\\data\\FSL_b\\pi\\{}\\Label".format(name),
    #            output_imgs_path="D:\\felixfu\\data\\FSL\\pi-\\img",
    #            output_masks_path="D:\\felixfu\\data\\FSL\\pi-\\mask",
    #            output_masks_gray_path="D:\\felixfu\\data\\FSL\\pi-\\mask_gray",
    #            pre_name="lcd_{}_".format(str(i)), fg_value=4, bg_value=255
    #             )
    # 5、获得此类的classX.txt
    # get_images_path(path="D:\\felixfu\\data\\FSL\\pi-\\img",
    #                 output_path="D:\\felixfu\\data\\FSL\\pi-", value=4)

    pass  # -------------------处理显示屏数据-------------------
    # 1、切割图片。重复每个文件夹；2、修改mask值；
    # names = ["0顶伤", "1凹凸点", "2膜翘", "3破损", "4气泡", "5异物", "6印痕"]
    # for i, name in enumerate(names):
    #     split_image_mask(images_path="D:\\felixfu\\data\\FSL_b\\lcd\\{}\\Original".format(name),
    #                  masks_path="D:\\felixfu\\data\\FSL_b\\lcd\\{}\\Label".format(name),
    #                  output_imgs_path="D:\\felixfu\\data\\FSL\\lcd_\\img",
    #                  output_masks_path="D:\\felixfu\\data\\FSL\\lcd_\\mask",
    #                  pre_name="lcd2_{}_".format(str(i)),
    #                  fg_value=5,
    #                  bg_value=255
    #                  )
    # 2、修改mask值
    # change_mask_val(path="D:\\felixfu\\data\\FSL\\lcd_\\mask", fg_value=5, bg_value=255)
    # # 3、选择有object的图
    # select_object(
    #     img_path="D:\\felixfu\\data\\FSL\\lcd_\\img",
    #     mask_path="D:\\felixfu\\data\\FSL\\lcd_\\mask",
    #     output_img_path="D:\\felixfu\\data\\FSL\\lcd-\\img",
    #     output_mask_path="D:\\felixfu\\data\\FSL\\lcd-\\mask")
    # # 4、测试： 获取有object的图片和mask是否正确, 测试mask与图片名字对应否，mask标注set长度是否为2，输出img和mask对应图
    # get_img_mask(imgs_path="D:\\felixfu\\data\\FSL\\lcd_\\img",
    #               masks_path="D:\\felixfu\\data\\FSL\\lcd_\\mask",
    #              masks_gray_path="D:\\felixfu\\data\FSL\\lcd_\\mask_gray", fg_value=5, bg_value=0)
    # # 5、获得此类的classX.txt
    get_images_path(path="D:\\felixfu\\data\\FSL\\lcd_\\img",
                    output_path="D:\\felixfu\\data\\FSL\\lcd_", value=5)
