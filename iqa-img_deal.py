import os
import shutil
import cv2
import numpy as np
import torchvision
import re
from PIL import Image


def deal_xt(path=""):
    # 重命名文件夹
    # all_folders = os.listdir(path)
    # for i, folder in enumerate(all_folders):
    #     os.rename(os.path.join(path, folder), os.path.join(path, str(i)))

    # 处理整个文件夹中的每个
    all_folders = sorted(os.listdir(path),key=lambda d:int(d))
    for folder in all_folders:
        folder_p = os.path.join(path, folder)

        # 处理本文件夹，所得所有目标图片
        all_files = [img for img in os.listdir(folder_p) if img[0] in str(set(range(0,10)))]

        # 处理label.txt文件，获得level和images. 并删除其他不用的图片
        if "label.txt" not in os.listdir(folder_p):
            print("{} folder not label.txt".format(folder))
            break
        labels = {}
        test = []
        with open(os.path.join(folder_p, "label.txt")) as f:
            for line in f:
                labels[line.strip().split("-")[1]] = line.strip().split("-")[0]  # 图片id:等级:
                test.append(line.strip().split("-")[1])
            if set(('0','1','2','3','4','5','6','7','8','9','0')) -set(test) !=set():
                print("{} folder lable.txt error!!!".format(folder))
                continue
        to_del = set(os.listdir(os.path.join(folder_p))) - set(all_files)

        # 处理每个文件,重命名
        for file in all_files:
            o = str(labels[file.split("-")[0]]) + ".bmp"
            os.rename(os.path.join(folder_p,file),
                      os.path.join(folder_p, o))

        # 删除其他不用的图片
        for file in to_del:
            os.remove(os.path.join(folder_p, file))


def gen_score(path="", output=""):
    all_folders = os.listdir(path)      # 获得所有文件夹
    label_path = os.path.join(output, "labels.txt")
    with open(label_path, "a+") as file:
        for folder in all_folders:  # 遍历每个文件夹
            all_images = os.listdir(os.path.join(path, folder))
            for image in all_images:    # 遍历每张图片
                id_image = image.split(".")[0]
                score = round((int(id_image) / 10) ** 2, 4)
                img = cv2.imread(os.path.join(path, folder, image))
                img_name = path.split("\\")[-1] + "_" + folder + "_" + str(score) + "_.bmp"
                cv2.imwrite(os.path.join(output, img_name), img)
                file.write("{} {}\n".format(img_name, score))


def padding_img(path=""):
    all_images = [img_p for img_p in os.listdir(path) if re.search(".bmp", img_p)]
    for img_p in all_images:
        image = Image.open(os.path.join(path, img_p))
        H, W, _ = np.array(image).shape

        size = max(H, W)
        if H != W:
            padding_h1 = int((size - H) / 2)
            padding_h2 = size - H - padding_h1
            padding_w1 = int((size - W) / 2)
            padding_w2 = size - W - padding_w1
            image = torchvision.transforms.Pad([padding_w1, padding_h1, padding_w2, padding_h2],
                                               padding_mode="reflect")(image)
            # image.show()
            image.save(os.path.join(path, img_p))


def re_name(path=""):
    # 重命名一个文件夹
    all_folders = os.listdir(path)
    for i, folder in enumerate(all_folders):
        all_images = os.listdir(os.path.join(path, folder))
        for j, img in enumerate(all_images):
            os.rename(os.path.join(path, folder, img), os.path.join(path, folder, str(j) + ".bmp"))
        os.rename(os.path.join(path, folder), os.path.join(path, str(i)))


def labelimg(path=""):
    all_folders = os.listdir(path)
    for folder in all_folders:
        all_images = sorted(os.listdir(os.path.join(path, folder)))
        for i, img_p in enumerate(all_images):
            os.rename(os.path.join(path, folder, img_p),
                      os.path.join(path, folder, str(i+1)+".bmp"))


def gen_score2(path="", output=""):
    all_folders = os.listdir(path)      # 获得所有文件夹
    label_path = os.path.join(output, "labels.txt")
    with open(label_path, "a+") as file:
        for folder in all_folders:  # 遍历每个文件夹
            all_images = os.listdir(os.path.join(path, folder))
            for image in all_images:    # 遍历每张图片
                id_image = image.split(".")[0]
                score = round(((int(id_image)-1) / 10 + 0.01) ** 2, 4)
                score = min(score, 1.0)
                img = cv2.imread(os.path.join(path, folder, image))
                img_name = path.split("\\")[-1] + "_" + folder + "_" + str(score) + "_.bmp"
                cv2.imwrite(os.path.join(output, img_name), img)
                file.write("{} {}\n".format(img_name, score))


def fix_name_label(path=""):
    count = 0
    all_images = os.listdir(path)
    label_path = os.path.join(path, "labels.txt")
    with open(label_path, "a+") as file:
        for img_p in all_images:
            img_p_split = img_p.split("_")  # 20210423_0_0.01_.bmp
            print(img_p_split)
            score = float(img_p_split[2])
            if score >= 1.0:
                img_o = img_p_split[0] + "_" + img_p_split[1] + "_1.0_.bmp"
                os.rename(os.path.join(path, img_p),
                          os.path.join(path, img_o))
                file.write("{} {}\n".format(img_o, round(score, 4)))
            file.write("{} {}\n".format(img_p, score))
            count += 1
    print("deal {} images".format(count))

if __name__ == "__main__":
    # 1、通过label.txt，排序文件，并重新命名
    # deal_xt(path="F:\\Data\\IQA\\20210423-\\20210423")
    # 2、重命名文件，并生成score
    # gen_score(path="F:\\Data\\IQA\\level\\20210423", output="F:\\Data\\IQA\\level\\dst_images")
    # 3、填充图像，把长方形转成正方形
    # padding_img(path="F:\\Data\\IQA\\level\\dst_images")

    # 4-1、使用预测结果，标注数据, 重命名
    # labelimg(path="F:\\20210424")
    # 4-2、使用4-1的数据生成数据集
    # gen_score2(path="F:\\20210424", output="F:\\dst_images_20210424")
    # 4-3、padding
    # padding_img(path="F:\\dst_images_20210424")
    # 4-4、修改score
    fix_name_label(path="F:\\Data\\IQA\\level\\dst_images")


