import os
import json
import shutil
import distutils
import distutils.dir_util
import re
from imagededup.methods import PHash, CNN, DHash




def count_split(input_path="F:\\Data\\深度学习",
                output="F:\\Data\\lcd\\20210703"):
    all_folders = os.listdir(input_path)

    for folder in all_folders:  # 遍历所有人
        print(folder, ":", end="")
        if os.path.isfile(os.path.join(input_path, folder)):    # 文件跳过，压缩包什么的
            print()
            continue

        count = 0
        for root, dirs, files in os.walk(os.path.join(input_path, folder)):
            if "BlackDot" in root.split("\\")[-1]:
                if not os.path.exists(os.path.join(output, "BlackDot")):
                    os.mkdir(os.path.join(output, "BlackDot"))
                distutils.dir_util.copy_tree(os.path.join(root), os.path.join(output, "BlackDot"))

            if "BrightDot" in root.split("\\")[-1]:
                if not os.path.exists(os.path.join(output, "BrightDot")):
                    os.mkdir(os.path.join(output, "BrightDot"))
                distutils.dir_util.copy_tree(os.path.join(root), os.path.join(output, "BrightDot"))

            if "Edge" in root.split("\\")[-1]:
                if not os.path.exists(os.path.join(output, "Edge")):
                    os.mkdir(os.path.join(output, "Edge"))
                distutils.dir_util.copy_tree(os.path.join(root), os.path.join(output, "Edge"))

            if "Midle" in root.split("\\")[-1]:
                if not os.path.exists(os.path.join(output, "Midle")):
                    os.mkdir(os.path.join(output, "Midle"))
                distutils.dir_util.copy_tree(os.path.join(root), os.path.join(output, "Midle"))

            count += len(files)
        print(count)


def del_ori(path=""):
    all_folders = os.listdir(path)
    for folder in all_folders:
        all_images = os.listdir(os.path.join(path, folder))
        for img in all_images:
            if re.search("Ori", img):
                os.remove(os.path.join(path, folder, img))


def del_leisi_img(path="", pre=""):
    phasher = DHash()
    folders = os.listdir(path=path)
    # 重命名所有文件
    for folder in folders:
        if os.path.isfile(os.path.join(path, folder)):continue
        if os.listdir(os.path.join(path, folder)) == []:continue
        all_images = os.listdir(os.path.join(path, folder))
        for i, image in enumerate(all_images):
            os.rename(os.path.join(path, folder, image),
                      os.path.join(path, folder, pre + "_" + folder + "_" + str(i) + ".bmp"))

    # 获得所有要删除的文件名
    for folder in folders:
        if os.path.isfile(os.path.join(path, folder)): continue
        if os.listdir(os.path.join(path, folder)) == []: continue
        duplicates = phasher.find_duplicates_to_remove(image_dir=os.path.join(path, folder),
                                                       max_distance_threshold=4,
                                                       outfile='my_duplicates_{}.json'.format(folder))

    for folder in folders:
        if os.path.isfile(os.path.join(path, folder)): continue
        if os.listdir(os.path.join(path, folder)) == []: continue
        with open("./my_duplicates_{}.json".format(folder), 'r') as load_f:
            load_dict = json.load(load_f)
        for name in load_dict:
            os.remove(os.path.join(path, folder, name))
        os.remove("./my_duplicates_{}.json".format(folder))


if __name__ == "__main__":
    # 1、从云文档中拷贝文件过来
    # count_split(input_path=r"F:\缺陷小图",
    #             output=r"F:\Data\lcd\origin\20210713")
    # # 2、删除Ori图片
    # del_ori(path=r"F:\Data\lcd\origin\20210713\Edge")
    # del_ori(path=r"F:\Data\lcd\origin\20210713\Midle")

    # 3、删除相似图片
    del_leisi_img(path=r"F:\Data\lcd\origin\20210713\Edge", pre="20210713")
    del_leisi_img(path=r"F:\Data\lcd\origin\20210713\Midle", pre="20210713")

    # del_leisi_img(path="F:\\Data\\lcd\\midle\\20210628", pre="20210628")
    #



