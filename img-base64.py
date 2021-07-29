# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import os
import base64
import random
import shutil
import cv2
import re
import numpy as np
from itertools import chain
from glob import glob
import json
from PIL import Image

from labelme import __version__
from labelme.logger import logger
from labelme import PY2
from labelme import QT4
from labelme import utils

import base64
import contextlib
import io
import json
import os.path as osp


def change_json(root_path=""):
    """
    修改json文件，使在增强图标注的json中某些字段修改为原图的内容
    :param root_path: 文件夹，其中有原图(bmp),增强图(png),json文件(json)
    :return: 原图(bmp),json文件(json)
    """
    # 获取所有原图 路径
    bmp_path = [path for path in os.listdir(root_path) if re.search(".bmp", path)]
    for bmp_p in bmp_path:
        png_p = bmp_p[:-3] + "png"      # 增强图 名称
        json_p = bmp_p[:-3] + "json"    # json标注文件名称

        # 获得图片的二进制内容
        def load_image_file(filename):
            try:
                image_pil = Image.open(filename)
            except IOError:
                logger.error("Failed opening image file: {}".format(filename))
                return

            # apply orientation to image according to exif
            image_pil = utils.apply_exif_orientation(image_pil)

            with io.BytesIO() as f:
                ext = osp.splitext(filename)[1].lower()
                if PY2 and QT4:
                    format = "PNG"
                elif ext in [".jpg", ".jpeg"]:
                    format = "JPEG"
                else:
                    format = "PNG"
                image_pil.save(f, format=format)
                f.seek(0)
                return f.read()

        imageData = load_image_file(os.path.join(root_path, bmp_p))
        imageData = base64.b64encode(imageData).decode("utf-8")     # 获得原图 二进制内容

        # 修改json中的内容
        with open(os.path.join(root_path, json_p), 'r',encoding='UTF-8') as load_f:
            label_json = json.load(load_f)
        label_json['imageData'] = imageData
        label_json['imagePath'] = bmp_p
        # 保存json
        with open(os.path.join(root_path, json_p), "w", encoding='UTF-8') as dump_f:
            json.dump(label_json, dump_f, ensure_ascii=False, indent=2)

        # 删除原来的png
        os.remove(os.path.join(root_path, png_p))


def bmp2png(root_path="", origin_path= r"F:\Data\LandingAI\20210726-origin"):
    bmp_images = [img_p for img_p in os.listdir(root_path) if img_p[-3:] == "bmp"]
    for img_p in bmp_images:
        # file_path = os.path.join(root_path, img_p)
        # img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        # cv2.imencode('.png', img)[1].tofile(os.path.join(root_path, img_p[:-3] + "png"))
        os.rename(os.path.join(root_path, img_p), os.path.join(root_path, img_p[:-3] + "png"))

        shutil.copy(os.path.join(origin_path, img_p), os.path.join(root_path, img_p))


def json2img(root_path=""):
    """
    用于判断png图片二进制 和 labelme标注获得的图片二进制，是否一样
    :param root_path:
    :return:
    """
    bmp_path = [path for path in os.listdir(root_path) if re.search(".bmp", path)]
    for bmp_p in bmp_path:
        png_p = bmp_p[:-3] + "png"
        json_p = bmp_p[:-3] + "json"

        # 获得图片的二进制内容
        def load_image_file(filename):
            try:
                image_pil = Image.open(filename)
            except IOError:
                logger.error("Failed opening image file: {}".format(filename))
                return

            # apply orientation to image according to exif
            image_pil = utils.apply_exif_orientation(image_pil)

            with io.BytesIO() as f:
                ext = osp.splitext(filename)[1].lower()
                if PY2 and QT4:
                    format = "PNG"
                elif ext in [".jpg", ".jpeg"]:
                    format = "JPEG"
                else:
                    format = "PNG"
                image_pil.save(f, format=format)
                f.seek(0)
                return f.read()

        imageData = load_image_file(os.path.join(root_path, png_p))
        imageData = base64.b64encode(imageData).decode("utf-8")

        # 获取json文件，并转成图片存储
        with open(os.path.join(root_path, json_p), 'r') as load_f:
            label_json = json.load(load_f)
        img = base64.b64decode(label_json['imageData'])
        with open(os.path.join(root_path, 'res.png'), 'wb') as f:
            f.write(img)

        assert imageData == label_json['imageData']


if __name__ == "__main__":

    # json2img(root_path="C:\\Users\\F04396\\Desktop\\t1")  # 测试使用&学习使用
    # bmp2png(r"F:\Data\LandingAI\标后", r"F:\Data\LandingAI\20210726-origin")
    # change_json(root_path=r"F:\Data\LandingAI\标后")     # 修改json文件中的imageData和imagePath字段