import os
import shutil

# all_images = os.listdir(r"F:\不规则增强结果\脏污\增强图")
#
# for img_p in all_images:
#     if img_p not in os.listdir(os.path.join(r"F:\不规则")): print("Error, not this image {}".format(img_p))
#     shutil.copy(os.path.join(r"F:\不规则", img_p),
#                 os.path.join(r"F:\不规则增强结果\脏污\原图"))
path = r"F:\Data\lcd\midle\20210628\TODO"
all_folders = os.listdir(path)
for folder in all_folders:
    for i, imgp in enumerate(os.listdir(os.path.join(path, folder))):
        os.rename(os.path.join(path, folder, imgp),
                  os.path.join(path, folder, "20210628_{}_{}.bmp".format(folder, i)))